from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Normal, Independent

import network.init_func as init


# ======================== Modules for AutoEncoder=======================================================
class ProbabilisticModel(nn.Module):
    def __init__(self, latent_dim, stochastic):
        super(ProbabilisticModel, self).__init__()
        self.latent_dim = latent_dim
        self.stochastic = stochastic
        self.latent_space = None
        self.gaussian_module = AxisAlignedConvGaussian(latent_dim)

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    def sampling(self, batch_size=1):
        if self.training:
            latent_value = self.latent_space.rsample((batch_size,))
        else:
            latent_value = self.latent_space.sample((batch_size,))
        if batch_size == 1:
            latent_value = latent_value.squeeze(0)
        return latent_value


class AxisAlignedConvGaussian(nn.Module):
    def __init__(self, latent_dim):
        super(AxisAlignedConvGaussian, self).__init__()
        self.latent_dim = latent_dim

    def forward(self, mu_log_sigma):
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma), validate_args=False), 1)
        return dist


class GaussianModule(ProbabilisticModel):
    def __init__(self, latent_dim, stochastic=True):
        super(GaussianModule, self).__init__(latent_dim, stochastic)
        self.default_batch_size = 1
        self._build()

    def _build(self):
        self.dummy_param = nn.Parameter(torch.empty(0))
        standard_latent_input = torch.zeros([self.default_batch_size, 2 * self.latent_dim]).cuda()
        self.latent_space = self.gaussian_module(standard_latent_input)

    def sampling_by_mu_sigma(self, mu_log_sigma, batch_size=1):
        latent_space = self.gaussian_module(mu_log_sigma)
        latent_value = latent_space.sample((batch_size,)).squeeze(1)
        return latent_value


class LinearAffineModule(nn.Module):
    def __init__(self, input_dim, output_dim, batchnorm=True, nonlinearity=nn.LeakyReLU(0.02)):
        super(LinearAffineModule, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(input_dim, output_dim), nonlinearity)

    def forward(self, x):
        return self.model(x)


class ProbabilisticEncoder(ProbabilisticModel):
    def __init__(self, model_func, latent_dim, stochastic=True, activation=False, clip_value=1.0):
        super(ProbabilisticEncoder, self).__init__(latent_dim, stochastic)
        self.model_func = model_func
        self.activation = activation
        self.clip_value = clip_value
        self.fc_layer = nn.Linear(model_func.n_outputs, 2 * latent_dim if stochastic else latent_dim)
        if activation:
            self.relu = nn.Tanh()

    def forward(self, x):
        encoding = self.model_func(x).view(x.size(0), -1)
        # We only want the mean of the resulting hxw image
        latent_variables = self.fc_layer(encoding)
        if self.activation:
            latent_variables = self.relu(latent_variables)
        latent_variables = self.clip_value * latent_variables

        if self.stochastic:
            self.latent_space = self.gaussian_module(latent_variables)
        return latent_variables


class LinearDecoder(nn.Module):
    """
    Adjust from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    """

    def __init__(self, latent_dim, output_shape):
        super(LinearDecoder, self).__init__()

        self.output_shape = output_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 16, normalize=False),
            *block(16, 64),
            *block(64, 128),
            nn.Linear(128, int(np.prod(output_shape))),
        )

    def forward(self, latent_variables):
        img = self.model(latent_variables)
        img = img.view(img.shape[0], -1)
        return img


class CovDecoder28x28(nn.Module):
    # Adjust from: https://github.com/AMLab-Amsterdam/DIVA/blob/4c5282a8e54feee01626f5e8a54595ea570ac169/paper_experiments/rotated_mnist/supervised/model_diva.py
    def __init__(self, input_dim, output_shape):
        super(CovDecoder28x28, self).__init__()

        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(8)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(24)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.de3 = nn.Sequential(nn.Conv2d(256, output_shape[0], kernel_size=1, stride=1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)
        out = self.de1(out)
        out = self.up2(out)
        out = self.de2(out)
        out = self.de3(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], *self.output_shape)
        return out


class CovDecoder84x84(nn.Module):
    # Adjust from: https://github.com/AMLab-Amsterdam/DIVA/blob/4c5282a8e54feee01626f5e8a54595ea570ac169/paper_experiments/rotated_mnist/supervised/model_diva.py
    def __init__(self, input_dim, output_shape):
        super(CovDecoder84x84, self).__init__()

        self.output_shape = output_shape
        self.fc1 = nn.Sequential(nn.Linear(input_dim, 1024, bias=False), nn.BatchNorm1d(1024), nn.ReLU())
        self.up1 = nn.Upsample(16)
        self.de1 = nn.Sequential(nn.ConvTranspose2d(64, 128, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(128), nn.ReLU())
        self.up2 = nn.Upsample(40)
        self.de2 = nn.Sequential(nn.ConvTranspose2d(128, 256, kernel_size=5, stride=1, padding=0, bias=False),
                                 nn.BatchNorm2d(256), nn.ReLU())
        self.up3 = nn.Upsample(80)
        self.de3 = nn.Sequential(
            nn.ConvTranspose2d(256, output_shape[0], kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(output_shape[0]), nn.ReLU())

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = out.view(-1, 64, 4, 4)
        out = self.up1(out)
        out = self.de1(out)
        out = self.up2(out)
        out = self.de2(out)
        out = self.up3(out)
        out = self.de3(out)
        out = self.sigmoid(out)
        out = out.view(out.shape[0], *self.output_shape)
        return out


class BranchDecoder(ProbabilisticModel):
    """
    DIVA module
    Use for reconstructing domain_label and class label
    Adjust from: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    """

    def __init__(self, input_dim, output_dim, stochastic=True, init_type='xavier'):
        super(BranchDecoder, self).__init__(output_dim, stochastic)
        if stochastic:
            output_dim = output_dim * 2

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.ReLU())
            return layers

        self.model = nn.Sequential(
            *block(input_dim, output_dim, normalize=True),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, latent_variables):
        out = self.model(latent_variables)
        if self.stochastic:
            self.latent_space = self.gaussian_module(out)
        return out


# ==================== Modules for Covariant shift =================================
class ProbabilisticSingleLayerLSTM(ProbabilisticModel):
    def __init__(self, input_dim, hidden_dim, stochastic=True, init_type='xavier'):
        super(ProbabilisticSingleLayerLSTM, self).__init__(input_dim, stochastic)
        self.cur_input_dim = 2 * input_dim if stochastic else input_dim
        self.lstm = nn.LSTMCell(self.cur_input_dim, hidden_dim)
        self.fc_affine_layer = nn.Linear(hidden_dim, self.cur_input_dim)
        self.register_buffer('h0', torch.zeros([1, self.cur_input_dim]))
        self.register_buffer('c0', torch.zeros([1, self.cur_input_dim]))
        init.init_weights(self.fc_affine_layer, init_type=init_type)

    def forward(self, z_t, hidden_state, cell_state):
        hidden_state, cell_state = self.lstm(z_t, (hidden_state, cell_state))
        latent_variables = self.fc_affine_layer(hidden_state)
        if self.stochastic:
            self.latent_space = self.gaussian_module(latent_variables)
        return latent_variables, hidden_state, cell_state


class StaticProbabilisticEncoder(ProbabilisticModel):
    def __init__(self, model_func, latent_dim, factorised=True, stochastic=True, init_type='xavier'):
        super(StaticProbabilisticEncoder, self).__init__(latent_dim, stochastic)
        self.model_func = model_func
        self.factorised = factorised
        self.zx_dim = 2 * latent_dim if stochastic else latent_dim
        self.conv_fc = nn.Sequential(LinearAffineModule(model_func.n_outputs, model_func.n_outputs))
        if self.factorised:
            self.z_linear = LinearAffineModule(model_func.n_outputs, 2 * latent_dim, batchnorm=False)
        else:
            self.bi_lstm_layer = nn.LSTM(model_func.n_outputs, self.latent_dim, 1, bidirectional=True, batch_first=True)
        self.fc_affine_layer = LinearAffineModule(2 * latent_dim, self.zx_dim, batchnorm=False)
        init.init_weights(self.fc_affine_layer, init_type=init_type)

    def forward(self, x, *args, **kwargs):
        """
        :param x: [bz, d, c, h, w]
        :return:
        """
        batch_size, domains = x.shape[:2]
        x = x.contiguous().view(batch_size * domains, *x.shape[2:])
        encoding = self.conv_fc(self.model_func(x))
        if self.factorised:
            latent_variables = self.z_linear(encoding)
        else:
            encoding = encoding.view(batch_size, domains, -1)
            lstm_out, _ = self.bi_lstm_layer(encoding)  # [bz, d, 2*latent_dim]
            frontal = lstm_out[:, :, 0:self.latent_dim]
            backward = lstm_out[:, :, self.latent_dim:]
            backward = torch.flip(backward, dims=[1])
            lstm_out = torch.cat((frontal, backward), dim=-1)
            latent_variables = lstm_out.view(batch_size * domains, -1)

        latent_variables = self.fc_affine_layer(latent_variables)
        if self.stochastic:
            self.latent_space = self.gaussian_module(latent_variables)

        return latent_variables


class DynamicProbabilisticEncoder(ProbabilisticModel):
    def __init__(self, model_func, latent_dim, env_latent_dim, factorised=True, stochastic=True, init_type='xavier'):
        super(DynamicProbabilisticEncoder, self).__init__(latent_dim, stochastic)
        self.model_func = model_func
        self.factorised = factorised
        self.env_latent_dim = env_latent_dim
        self.zw_dim = 2 * latent_dim if stochastic else latent_dim
        self.conv_fc = nn.Sequential(LinearAffineModule(model_func.n_outputs, model_func.n_outputs))

        if factorised:
            self.z_linear = LinearAffineModule(model_func.n_outputs, self.zw_dim, batchnorm=False)
        else:
            self.z_lstm = nn.LSTM(model_func.n_outputs, self.zw_dim, num_layers=1, bidirectional=False, batch_first=True)

        self.fc_affine_layer = nn.Linear(self.zw_dim, self.zw_dim)
        init.init_weights(self.fc_affine_layer, init_type=init_type)

    def forward(self, x, static_z=None):
        """
        :param x: [bz, d, c, h, w]
        :return:
        """
        batch_size, domains = x.shape[:2]
        x = x.contiguous().view(batch_size * domains, *x.shape[2:])
        encoding = self.conv_fc(self.model_func(x))

        if self.factorised:
            latent_variables = self.z_linear(encoding)  # [batch_size*domains, latent_dim]
        else:
            encoding = encoding.view(batch_size, domains, -1)  # [batch_size, domains, feature_dim]
            latent_variables, _ = self.z_lstm(encoding)
            latent_variables = latent_variables.contiguous().view(batch_size * domains, -1)

        latent_variables = self.fc_affine_layer(latent_variables)

        if self.stochastic:
            self.latent_space = self.gaussian_module(latent_variables)

        latent_variables = latent_variables.view(batch_size, domains, -1)
        # None that batch_size first
        return latent_variables


# ==================== Modules for Concept shift =================================
class ProbabilisticCategoryModel(nn.Module):
    """
    Category distribution
    """

    def __init__(self, latent_dim, stochastic):
        super(ProbabilisticCategoryModel, self).__init__()
        self.latent_dim = latent_dim
        self.stochastic = stochastic
        self.gumbel_prior = None
        self.latent_space = None

    @abstractmethod
    def forward(self, x):
        pass

    def sampling(self, batch_size=1):
        # Gumbel-Softmax Trick, please refer to
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax
        # https://github.com/shaabhishek/gumbel-softmax-pytorch/blob/master/Categorical%20VAE.ipynb
        # We want to generate the percentage of each category here, thus not using hard mode
        if self.training:
            if batch_size == 1:
                latent_value = F.gumbel_softmax(self.gumbel_prior, tau=1., hard=False).unsqueeze(0)
                latent_value = latent_value.squeeze(0)
            else:
                repeated_gumbel_prior = self.gumbel_prior.repeat([batch_size, 1])
                latent_value = F.gumbel_softmax(repeated_gumbel_prior, tau=1., hard=False).unsqueeze(1)
        else:
            latent_value = self.latent_space.probs
            latent_value = latent_value.expand(batch_size, -1).unsqueeze(1)
        return latent_value


class ProbabilisticCatSingleLayer(ProbabilisticCategoryModel):
    """
    The module for  pv
    """

    def __init__(self, input_dim, hidden_dim=64, stochastic=True, init_type='xavier'):
        super(ProbabilisticCatSingleLayer, self).__init__(hidden_dim, stochastic)
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.fc_affine_layer = nn.Linear(hidden_dim, input_dim)
        self.register_buffer('h0', torch.zeros([1, hidden_dim]))
        self.register_buffer('c0', torch.zeros([1, hidden_dim]))
        init.init_weights(self.fc_affine_layer, init_type=init_type)

    def forward(self, z_t, hidden_state, cell_state):
        hidden_state, cell_state = self.lstm(z_t, (hidden_state, cell_state))
        latent_variables = self.fc_affine_layer(hidden_state)

        if self.stochastic:
            self.gumbel_prior = latent_variables
            logits_z = F.log_softmax(latent_variables, dim=-1)
            self.latent_space = torch.distributions.Categorical(logits=logits_z, validate_args=False)

        return latent_variables, hidden_state, cell_state


class DynamicCatEncoder(ProbabilisticCategoryModel):
    """
    The module for qzv
    """
    def __init__(self, input_dim, env_latent_dim, hidden_dim=64, factorised=True, stochastic=True, init_type='xavier'):
        super(DynamicCatEncoder, self).__init__(input_dim, stochastic)
        self.factorised = factorised
        self.env_latent_dim = env_latent_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.conv_fc = nn.Sequential(LinearAffineModule(hidden_dim, hidden_dim))

        if self.factorised:
            self.z_linear = LinearAffineModule(hidden_dim, input_dim, batchnorm=False)
        else:
            self.z_lstm = nn.LSTM(hidden_dim, input_dim, num_layers=1, bidirectional=False, batch_first=True)

        self.fc_affine_layer = nn.Linear(input_dim, input_dim)
        init.init_weights(self.fc_affine_layer, init_type=init_type)

    def forward(self, y, static_z=None):
        """
        :param y: [bz, d, class_num]
        :param static_z:
        :return:
        """
        batch_size, domains = y.shape[:2]
        y = y.contiguous().view(batch_size * domains, *y.shape[2:])
        encoding = self.conv_fc(self.proj(y))
        if self.factorised:
            latent_variables = self.z_linear(encoding)
        else:
            encoding = encoding.view(batch_size, domains, -1)
            latent_variables, _ = self.z_lstm(encoding)
            latent_variables = latent_variables.contiguous().view(batch_size * domains, -1)

        latent_variables = self.fc_affine_layer(latent_variables)

        if self.stochastic:
            self.gumbel_prior = latent_variables
            logits_z = F.log_softmax(latent_variables, dim=-1)
            self.latent_space = torch.distributions.Categorical(logits=logits_z, validate_args=False)

        latent_variables = latent_variables.view(batch_size, domains, -1)

        return latent_variables

