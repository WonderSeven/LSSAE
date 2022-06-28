import pdb

import torch
from torch.autograd import Variable

from .submodules import *
from .cla_func import *
from .loss_func import *
from engine.utils import one_hot
from engine.configs import Algorithms


class AbstractAutoencoder(nn.Module):
    def __init__(self, model_func, cla_func, hparams):
        super().__init__()
        self.model_func = model_func
        self.cla_func = cla_func
        self.hparams = hparams
        self.feature_dim = model_func.n_outputs
        self.data_size = hparams['data_size']
        self.num_classes = hparams['num_classes']
        self.seen_domains = hparams['source_domains']

        self.zc_dim = hparams['zc_dim']
        self.zw_dim = hparams['zw_dim']
        self.zv_dim = hparams['zv_dim']

        self.recon_criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _get_decoder_func(self):
        if len(self.data_size) > 2:
            if self.data_size[-1] == 28:
                decoder_class = CovDecoder28x28
            elif self.data_size[-1] == 84:
                decoder_class = CovDecoder84x84
            else:
                raise ValueError('Don\'t support shape:{}'.format(self.hparams['data_size']))
        else:
            decoder_class = LinearDecoder
        return decoder_class

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def update(self, minibatches, unlabeled=False):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        pass

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        pass

    def calc_recon_loss(self, recon_x, x):
        recon_loss = self.recon_criterion(recon_x, x)
        recon_loss = recon_loss.sum()
        return recon_loss

    def update_scheduler(self):
        pass


@Algorithms.register('vae')
class VAE(AbstractAutoencoder):
    """
    Implementation of Vanilla VAE.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(VAE, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.encoder = ProbabilisticEncoder(self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.decoder = self._get_decoder_func()(self.hparams['zc_dim'], self.data_size)

        self.domain_cla_func = SingleLayerClassifier(self.hparams['zc_dim'], self.seen_domains)

        self.opt = torch.optim.Adam([{'params': self.encoder.parameters()},  #
                                     {'params': self.decoder.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

        self.opt_cla = torch.optim.Adam(params=self.cla_func.parameters(), lr=self.hparams["lr"],
                                        weight_decay=self.hparams['weight_decay'])

        self.opt_domain_cla = torch.optim.Adam(params=self.domain_cla_func.parameters(), lr=self.hparams["lr"],
                                               weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            # Step1: Optimize VAE
            _ = self.encoder(x)
            zx_q = self.encoder.sampling()
            recon_x = self.decoder(zx_q)
            recon_loss = self.calc_recon_loss(recon_x, x)
            recon_kl_loss = kl_divergence(self.encoder.latent_space, self.gaussian.latent_space)
            vae_loss = recon_loss + recon_kl_loss
            self.opt.zero_grad()
            vae_loss.backward()
            self.opt.step()

            # Classification
            zx_mu = Variable(self.encoder.latent_space.base_dist.loc, requires_grad=True)
            pred_logit = self.cla_func(zx_mu)
            cla_loss = self.criterion(pred_logit, y)
            self.opt_cla.zero_grad()
            cla_loss.backward()
            self.opt_cla.step()

            domain_y = torch.ones_like(y) * domain_idx
            zx_mu = Variable(self.encoder.latent_space.base_dist.loc, requires_grad=True)
            domain_logit = self.domain_cla_func(zx_mu)
            domain_cla_loss = self.criterion(domain_logit, domain_y)
            self.opt_domain_cla.zero_grad()
            domain_cla_loss.backward()
            self.opt_domain_cla.step()

            # Append status for each domain
            cla_losses.append(cla_loss)
            all_y_pred.append(pred_logit)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        training = False
        _ = self.encoder(x)
        zx_q = self.encoder.latent_space.base_dist.loc
        output = self.cla_func(zx_q)
        return output

    def reconstruct_for_test(self, x, generative=False):
        with torch.no_grad():
            if generative:
                zx = self.gaussian.sampling(x.size(0))
            else:
                _ = self.encoder(x)
                zx = self.encoder.sampling()
            recon_x = self.decoder(zx)
        return recon_x


@Algorithms.register('diva')
class DIVA(AbstractAutoencoder):
    """
    Implementation of DIVA.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(DIVA, self).__init__(model_func, cla_func, hparams)
        self.model_func = model_func
        self.cla_func = cla_func
        self.aux_loss_multiplier_d = 3500
        self.aux_loss_multiplier_y = 2000
        self._build()

    def _build(self):
        # Static env components
        self.gaussian = GaussianModule(self.hparams['zc_dim'])
        self.qzx = ProbabilisticEncoder(self.model_func, self.hparams['zc_dim'], self.hparams['stochastic'])
        self.qzy = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])
        self.qzd = ProbabilisticEncoder(copy.deepcopy(self.model_func), self.hparams['zdy_dim'],
                                        self.hparams['stochastic'])

        self.px = self._get_decoder_func()(self.hparams['zc_dim'] + 2 * self.hparams['zdy_dim'], self.data_size)

        self.pzd = BranchDecoder(self.seen_domains, self.hparams['zdy_dim'], self.hparams['stochastic'])
        self.pzy = BranchDecoder(self.num_classes, self.hparams['zdy_dim'], self.hparams['stochastic'])

        # Auxiliary branch
        self.qd = SingleLayerClassifier(self.hparams['zdy_dim'], self.seen_domains)
        self.qy = self.cla_func

        self.opt = torch.optim.Adam([{'params': self.qzx.parameters()},  #
                                     {'params': self.qzy.parameters()},
                                     {'params': self.qzd.parameters()},
                                     {'params': self.px.parameters()},
                                     {'params': self.pzd.parameters()},
                                     {'params': self.pzy.parameters()},
                                     {'params': self.qd.parameters()},
                                     {'params': self.qy.parameters()}],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        cla_losses, all_y_pred, all_y = [], [], []

        for domain_idx, (x, y) in enumerate(minibatches):
            domain_y = torch.ones_like(y) * domain_idx

            _ = self.qzx(x)
            _ = self.qzy(x)
            _ = self.qzd(x)

            zx_q = self.qzx.sampling()
            zy_q = self.qzy.sampling()
            zd_q = self.qzd.sampling()

            recon_x = self.px(torch.cat([zd_q, zx_q, zy_q], dim=1))

            one_hot_y = one_hot(y, self.num_classes, x.device)
            one_hot_d = one_hot(domain_y, self.seen_domains, x.device)

            _ = self.pzy(one_hot_y)
            _ = self.pzd(one_hot_d)

            d_hat = self.qd(zd_q)
            y_hat = self.qy(zy_q)

            CE_x = self.calc_recon_loss(recon_x, x)
            zd_p_minus_zd_q = torch.sum(self.pzd.latent_space.log_prob(zd_q) - self.qzd.latent_space.log_prob(zd_q))
            KL_zx = torch.sum(self.gaussian.latent_space.log_prob(zx_q) - self.qzx.latent_space.log_prob(zx_q))
            zy_p_minus_zy_q = torch.sum(self.pzy.latent_space.log_prob(zy_q) - self.qzy.latent_space.log_prob(zy_q))

            # Semantic classification
            CE_y = self.criterion(y_hat, y)
            # Domain classification
            CE_d = self.criterion(d_hat, domain_y)

            loss = CE_x - zd_p_minus_zd_q - KL_zx - zy_p_minus_zy_q + self.aux_loss_multiplier_d * CE_d + \
                   self.aux_loss_multiplier_y * CE_y
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Append status for each domain
            cla_losses.append(loss)
            all_y_pred.append(y_hat)
            all_y.append(y)

        # Record training procedure
        cla_losses = torch.mean(torch.stack(cla_losses, dim=0))
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y = torch.cat(all_y, dim=0)

        return cla_losses, all_y_pred, all_y

    def predict(self, x, *args, **kwargs):
        _ = self.qzy(x)
        zx_q = self.qzy.latent_space.base_dist.loc
        output = self.qy(zx_q)
        return output

    def predict_domain(self, x, *args, **kwargs):
        _ = self.qzd(x)
        zd_q = self.qzd.latent_space.base_dist.loc
        output = self.qd(zd_q)
        return output

    def reconstruct_for_test(self, x, generative=False):
        with torch.no_grad():
            _ = self.encoder(x)
            if generative:
                zx_q = self.gaussian.sampling(x.size(0))
            else:
                zx_q = self.qzx.sampling()
            _ = self.qzy(x)
            _ = self.qzd(x)
            zd_q = self.qzd.sampling()
            zy_q = self.qzy.sampling()
            recon_x = self.px(torch.cat([zd_q, zx_q, zy_q], dim=1))
        return recon_x


@Algorithms.register('lssae')
class LSSAE(AbstractAutoencoder):
    """
    Implementation of LSSAE.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(LSSAE, self).__init__(model_func, cla_func, hparams)
        self.factorised = True

        self.aux_loss_multiplier_y = hparams['coeff_y']
        self.ts_multiplier = hparams['coeff_ts']

        self._build()
        self._init()

    def _build(self):
        # Static env components
        self.static_prior = GaussianModule(self.zc_dim)
        self.dynamic_w_prior = ProbabilisticSingleLayerLSTM(input_dim=self.zw_dim,
                                                            hidden_dim=2 * self.zw_dim,
                                                            stochastic=self.hparams['stochastic'])

        self.dynamic_v_prior = ProbabilisticCatSingleLayer(input_dim=self.zv_dim,
                                                           hidden_dim=2 * self.zv_dim,
                                                           stochastic=self.hparams['stochastic'])

        self.static_encoder = StaticProbabilisticEncoder(self.model_func,
                                                         self.zc_dim,
                                                         stochastic=self.hparams['stochastic'])
        self.dynamic_w_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                             self.zw_dim,
                                                             self.zc_dim,
                                                             factorised=self.factorised,
                                                             stochastic=self.hparams['stochastic'])
        self.dynamic_v_encoder = DynamicCatEncoder(self.zv_dim,
                                                   self.zc_dim,
                                                   factorised=self.factorised,
                                                   stochastic=self.hparams['stochastic'])

        self.decoder = self._get_decoder_func()(self.zc_dim + self.zw_dim, self.data_size)

        self.category_cla_func = self.cla_func

        self.opt = torch.optim.Adam([{'params': self.static_encoder.parameters()},
                                     {'params': self.category_cla_func.parameters()},
                                     {'params': self.decoder.parameters()},
                                     {'params': self.dynamic_w_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_encoder.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_w_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_prior.parameters(), 'lr': 0.1 * self.hparams["lr"]}
                                     ],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    @staticmethod
    def gen_dynamic_prior(prior_net, prior_latent_dim, domains, batch_size=1, stochastic=False):
        z_out, z_out_value = None, None
        hx = Variable(prior_net.h0.detach().clone(), requires_grad=True)
        cx = Variable(prior_net.c0.detach().clone(), requires_grad=True)

        init_prior = torch.zeros([2 * prior_latent_dim if stochastic else prior_latent_dim]).cuda()

        z_t = Variable(init_prior.detach().clone(), requires_grad=True).unsqueeze(0)

        for _ in range(domains):
            z_t, hx, cx = prior_net(z_t, hx, cx.detach().clone())

            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_out_value = prior_net.sampling(batch_size)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_out_value = torch.cat((z_out_value, prior_net.sampling(batch_size)), dim=1)
        return z_out, z_out_value

    def update(self, minibatches, unlabeled=None):
        """
        :param minibatches: list
        :param unlabeled:
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, data_size]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]

        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)  # [batch_size, source_domains, data_size]
        all_y = torch.transpose(all_y, 0, 1)  # [batch_size, source_domains]

        # ------------------------------ Covariant shift  -------------------------------
        static_qx_latent_variables = self.static_encoder(all_x)  # [batch_size, zc_dim*2]
        dynamic_qw_latent_variables = self.dynamic_w_encoder(all_x, None)  # [batch_size, source_domains, zw_dim*2]
        dynamic_pw_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_w_prior, self.zw_dim, domains, batch_size,
                                                                self.hparams['stochastic'])  # [1, source_domains, zw_dim*2]

        zc = self.static_encoder.sampling()
        zw = self.dynamic_w_encoder.sampling()
        recon_x = self.decoder(torch.cat([zc, zw], dim=1))
        all_x = all_x.contiguous().view(batch_size * domains, *all_x.shape[2:])
        CE_x = self.calc_recon_loss(recon_x, all_x)

        # Distribution loss
        # kld on zc
        static_kld = -1.0 * torch.sum(1 + static_qx_latent_variables[:, self.zc_dim:] -
                                      torch.pow(static_qx_latent_variables[:, :self.zc_dim], 2) -
                                      torch.exp(static_qx_latent_variables[:, self.zc_dim:]))
        # kld on zw
        dynamic_qw_mu, dynamic_qw_log_sigma = dynamic_qw_latent_variables[:, :, :self.zw_dim], \
                                              dynamic_qw_latent_variables[:, :, self.zw_dim:]
        dynamic_pw_mu, dynamic_pw_log_sigma = dynamic_pw_latent_variables[:, :, :self.zw_dim], \
                                              dynamic_pw_latent_variables[:, :, self.zw_dim:]
        dynamic_qw_sigma = torch.exp(dynamic_qw_log_sigma)
        dynamic_pw_sigma = torch.exp(dynamic_pw_log_sigma)

        dynamic_w_kld = 1.0 * torch.sum(dynamic_pw_log_sigma - dynamic_qw_log_sigma + ((dynamic_qw_sigma + torch.pow(dynamic_qw_mu - dynamic_pw_mu, 2)) / dynamic_pw_sigma) - 1)

        # ------------------------------ Concept shift  -------------------------------
        all_y = all_y.contiguous().view(-1)
        one_hot_y = one_hot(all_y, self.num_classes, all_y.device)
        one_hot_y = one_hot_y.view(batch_size, domains, -1)
        dynamic_qv_latent_variables = self.dynamic_v_encoder(one_hot_y, None)

        dynamic_pv_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_v_prior, self.zv_dim, domains, batch_size,
                                                                False)  # [1, source_domains, zv_dim]

        # recon y
        zv = self.dynamic_v_encoder.sampling()
        zv.view(batch_size, domains, -1)
        recon_y = self.category_cla_func(torch.cat([zv, zc], dim=1))
        CE_y = self.aux_loss_multiplier_y * self.criterion(recon_y, all_y)

        # kld on zv
        dynamic_v_kld = torch.sum(torch.softmax(dynamic_qv_latent_variables, dim=-1) *
                                  (torch.log_softmax(dynamic_qv_latent_variables, dim=-1) -
                                   torch.log_softmax(dynamic_pv_latent_variables, dim=-1)))

        # temporal smooth constrain on prior_dynamic_latent_variables
        ts_w_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qw_latent_variables)
        ts_v_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qv_latent_variables)

        total_loss = (CE_x + static_kld + dynamic_w_kld + dynamic_v_kld) / batch_size + CE_y + ts_w_loss + ts_v_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        recon_x_loss = CE_x / batch_size
        static_loss = static_kld / batch_size
        dynamic_w_kld = dynamic_w_kld / batch_size
        dynamic_v_kld = dynamic_v_kld / batch_size

        print('Total loss:{:.3f}, recon_x_loss:{:.3f}, recon_y_loss:{:.3f}, static_loss:{:.3f}, dynamic_w_loss:{:.3f}, '
              'dynamic_v_loss:{:.3f}, TS_W_loss:{:.3f}, TS_V_loss:{:.3f}'.
              format(total_loss, recon_x_loss, CE_y, static_loss, dynamic_w_kld, dynamic_v_kld, ts_w_loss, ts_v_loss))

        return CE_y, recon_y, all_y

    def predict(self, x, domain_idx, *args, **kwargs):
        _ = self.static_encoder(x.unsqueeze(1))
        zc = self.static_encoder.latent_space.base_dist.loc
        _, zv_prob = self.gen_dynamic_prior(self.dynamic_v_prior, self.zv_dim, domain_idx+1, x.size(0), False)  # [1, source_domains, zv_dim]
        zv = zv_prob[:, -1, :]
        y_logit = self.category_cla_func(torch.cat([zv, zc], dim=1))
        return y_logit

    def reconstruct_for_test(self, minibatches, generative=False):
        """
        :param minibatches:
        :param generative: True or False
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, 2]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]
        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)  # [batch_size, source_domains, 2]
        all_y = torch.transpose(all_y, 0, 1)  # [batch_size, source_domains]

        static_qx_latent_variables = self.static_encoder(all_x)  # [48, 40]
        _ = self.dynamic_w_encoder(all_x, static_qx_latent_variables[:, : self.zc_dim])  # [48, 15, 40]
        _, zw = self.gen_dynamic_prior(self.dynamic_w_prior, self.zw_dim, domains, batch_size)  # [1, 15, 40]

        all_y = all_y.contiguous().view(-1)

        domain_idx = torch.arange(domains).to(all_x.device)
        domain_idx = domain_idx.unsqueeze(0).expand(batch_size, -1)
        domain_idx = domain_idx.contiguous().view(-1)

        if generative:
            # sample from static gaussian
            zc = self.static_encoder.sampling(batch_size)
            zw = zw.contiguous().view(batch_size, domains, -1)
            zw = zw[:, 0, :]
            zw = zw.expand(batch_size * domains, -1)
        else:
            zc = self.static_encoder.sampling()
            zw = self.dynamic_w_encoder.sampling()  # [720, 20]
        recon_x = self.decoder(torch.cat([zc, zw], dim=1))

        return recon_x, all_y, domain_idx
