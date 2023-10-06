        lr = 5e-5

        self.aux_loss_multiplier_y = 100  # 100
        self.lambda_zc_mmd, self.lambda_zw_mmd, self.lambda_zv_mmd = 5.0, 1.0, 1.0
        self.noise_mean, self.noise_var = 0., 0.5

