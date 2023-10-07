        self.aux_loss_multiplier_y = 70
        self.lambda_zc_mmd, self.lambda_zw_mmd, self.lambda_zv_mmd = 1.0, 1.0, 10.0
        self.noise_mean, self.noise_var = 0., 0.9

        total_loss = (CE_x / domains + static_zc_mmd + dynamic_zw_mmd + dynamic_zv_mmd) / batch_size + CE_y
