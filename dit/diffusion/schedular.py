import torch
import torch.nn as nn


class NoiseSchedular:
    def __init__(self, timesteps = 1000, beta_start = 1e-4, beta_end = 0.02, device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.timesteps = timesteps

        #varinace schedular
        betas = torch.linspace(beta_start, beta_end, timesteps, device = device) #shape: (T, )

        #alpha_t = 1 - beta_t
        alphas = 1.0 - betas #shape: (T, )

        # alpha_bar_t = cummulative product of alphas till timestep 't'

        alpha_bar_t = torch.cumprod(alphas, dim = 0) #shape: (T, )

         # alpha_bar_{t-1}, shifted by one, with 1.0 prepended for t=0

        alpha_bar_t_minus_one = alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0], device=device), alpha_bar_t[:-1]
        ])

        self.betas = betas
        self.alphas = alphas
        self.alpha_cum_prod = alpha_bar_t
        self.alpha_cum_prod_prev = alpha_bar_t_minus_one

        #precompute 

        self.sqrt_alphas_cumprod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alpha_cum_prod)

        # Posterior Distribution: # **Posterior Distribution**

        # The posterior distribution q(xt−1∣xt,x0) can be derived from the forward process and is given by a Gaussian distribution with the mean and variance computed:

        #posterior varinace: sigma_i_squared: 
        self.posterior_variance = (
            betas * (1.0 - alpha_bar_t_minus_one)  / (1.0 - alpha_bar_t)
        )

        #posterior mean coefficients: 

        self.posterior_mean_coeff_1 = (
            betas * torch.sqrt(alpha_bar_t_minus_one) / (1.0 - alpha_bar_t)
        )

        self.posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1.0 - alpha_bar_t_minus_one) / (1.0 - alpha_bar_t)
        )


    def _extract(self, arr, t, shape):
        # arr: precomputer lookup tables shape: (T, ) 
        # arr[0] is the value for timestep t = 0
        # t shape: (B, ) 
        out = arr.to(t.device)[t] #outputs arr[t] #output sqrt_alpha_cumprod[t]
        return out.reshape(t.shape[0], *([1] * (len(shape) - 1))) 
    
    def q_sample(self, x0, t, noise):
        # closed-form forward process: jump directly to noise level t
        # x0: (B, C, H, W)   t: (B,)   noise: (B, C, H, W)
        sqrt_ac = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_ac = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_one_minus_ac * noise
    
    def q_posterior_mean(self, x0_hat, xt, t):
        #xt and x0_hat: shape (B, C, H, W)
        coef1 = self._extract(self.posterior_mean_coeff_1, t, xt.shape)
        coef2 = self._extract(self.posterior_mean_coeff_2, t, xt.shape)
        return coef1 * x0_hat + coef2 * xt
    
    def posterior_std(self, t, shape):
        var = self._extract(self.posterior_variance, t, shape)
        return torch.sqrt(var)