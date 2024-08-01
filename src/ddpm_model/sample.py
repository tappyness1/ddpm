import torch
import numpy as np

from src.ddpm_model.diffusion import TimeEmbedding, ResidualBlock, AttentionBlock

# the thing to remove the noise
class DDPMSampler:

    def __init__(self, generator: torch.Generator, num_training_steps=1000, beta_start:float  = 0.00085, beta_end: float = 0.0120):
        # will use a linear scheduler
        
        # define beta scheduler which is 1000 numbers between beta_start and beta_end
        # this is called the scaled linear scheduler which can be found on HF
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype = torch.float32).pow(2)

        # closed form solution which allows you to calculate the noise level at any given timestep. This is the alpha. We comput alpha_t as 1 - beta_t
        # alpha_bar is the multiplication between alpha_1 and alpha_2 and so on till alpha_t
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, 0) # [alpha_0, alpha_0 * alpha_1, alpha_0 * alpha_1 * alpha_2, ..., alpha_0 * alpha_1 * ... * alpha_t-1]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps

        self.timesteps = torch.from_numpy(np.arange(0, num_training_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps):
        # re-define the timesteps based on the number of inference steps

        self.num_inference_steps = num_inference_steps
        # 999, 998 ... 0 to be spaced out based on the number of inference steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).copy()

    def add_noise(self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_cumprod = self.alpha_cumprod.to(device = original_samples.device, dtype = original_samples.dtype)
        timesteps = timesteps.to(device = original_samples.device)

        sqrt_alpha_prod = alpha_cumprod[timesteps]
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (self.one - alpha_cumprod[timesteps]) ** 0.5 # this is the standard deviation
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(original_samples.shape, generator = self.generator, device = original_samples.device, dtype = original_samples.dtype)
        noisy_samples = (sqrt_alpha_prod * original_samples) + sqrt_one_minus_alpha_prod*noise
        return noisy_samples
    
    def _get_variance(self, timestep: int) -> torch.FloatTensor:
        prev_timestep = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        prev_alpha_prod_t = self.alpha_cumprod[prev_timestep] if prev_timestep >= 0 else self.one
        curr_alpha_t = alpha_prod_t / prev_alpha_prod_t
        var = (1 - prev_alpha_prod_t) / (1 - alpha_prod_t) * (1 - curr_alpha_t)
        return var

    def step(self, timestep: int, latents: torch.FloatTensor, model_output: torch.FloatTensor) -> torch.FloatTensor:
        t = timestep
        prev_t = self._get_previous_timestep(t)
        alpha_prod_t = self.alpha_cumprod[t]
        alpha_prod_prev_t = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_prev_t
        current_alpha_t = alpha_prod_t / alpha_prod_prev_t
        current_beta_t = 1 - current_alpha_t

        # compute the predicted original sample using the formula (15) in the DDPM paper
        pred_original_sample = (latents - (beta_prod_t ** 0.5 * model_output)) / (alpha_prod_t ** 0.5)

        # compute the coefficient for pred_original_sample and the original sample x_t
        # computing (6) and (7) in the DDPM paper
        pred_original_sample_coeff = (alpha_prod_prev_t ** 0.5 * current_beta_t ) / (beta_prod_t) 
        current_sample_coeff = current_alpha_t ** 0.5 * beta_prod_t_prev / beta_prod_t
        
        # compute the predicted previous sample mean
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * latents

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(model_output.shape, generator = self.generator, device = device, dtype = model_output.dtype)
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0,1) -> N(mu, sigma ^ Z)
        # X = mu + sigma * Z where Z ~ N(0,1)

        pred_prev_sample = pred_prev_sample + variance

        return variance
    
    def set_strength(self, strength: float = 1):
        # strength - the higher it is, the higher the amount of noise to denoise
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        # start from a percentage of the image based on the strength.
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step 

    def _get_previous_timestep(self, timestep: int) -> int:

        # eg if you have step 999 (1000), then the prev time step is 999 - 20 (1000 - 20)
        # assuming you have 50 inference steps and 1000 num_training_steps 
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t