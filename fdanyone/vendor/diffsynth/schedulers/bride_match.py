import torch, math


class BridgeMatchScheduler:

    def __init__(
        self,
        num_train_timesteps=1000,
        num_inference_steps=8,
        sigma_max=1.0,
        bridge_noise_sigma=0.005,
    ):
        self.sigma_max = sigma_max
        self.sigma_min = sigma_max / num_train_timesteps
        self.num_train_timesteps = num_train_timesteps  # train timesteps for base model
        self.num_inference_steps = num_inference_steps  # inference steps for bridge matching
        self.bridge_noise_sigma = bridge_noise_sigma

        self.set_timesteps(self.num_inference_steps)

    def set_timesteps(self, num_inference_steps=8, training=False):
        sigma_start = self.sigma_max
        sigma_end = self.sigma_max / num_inference_steps
        self.sigmas = torch.linspace(sigma_start, sigma_end, num_inference_steps)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self.training = training

    def retrieve_sigma(self, timestep):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        return sigma

    def get_noise_term(self, sigma, sample):
        # bridge noise term == 0 when sigma == 1 or 0
        return self.bridge_noise_sigma * (sigma * (1.0 - sigma)) ** 0.5 * torch.randn_like(sample)

    def step(self, model_output, timestep, sample, to_final=False, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]

        prev_sample = sample + model_output * (sigma_ - sigma) + self.get_noise_term(sigma_, sample)
        return prev_sample

    def add_noise(self, tgt_sample, src_sample, timestep):
        sigma = self.retrieve_sigma(timestep)
        noisy_sample = sigma * src_sample + (1 - sigma) * tgt_sample + self.get_noise_term(sigma, tgt_sample)
        return noisy_sample

    def training_target(self, tgt_sample, noisy_sample, timestep):
        sigma = self.retrieve_sigma(timestep)

        target = (noisy_sample - tgt_sample) / sigma
        return target

    def denoised_sample(self, prediction, noisy_sample, timestep):
        sigma = self.retrieve_sigma(timestep)

        sample = noisy_sample - prediction * sigma
        return sample

    def training_weight(self, timestep):
        return 1.0
