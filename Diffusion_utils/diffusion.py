import numpy as np
import torch
import torch.nn as nn
from .helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract
)

'''
最終目標 : def Reverse Process() : 傳入一個 batch 的 state，根據該 state 生出 action (clampped)。
input : 
    * state : shape (batch_size, state_dim)
output : 
    * action : shape (batch_size, action_dim)
'''

'''
* model : GDM in model.py
* max_action : 動作的範圍
* beta_schedule : beta 隨時間變化的規則 (這邊採 vp_beta_schedule in helpers.py)
* denoise_step : Inference 時 GDM 對 x_T 進行去躁的步數
* clipped_denoise : 是否將 x_0_hat 和 x_0 clipped 於 [-max_action, max_action]
'''
class Diffusion(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim,
        model, 
        device,
        max_action,  # 1
        beta_schedule= 'vp',
        denoise_steps= 5,
        clip_denoised= True,
        DDIM= False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = model
        self.device = device
        self.max_action = max_action
        self.denoise_steps = denoise_steps
        self.clip_denoised = clip_denoised
        self.beta_schedule = beta_schedule
        self.DDIM = DDIM

        '''
        we use register buffer to register the values with the class,
        once the instance of this class is generated, the instance will be packed with these values.
        if the instance is moved to the device, these values will be too.
        we use register buffer here to avoid moving every values one-by-one.
        if we want to call the values in the register buffer, we just type self.'variable_name'.
        we can't use the values in register buffer in the def __init__()
        '''
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # gain betas and alphas_relative parameters of each denoise_steps
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # set beta_schedule & gain betas
        # return 依照 beta_schedule 得出各時間步的 beta_t 值。shape (denoise_steps) tensor
        if self.beta_schedule == 'vp': betas = vp_beta_schedule(timesteps= denoise_steps)
        elif self.beta_schedule == 'cosin': betas = cosine_beta_schedule(timesteps= denoise_steps)
        elif self.beta_schedule == 'linear': betas = linear_beta_schedule(timesteps= denoise_steps)
        # gain alphas (alpha_t = 1 - beta_t), shape (denoise_steps)
        alphas = 1. - betas
        # alpha_bar (alpha_bar_t = alpha_0 * ... * alpha_t), shape (denoise_steps)
        alphas_cumprod = torch.cumprod(alphas, dim= 0)
        # alphas_cumprod_prev[t] = alphas_cumprod[t-1] = alpha_bar_(t-1)
        # alphas_cumprod_prev = [1, alphas_cumprod[0], alphas_cumprod[1], ..., alphas_cumprod[t-2]] (t: denoise_steps)
        alphas_cumprod_prev = torch.cat([torch.tensor([1]), alphas_cumprod[:-1]])  # alphas_cumprod[:-1] = [alphas_cumprod[0], alphas_cumprod[1], ..., alphas_cumprod[t-2]]
        
        self.register_buffer('alphas', alphas)
        self.register_buffer('batas', betas)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # pre-calculate some quantities using in forward and reverse process
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # sqrt_alphas_cumprod[t] = sqrt(alpha_bar_t), shape (denoise_steps)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # sqrt_one_minus_alphas_cumprod[t] = sqrt(1 - alpha_bar_t)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # log_one_minus_alphas_cumprod[t] = log(1 - alpha_bar_t)
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # sqrt_recip_alphas_cumprod[t] = sqrt(1 / alpha_bar_t)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # sqrt_recipm1_alphas_cumprod[t] = sqrt((1 / alpha_bar_t) - 1)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # pre-calculations of the posterior distribution q(x_(t-1)|x_t, x_0) & p(x_(t-1)|x_t, x_0_hat) for each t
        # we use posteroir_log_variance_clipped instead of posterior_variance in the rest of the GDM process
        # why log ? use log to stablize the value
        # why clip ? to avoid log(0)
        #-------------------------------------------------------------------------------------------------------------------------------------------#
        # 1. variance of the posterior distribution, shape (denoise_steps)
        # \beta_t * I in the paper, more accurate here
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        # if posterior_variance = 0, it'll cause log(0) = -inf
        # to avoid posterior_variance = 0 in the very first timeslot (alphas_cumprod_prev[0] = 1), replace 0 with 1e-20
        # posterior_log_variance_clipped[t] = log(variance[t]) = log(\sigma[t]^2)
        self.register_buffer('posterior_log_variance_clipped', torch.log(torch.clamp(posterior_variance, min= 1e-20)))
        # 2. mean of the posterior distritbution q(x_(t-1)|x_t, x_0) & p(x_(t-1)|x_t, x_0_hat) for each t
        # mean(x_t | x_0) = coef1 * x_0 + coef2 * x_t
        self.register_buffer('posterior_mean_coef1', betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 求出一個 batch 各資料的 x_0_hat
    # x_t : shape (batch_size, action_dim)
    # t : shape (batch_size)
    # predicted_noise_GDM : shape (batch_size, action_dim)
    # return x_0_hat of each data in a batch, shape (batch_size, action_dim)
    # extract : extract coef of each t in a batch
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def x_0_hat(self, x_t, t, predicted_noise_GDM):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - 
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * predicted_noise_GDM
        )
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 對一個 batch 中各資料做一次 denoise 後得到各資料的 mean & variance
    # x_t : shape (batch_size, action_dim)
    # t : shape (batch_size)
    # state : shape (batch_size, state_dim)
    # return : parameters of posterior distribution of each data in a batch
    #     1. posterior_mean : shape (batch_size, action_dim)
    #     2. posterior_variance : shape (batch_size, 1)
    #     3. posterior_log_variance_clipped : shape (batch_size, 1)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def posterior_mean_variance(self, x_t, t, state):
        # x_0_hat of each data in a batch, shape (batch_size, action_dim)
        x_0_hat = self.x_0_hat(x_t= x_t, t= t, predicted_noise_GDM= self.model(state= state, x_t= x_t, time= t))
        # clip x_0_hat in [-max_action, max_action]
        # clamp_ -> in-place, clamp -> return new tensor (but preserve gradient), torch.clamp() is recommanded
        if self.clip_denoised: x_0_hat.clamp(-self.max_action, self.max_action)
        # use tanh to provide more smooth gradient
        # if self.clip_denoised: x_0_hat = torch.tanh(x_0_hat)
        # 但 tanh 在接近 1 & -1 這種邊界的時候梯度會是 0，所以改回 torch.clamp，讓他在邊界時維持強度，因為 Critic 一直起不來

        # calculate mean & variance & log of posterior distribution
        # posterior_mean : shape (batch_size, action_dim)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0_hat + 
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # posterior_variance : shape (batch_size, 1)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        # posterior_log_variance_clipped : shape (batch_size, 1)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 對一個 batch 中各資料做一次 denoise 後得到各資料的 x_(t-1) by DDPM
    # x_t : shape (batch_size, action_dim)
    # t : shape (batch_size)
    # state : shape (batch_size, state_dim)
    # return x_(t-1) of each data in a batch by DDPM, shape (batch_size, action_dim)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def DDPM_single_denoise_step(self, x_t, t, state):
        batch_size, _ = x_t.shape

        # do single denoise step of each data and gain their posterior distributions parameters
        model_mean, model_variance, model_log_variance = self.posterior_mean_variance(x_t= x_t, t= t, state= state)
        
        # except final denoise step (t = 0), each x_(t-1) need to add a noise to enhance exploration 
        # (that means only x_0 is the pure mean of the posterior distribution (without randomness), otherwise(x_(t-1) | t>1) are sampled by reparameterization (with randomness))
        # generate independent noises (obey gaussian distribution N(0, I)) with shape (batch_size, action_dim)
        noise = torch.randn_like(x_t)
        # generate a noise mask to avoid adding noise at t=0
        # ex: t = [1, 0, 2] -> (t==0) = [false(0), true(1), false(0)] -> (1 - (t==0)) = [1, 0, 1]
        nonzero_mask = (1 - (t == 0).float()).reshape(batch_size, *((1,) * (len(x_t.shape) - 1)))  # shape (batch_size, 1)
        # generate x_(t-1) by reparameterization (x_(t-1) = mean + sigma * noise) noise ~ gaussian distribution
        # 0.5 * (log(sigma^2)) = log(sigma), log(sigma).exp() = sigma
        # nonzero_mask (batch_size, 1) * noise (batch_size, action_dim) -> shape (batch_size, action_dim)
        return model_mean + (0.5 * model_log_variance).exp() * nonzero_mask * noise
        
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 對一個 batch 中各資料做一次 denoise 後得到各資料的 x_(t-1) by DDIM
    # x_t : shape (batch_size, action_dim)
    # t : shape (batch_size)
    # state : shape (batch_size, state_dim)
    # return x_(t-1) of each data in a batch by DDIM, shape (batch_size, action_dim)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def DDIM_single_denoise_step(self, x_t, t, state):
        batch_size, _ = x_t.shape
        # predicted_epsilon of each data in batch, shape (batch_size, action_dim)
        predicted_noise_GDM= self.model(state= state, x_t= x_t, time= t)
        # x_0_hat of each data in a batch, shape (batch_size, action_dim)
        x_0_hat = self.x_0_hat(x_t= x_t, t= t, predicted_noise_GDM= predicted_noise_GDM)
        return (
            extract(torch.sqrt(self.alphas_cumprod_prev), t, x_t.shape) * x_0_hat + 
            extract(torch.sqrt(1. - self.alphas_cumprod_prev), t, x_t.shape) * predicted_noise_GDM
        )

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 對一個 batch 中各資料做一遍完整的 (self.denoise_steps 次 denoise) reverse process 後得到各資料最終的 action (clampped)。
    # state : shape (batch_size, state_dim)
    # x_t_shape : tuple (batch_size, action_dim)
    # return x_0 (action) of each data in a batch, shape (batch_size, action_dim)
    # 這邊的 x_0 都是有追蹤梯度的
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def reverse_process(self, state):
        batch_size, _ = state.shape
        # x_t : shape (batch_size, action_dim), obey the gaussian distirbution N(0, I)
        x_t = torch.randn((batch_size, self.action_dim), device= self.device)
        
        # do reverse process for each data in a batch
        for i in reversed(range(0, self.denoise_steps)):  # self.denoise_steps-1, self.denoise_steps-2, ..., 1, 0
            # will be used in extract() as indices of gather() so its dtype must be torch.long, shape (batch_size)
            timesteps = torch.full((batch_size,), i, device= self.device, dtype= torch.long)  
            # do the ith denoise step to each data in a batch then return x_(t-1) of each data, shape (batch_size, action_dim)
            # DDPM
            if not self.DDIM: x_next = self.DDPM_single_denoise_step(x_t= x_t, t= timesteps, state= state)
            # DDIM
            else: x_next = self.DDIM_single_denoise_step(x_t= x_t, t= timesteps, state= state)
        
        # return torch.tanh(x_next)
        return torch.clamp(x_next, -self.max_action, self.max_action)
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 可以透過 diffusion(state) 來做 diffusion.reverse_process(state) 
    # state : shape (batch_size, state_dim)
    # return action (clampped), shape (batch_size, action_dim)
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    def forward(self, state):
        return self.reverse_process(state= state)

    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # forward process
    # 傳入一個 batch 的 x_0 (batch_size, action_dim)，加躁後回傳 x_t (batch_size, action_dim)
    # x_0 : shape (batch_size, action_dim)
    # state : shape (batch_size, state_dim)
    # nosie : shape (batch_size, action_dim)
    # t : shape (batch_size)
    def forward_process(self, x_0, noise, t):
        # do forward process
        # x_t shape (batch_size, action_dim)
        x_t = (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + 
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )
        return x_t
    
    #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    # 將一個 batch 的 x_0 做 forward process 後 (得 x_t, \epsilon)，與 model(state, x_t(隨機噪聲), t) 輸出的預測噪聲 (\epsilon_\theta) 進行 MSE 計算後回傳
    # return MSE(\epsilon, \epsilon_\theta)
    # x_0 : shape (batch_size, action_dim)
    # state : shape (batch_size, state_dim)
    # return torch.tensor(loss) ex.torch.tensor(9) (shape ()), not torch.tensor([9]) (shape (1))
    def loss(self, x_0, state):
        # generate noise ~ N(0, I) with shape (batch_size, action_dim)
        # don't need to set device because randn_like will automatically set the same device as the input tensor
        noise = torch.randn_like(x_0)  
        # generate independent times for each data in a batch (t is in the range [0, self.denoise_step])
        # change dtype to long() because t is used to being an index in Gather() which requires dtype to be long()
        t = torch.randint(low= 0, high= self.denoise_steps, size= (x_0.shape[0],), device= x_0.device).long()
        # evalute x_t by doing forward process to x_0, shape (batch_size, action_dim)
        x_t = self.forward_process(x_0= x_0, noise= noise, t= t)
        # get predicted noise, shape (batch_size, action_dim)
        predicted_noise = self.model(state= state, x_t= x_t, time= t)
        return nn.functional.mse_loss(predicted_noise, noise)

