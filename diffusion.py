import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    """
    Extracts the tensor at the given time step.
    Args:
        a: The total time steps.
        t: The time step to extract.
        x_shape: The reference shape.
    Returns:
        The extracted tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_schedule(timesteps, s=0.008):
    """
    Defines the cosine schedule for the diffusion process,
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Args:
        timesteps: The number of timesteps.
        s: The strength of the schedule.
    Returns:
        The computed alpha.
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clip(alphas, 0.001, 1)

# normalization functions
def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# DDPM implementation
class Diffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        channels=3,
        timesteps=1000,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.model = model
        self.num_timesteps = int(timesteps)

        """
        Initializes the diffusion process.
            1. Setup the schedule for the diffusion process.
            2. Define the coefficients for the diffusion process.
        Args:
            model: The model to use for the diffusion process.
            image_size: The size of the images.
            channels: The number of channels in the images.
            timesteps: The number of timesteps for the diffusion process.
        """
        self.alpha=cosine_schedule(self.num_timesteps)
        self.alpha_cumprod= torch.cumprod(self.alpha, axis=0)
        
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod_prev= torch.sqrt(self.alphas_cumprod_prev)
        
        self.sqrt_alphas_cumprod= torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alphas_cumprod= torch.sqrt(1-self.alpha_cumprod)
        
    # backward diffusion
    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        """
        Samples from the reverse diffusion process at time t_index.
        Args:
            x: The initial image.
            t: a tensor of the time index to sample at.
            t_index: a scalar of the index of the time step.
        Returns:
            The sampled image.
        """
        alphas_cumprod_t= extract(self.alpha_cumprod,t,x.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t=extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        epsilon_t= self.model(x,t)
        x_0= (x-sqrt_one_minus_alphas_cumprod_t*epsilon_t)/sqrt_alphas_cumprod_t
        
        sqrt_alphas_cumprod_prev= extract(self.sqrt_alphas_cumprod_prev,t,x.shape) 
        alphas_cumprod_prev=extract(self.alphas_cumprod_prev,t,x.shape)
        
        alphas_t= extract(self.alpha,t,x.shape)
        sqrt_alphas_t= torch.sqrt(alphas_t)
        
        mean= (sqrt_alphas_t*(1-alphas_cumprod_prev)*x)/(1-alphas_cumprod_t) + (sqrt_alphas_cumprod_prev*(1-alphas_t)*x_0)/(1-alphas_cumprod_t)
        
        std= (1-alphas_cumprod_prev)*(1-alphas_t)/(1-alphas_cumprod_t)        
        
        if t_index == 0:
            x_0=torch.clamp(x_0,-1,1)
            return x_0
        
        else:
            z=torch.randn_like(x)
            x_t= mean+ torch.sqrt(std)*z
            x_t= torch.clamp(x_t,-1,1)
            return x_t
        # ####################################################

    @torch.no_grad()
    def p_sample_loop(self, img):
        """
        Samples from the noise distribution at each time step.
        Args:
            img: The initial image that randomly sampled from the noise distribution.
        Returns:
            The sampled image.
        """
        b = img.shape[0]
        
        for t_index in (reversed(range(0,self.num_timesteps))):
            img= self.p_sample(img, torch.full((b,),t_index), t_index)
            
        img= torch.clamp(img,0,1)
        img= unnormalize_to_zero_to_one(img)
        return img
        # ####################################################

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Samples from the noise distribution at each time step.
        Args:
            batch_size: The number of images to sample.
        Returns:
            The sampled images.
        """
        self.model.eval()
        sampled_noise = torch.randn(batch_size,self.channels,self.image_size,self.image_size)
        
        img = self.p_sample_loop(sampled_noise)
        return img

    # forward diffusion
    def q_sample(self, x_0, t, noise=None):
        """
        Samples from the noise distribution at time t. Apply alpha interpolation between x_start and noise.
        Args:
            x_0: The initial image.
            t: The time index to sample at.
            noise: The noise tensor to sample from. If None, noise will be sampled.
        Returns:
            The sampled image.
        """
        if noise==None:
            noise= torch.randn.like(x_0)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod=extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        x_t = sqrt_alphas_cumprod_t*x_0 + sqrt_one_minus_alphas_cumprod*noise
        return x_t

    def p_losses(self, x_0, t, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            t: The time index to compute the loss at.
            noise: The noise tensor to use. If None, noise will be sampled.
        Returns:
            The computed loss.
        """
        if noise==None:
            noise = torch.randn_like(x_0)
        x_noisy= self.q_sample(x_0,t,noise)
        pred_noise=self.model(x_noisy,t)
        loss = F.l1_loss(noise, pred_noise)

        return loss
        # ####################################################

    def forward(self, x_0, noise):
        """
        Computes the loss for the forward diffusion.
        Args:
            x_0: The initial image.
            noise: The noise tensor to use.
        Returns:
            The computed loss.
        """
        b, c, h, w, device, img_size, = *x_0.shape, x_0.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        t=torch.randint(0,self.num_timesteps,(1,))
        loss+=self.p_losses(x_0,t,noise[t])        
        return loss
