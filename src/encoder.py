import torch 
import torch.nn as nn
from torch.nn import functional as F
from decoder import VAE_attentionBlock , VAE_residualBlock

class VAE(nn.Sequential):
    super().__init__(
        #same size (batch , channel , h, w)
        nn.Conv2d(3,128, kernel_size=3,padding=1),
        #(batch_size, 128, h, w)

        VAE_residualBlock(128, 128),
        
        #(batch_size, 128, h, w)
        VAE_residualBlock(128,128),
        #(batch, 128, h/2, w/2)
        nn.Conv2d(128,128,3, stride=2, padding=0),


        VAE_residualBlock(128, 256),

        #(batch, 256, h/2, w/2)
        VAE_residualBlock(256,256),

        #(bacth, 256 , h/4, w/4)
        nn.Conv2d(256,256 , 3, stride=2, padding=0),

        VAE_residualBlock(256, 512),
        VAE_residualBlock(512,512),

        # (batch ,512, h/8, w/8)
        nn.Conv2d(512, 512 , 3, stride=2, padding=0),

        VAE_residualBlock(512,512),
        VAE_residualBlock(512,512),
        VAE_residualBlock(512,512),

        VAE_attentionBlock(512),

        
        VAE_residualBlock(512,512)   , 
        nn.GroupNorm(32,512),

        nn.SiLU(),

        nn.Conv2d(512, 8, kernel_size=3 , padding=1),

        nn.Conv2d(8, 8, kernel_size=1 , padding=0),
    )
    def forward(self, x:torch.Tensor, noise:torch.Tensor) -> torch.Tensor:
        

        for module in self:
            if getattr(module , "stride", None) == (2,2):
                x = F.pad(x, (0,1,0,1))
            x= module(x)

        mean , log_variance = torch.chunk(x, 2, dim= 1)

        log_variance = torch.clamp(log_variance, -30, 20)

        variance = log_variance.exp()

        stdev = variance.sqrt()

        x = mean +stdev * noise
        x*=0.18215
        return x



