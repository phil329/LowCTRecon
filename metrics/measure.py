import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from .ssim import ssim

from kornia.filters import get_gaussian_kernel2d, filter2d



def compute_measure(x,y):
    
    result = {}
    
    # transform value_range to (0,1)
    x = x.clamp(-1,1)*0.5+0.5
    y = y.clamp(-1,1)*0.5+0.5
    
    # calculate SSIM
    ssim_ = ssim(x, y, data_range=1, size_average=False).mean()
    
    # transform value_range to (0,255)
    x = x*255
    y = y*255
    
    mse = torch.mean(torch.pow(x - y, 2))
    rmse = torch.sqrt(mse)
    
    psnr = 20 * torch.log10(255./rmse)
    
    result = {'psnr':psnr,'rmse':rmse,'ssim':ssim_}
    return result