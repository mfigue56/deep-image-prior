import os
from .common_utils import *
import numpy as np
import scipy.io
import matplotlib.image as mpimg 
        
def get_noisy_image(path_to_image, img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    #Marcos Additions
    ##transform image to k-space (Fourier Transfrom)
    im = mpimg.imread(path_to_image)
    x = im[::2,::2,0]
    y = np.fft.fft2(x)
    clim = np.quantile(np.abs(y.reshape(-1)), [0.01,0.99])
    #zero padding
    y = np.fft.fftshift(y)
    z = np.fft.ifft2(y, s=im.shape[:2]) #zero padded ifft
    clim = np.quantile(np.abs(z.reshape(-1)),[0.01,0.99])
    
    #changed img_np to clim
    img_noisy_np = np.clip(clim + np.random.normal(scale=sigma, size=clim.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np
