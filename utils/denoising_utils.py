import os
from .common_utils import *
import numpy as np
import scipy.io
import matplotlib.image as mpimg 
from PIL import Image, ImageSequence
        
def get_noisy_image(path_to_image, img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    #Marcos Code
    im = Image.open(path_to_image)
    img = im.convert('L')
    
    ft = np.fft.fftshift(np.fft.fft2(img, norm = 'forward'))
    noise = np.random.normal(0, .1, ft.shape)
    noisy_signal = ft + noise
    ift = np.abs(np.fft.ifft2(noisy_signal, norm = 'forward')) #inverse transfrom
    ift - ift.insert(1,0)
    
    img_noisy_np = ift

    img_noisy = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy)

    return img_noisy_pil, img_noisy_np
