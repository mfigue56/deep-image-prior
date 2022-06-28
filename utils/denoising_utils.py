import os
from .common_utils import *
import numpy as np
import scipy.io
import matplotlib.image as mpimg 
from PIL import Image, ImageSequence
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.sparse as sp
from scipy.linalg import dft
from scipy.sparse.linalg import spsolve
        
def get_noisy_image(path_to_image, img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """

    #Marcos Code
    im = Image.open(path_to_image)
    img = im.convert('L')
    
    ft = np.fft.fftshift(np.fft.fft2(img_np))
    noise = np.random.normal(0, .1, ft.shape)
    noisy_signal = ft + noise
    ift = np.abs(np.fft.ifft2(noisy_signal)) #inverse transfrom
    #ift = np.insert(ift,1,0)
    
    img_noisy_np = ift

    #img_noisy = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    #img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_np

def radial_alias(phase, Nx, Ny):
  t = np.linspace(-1,1,500)

  xs = t*np.cos(phase)

  ys = t*np.sin(phase)

  #pc = Nc//2 + (cs*Nc//2 - 1).astype(int)
  px = Nx//2 + (xs*Nx//2 - 1).astype(int)
  py = Ny//2 + (ys*Ny//2 - 1).astype(int)

  trajectory = np.vstack((px, py)).T
  trajectory = np.unique(trajectory,axis = 0)

  A = np.zeros((Nx, Ny)).astype(bool)
  A[trajectory[:,0], trajectory[:,1]] = True

  return A
  
def get_noisy_image_radial(path_to_image,img_np, sigma):
  imgs_arr = img_np
  im0 = imgs_arr[0]
  Nx, Ny = im0.shape
  golden_angle = 2.39996322972865332 #radians
  golden_angle = 111.24*np.pi/180
  phases = np.arange(10)*golden_angle
  A = np.zeros((Nx,Ny))
  for phase in phases:
     A += radial_alias(phase, Nx, Ny)
  fim0 = fftshift(fft2(im0))
  fim0[A == 0] = 0.0
  img0 = np.log(np.abs(fim0))
  img1 = ifft2(fim0)
  noisy_image_np = img1[np.newaxis,:,:]      
  return img0, img1, noisy_image_np
