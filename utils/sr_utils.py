from .common_utils import *
from PIL import Image, ImageSequence
import operator


def put_in_center(img_np, target_size):
    img_out = np.zeros([3, target_size[0], target_size[1]])
    
    bbox = [
            int((target_size[0] - img_np.shape[1]) / 2),
            int((target_size[1] - img_np.shape[2]) / 2),
            int((target_size[0] + img_np.shape[1]) / 2),
            int((target_size[1] + img_np.shape[2]) / 2),
    ]
    
    img_out[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = img_np
    
    return img_out

def cropND(img, bounding):
    start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]

def load_LR_HR_imgs_sr(fname, imsize, factor, enforse_div32=None):
    '''Loads an image, resizes it, center crops and downscales.

    Args: 
        fname: path to the image
        imsize: new size for the image, -1 for no resizing
        factor: downscaling factor
        enforse_div32: if 'CROP' center crops an image, so that its dimensions are divisible by 32.
    '''
    img_orig_pil, img_orig_np = get_image(fname, -1)
    ######Marcos Changes#######
    img = Image.open(fname)
    img = img.convert('P')
    imgs_arr = []

    for frame in ImageSequence.Iterator(img):
        imgs_arr.append(np.array(frame.convert('L'))[80:200, 30:150])

    imgs_arr = np.array(imgs_arr)/256
    im1 = imgs_arr[0]
    #######Marcos Changed^##########
    
    if imsize != -1:
        img_orig_pil, img_orig_np = get_image(fname, imsize)
        
    # For comparison with GT
    if enforse_div32 == 'CROP':
        new_size = (img_orig_pil.size[0] - img_orig_pil.size[0] % 32, 
                    img_orig_pil.size[1] - img_orig_pil.size[1] % 32)

        bbox = [
                (img_orig_pil.size[0] - new_size[0])/2, 
                (img_orig_pil.size[1] - new_size[1])/2,
                (img_orig_pil.size[0] + new_size[0])/2,
                (img_orig_pil.size[1] + new_size[1])/2,
        ]

        img_HR_pil = img_orig_pil.crop(bbox)
        img_HR_np = pil_to_np(img_HR_pil)
    else:
        img_HR_pil, img_HR_np = img_orig_pil, img_orig_np
        
    LR_size = [
               img_HR_pil.size[0] // factor, 
               img_HR_pil.size[1] // factor
    ]

    #img_LR_pil = img_HR_pil.resize(LR_size, Image.ANTIALIAS)
    #img_LR_np = pil_to_np(img_LR_pil)
    
    
    ######Changes by Marcos#####
    ft = np.fft.fftshift(np.fft.fft2(img))
    H,W = (ft.shape)
    F = factor  #factor
    h = int(H/F)
    w = int(W/F)
    a =int((H-h)/F)
    b =int((W-w)/F)
    
    #cropping, padding, and inversing
    ftcrop = cropND(ft, (h,w)) #crops the array
    ftpad = np.pad(ftcrop, ([a,a],[b,b]), mode='constant', constant_values=0) #fills the rest of the array with zeroes 
    ift = np.abs(np.fft.ifft2(ftpad)) #inverse transfrom
    img_LR_pil = ift
    #img_LR_pil = np_to_pil(img_LR_np)
    img_LR_np = pil_to_np(img_LR_pil)
    ######### Changed by Marcos ^#######

    print('HR and LR resolutions: %s, %s' % (str(img_HR_pil.size), str (img_LR_pil.size)))

    return {
                'orig_pil': img_orig_pil,
                'orig_np':  img_orig_np,
                'LR_pil':  img_LR_pil, 
                'LR_np': img_LR_np,
                'HR_pil':  img_HR_pil, 
                'HR_np': img_HR_np
           }


def get_baselines(img_LR_pil, img_HR_pil):
    '''Gets `bicubic`, sharpened bicubic and `nearest` baselines.'''
    img_LR_pil2 = sum(img_LR_pil)  #convert tuple to integer?
    img_bicubic_pil = img_LR_pil2.resize(img_HR_pil.size, Image.BICUBIC)
    img_bicubic_np = pil_to_np(img_bicubic_pil)

    img_nearest_pil = img_LR_pil2.resize(img_HR_pil.size, Image.NEAREST)
    img_nearest_np = pil_to_np(img_nearest_pil)

    #img_bic_sharp_pil = img_bicubic_pil.filter(PIL.ImageFilter.UnsharpMask())
    #img_bic_sharp_np = pil_to_np(img_bic_sharp_pil)

    return img_bicubic_np, img_nearest_np #img_bic_sharp_np,



def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
        
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta` 
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    
    return torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))
