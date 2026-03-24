import random

import numpy as np
import torch
from PIL import Image


#---------------------------------------------------------#
#   Convert the image to an RGB image to prevent errors when predicting with grayscale images.
#   The code only supports the prediction of RGB images. All other types of images will be converted to RGB.
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   Resize the input image
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   Obtain the learning rate
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   Set the seeds
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   Set the seed for the DataLoader
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
#normalization
def preprocess_input(image):
    image /= 255.0
    return image

#Display configuration information
def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

#
def download_weights(backbone, model_dir="./model_data"):
    import os

    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'hrnetv2_w18' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w18_imagenet_pretrained.pth",
        'hrnetv2_w32' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w32_imagenet_pretrained.pth",
        'hrnetv2_w48' : "https://github.com/bubbliiiing/hrnet-pytorch/releases/download/v1.0/hrnetv2_w48_imagenet_pretrained.pth",
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)