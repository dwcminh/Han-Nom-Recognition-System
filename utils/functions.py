import csv
import os

import tensorflow as tf
import numpy as np
from skimage.transform import resize

def saveResultToCsv(result, save_path):
    with open(save_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, result[0].keys())
        writer.writeheader()
        writer.writerows(result)
        
def distortion_free_resize(image, size = (432, 48), align_top=True):
    image = np.array(image)

    # Calculate the aspect ratio to preserve
    aspect_ratio = image.shape[1] / image.shape[0]
    new_height, new_width = size
    target_aspect = new_width / new_height
    
    # Calculate new size to preserve aspect ratio
    if aspect_ratio > target_aspect:
        # Width is the limiting factor
        resize_width = new_width
        resize_height = np.round(new_width / aspect_ratio).astype(int)
    else:
        # Height is the limiting factor
        resize_height = new_height
        resize_width = np.round(new_height * aspect_ratio).astype(int)
        
    # Resize the image
    resized_image = resize(image, (resize_height, resize_width), preserve_range=True, anti_aliasing=True).astype(image.dtype)

    # Calculate padding
    pad_height = new_height - resized_image.shape[0]
    pad_width = new_width - resized_image.shape[1]
    
    # Apply padding if needed
    if pad_height == 0 and pad_width == 0:
        return resized_image
    
    pad_height_top = pad_height_bottom = pad_height // 2
    if pad_height % 2 != 0 and not align_top:
        pad_height_top += 1
    elif pad_height % 2 != 0 and align_top:
        pad_height_bottom += 1
    
    pad_width_left = pad_width_right = pad_width // 2
    if pad_width % 2 != 0:
        pad_width_left += 1
        
    # Pad the image
    padded_image = np.pad(resized_image, 
                          ((pad_height_top, pad_height_bottom), 
                           (pad_width_left, pad_width_right), 
                           (0, 0)), 
                          'constant', constant_values=255)
    
    return padded_image    