"""
Author: Swaraj Kaondal, Miranda lab

Description:
This script uses the RIFE model to create interpolated images of all the real images in a time series.
It interpolates an image in the seires by taking i and i+2 image and interpolating the i+1 image of that series.
"""

import os
import cv2
import torch
import argparse
from torch.nn import functional as F
import warnings
import sys
from utils import load_pretrained_model

def interpolate_imgs(input_dir, output_dir, model):
    """
    Creates a series of interpolated images by taking i and i+2 image in a seires and interpolating the i+1 image.
    It is expected that the series of the images in the input_dir match the order when the images are arranged in the
    lexicographical order.

    Parameters:
    input_dir(String) - path to the series of original frames.
    output_dir(String) - path where the interpolated images will be stored.
    model - RIFE model for image interpolation.

    Returns:
    None
    """
    image_names = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
    image_names.sort()
    print(f"Found {len(image_names)} images. Creating interpolated images of every image except the first and the last image.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_index = 0
    while img_index < len(image_names)-2:
        print(f"Interpolating {image_names[img_index+1]}", end='\r', flush=True)
        
        img0 = cv2.imread(input_dir +"/"+ image_names[img_index])
        img1 = cv2.imread(input_dir +"/"+ image_names[img_index+2])

        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        interpolated_img = model.inference(img0, img1)

        cv2.imwrite(output_dir+f'/{image_names[img_index+1]}', (interpolated_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        img_index += 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Interpolation for all the images')
    parser.add_argument('--input_dir', required=True, help="input directory for images to be interpolated")
    parser.add_argument('--output_dir', default='./interpolated_images', help="output directory for images to be interpolated")
    parser.add_argument('--model_dir', type=str, default='./train_log', help='directory with trained model files')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    try:
        model = load_pretrained_model(args.model_dir)
    except Exception as e:
        print(f"Failed to load RIFE model, please check if the correct model exists on the model_dir path: {e}")
        sys.exit(1)
    
    interpolate_imgs(args.input_dir, args.output_dir, model)
    


