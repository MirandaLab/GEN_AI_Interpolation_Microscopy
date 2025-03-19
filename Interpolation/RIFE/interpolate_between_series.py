"""
Author: Swaraj Kaondal, Miranda lab

Description:
This script uses the RIFE model to create interpolated images between the images in a time series.
"""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import shutil
import torch
import argparse
from torch.nn import functional as F
import warnings
import sys
import cv2
from utils import load_pretrained_model
from common_utils import move_images

def interpolate_imgs(input_dir, output_dir, model):
    """
    Interpolates images between the images in the series.
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
    print(f"Found {len(image_names)} images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_counter = 0
    img_index = 0
    img1 = None
    while img_index < len(image_names)-1:
        print(f"Interpolating an image between {image_names[img_index]} and {image_names[img_index+1]}", end='\r', flush=True)
        
        img0 = cv2.imread(input_dir +"/"+ image_names[img_index])
        img1 = cv2.imread(input_dir +"/"+ image_names[img_index+1])

        img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

        _, _, h, w = img0.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        interpolated_img = model.inference(img0, img1)

        cv2.imwrite(output_dir+f'/{img_counter:05}.png', (img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        cv2.imwrite(output_dir+f'/{(img_counter+1):05}.png', (interpolated_img[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
        img_index += 1
        img_counter += 2
    cv2.imwrite(output_dir+f'/{(img_counter):05}.png', (img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Interpolation for all the images')
    parser.add_argument('--input_dir', required=True, help="input directory for images to be interpolated")
    parser.add_argument('--output_dir', default='./interpolated_images', help="output directory for images to be interpolated")
    parser.add_argument('--model_dir', type=str, default='./train_log', help='directory with trained model files')
    parser.add_argument('--cycles', type=int, default=2, help='Number of interpolation cycles. The')

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
    
    temp_dir = "./temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    move_images(args.input_dir, temp_dir)

    cycle_count = 0
    while cycle_count < args.cycles:
        print(f"\n\nCycle number {cycle_count+1}\n")
        interpolate_imgs(temp_dir, args.output_dir, model)

        shutil.rmtree(temp_dir)
        if cycle_count < args.cycles - 1:
            os.makedirs(temp_dir)
            move_images(args.output_dir, temp_dir)
            shutil.rmtree(args.output_dir)

        cycle_count += 1





