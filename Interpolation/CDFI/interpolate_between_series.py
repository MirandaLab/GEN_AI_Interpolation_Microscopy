"""
This script uses the CDFI model to create interpolated images between the images in a time series.
"""
import os
import argparse
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite

from models.cdfi_adacof import CDFI_adacof
from common_utils import move_images

def parse_args():
    parser = argparse.ArgumentParser(description='Video Interpolation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')

    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./interpolated_images')
    parser.add_argument('--cycles', type=int, default=2)

    return parser.parse_args()

def interpolate_imgs(input_dir, output_dir, model):
    """
    Interpolates images between the images in the series.
    It is expected that the series of the images in the input_dir match the order when the images are arranged in the
    lexicographical order.

    Parameters:
    input_dir(String) - path to the series of original frames.
    output_dir(String) - path where the interpolated images will be stored.
    model - CDFI model for image interpolation.

    Returns:
    None
    """
    image_names = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
    image_names.sort()
    print(f"Found {len(image_names)} images.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    img_count = 0
    img_index = 0
    frame2 = None
    while img_index < len(image_names)-1:
        print(f"Interpolating an image between {image_names[img_index]} and {image_names[img_index+1]}", end='\r', flush=True)

        frame_name1 = input_dir + '/' + image_names[img_index]
        frame_name2 = input_dir + '/' + image_names[img_index+1]

        frame1 = transform(Image.open(frame_name1)).unsqueeze(0).cuda()
        frame2 = transform(Image.open(frame_name2)).unsqueeze(0).cuda()

        model.eval()
        with torch.no_grad():
            frame_out = model(frame1, frame2)

        imwrite(frame1.clone(), f"{output_dir}/{img_count:05}.png" , range=(0, 1))
        imwrite(frame_out.clone(), f"{output_dir}/{(img_count+1):05}.png" , range=(0, 1))
        img_count += 2
        img_index += 1
    imwrite(frame2.clone(), f"{output_dir}/{img_count:05}.png" , range=(0, 1))

if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    transform = transforms.Compose([transforms.ToTensor()])
    
    model = CDFI_adacof(args).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

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
