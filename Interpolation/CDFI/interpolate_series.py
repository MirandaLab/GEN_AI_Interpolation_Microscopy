"""
Author: Swaraj Kaondal, Miranda lab
Adapted from interpolate_video.py:
Source: https://github.com/tding1/CDFI/blob/main/interpolate_video.py
Original work by tding1, modified for custom requirements.

Description:
This script uses the CDFI model to create interpolated images of all the real images in a time series.
It interpolates an image in the seires by taking i and i+2 image and interpolating the i+1 image of that series.
"""
import os
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image as imwrite

from models.cdfi_adacof import CDFI_adacof


def parse_args():
    parser = argparse.ArgumentParser(description='Video Interpolation')

    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/CDFI_adacof.pth')

    parser.add_argument('--kernel_size', type=int, default=11)
    parser.add_argument('--dilation', type=int, default=2)

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./interpolated_images')

    return parser.parse_args()

def interpolate_imgs(input_dir, output_dir, model):
    """
    Creates a series of interpolated images by taking i and i+2 image in a seires and interpolating the i+1 image.
    It is expected that the series of the images in the input_dir match the order when the images are arranged in the
    lexicographical order.

    Parameters:
    input_dir(String) - path to the series of original frames.
    output_dir(String) - path where the interpolated images will be stored.
    model - pre-trained model.

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

        frame_name1 = input_dir + '/' + image_names[img_index]
        frame_name2 = input_dir + '/' + image_names[img_index+2]

        frame1 = transform(Image.open(frame_name1)).unsqueeze(0).cuda()
        frame2 = transform(Image.open(frame_name2)).unsqueeze(0).cuda()

        model.eval()
        with torch.no_grad():
            frame_out = model(frame1, frame2)

        imwrite(frame_out.clone(), f"{output_dir}/{image_names[img_index+1]}" , range=(0, 1))
        img_index += 1

if __name__ == "__main__":
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    transform = transforms.Compose([transforms.ToTensor()])
    
    model = CDFI_adacof(args).cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    interpolate_imgs(args.input_dir, args.output_dir, model)
