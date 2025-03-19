"""
Author: Swaraj Kaondal, Miranda lab

Description:
This script uses the FILM model to create interpolated images between the images in a time series.
"""
import os
import argparse
import shutil
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import interpolator as interpolator_lib
import util
from common_utils import move_images

def parse_args():
    parser = argparse.ArgumentParser(description='Video Interpolation')

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./interpolated_images")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cycles', type=int, default=2)

    return parser.parse_args()

def interpolate_imgs(input_dir, output_dir, model_path):
    """
    Interpolates images between the images in the series.
    It is expected that the series of the images in the input_dir match the order when the images are arranged in the
    lexicographical order.

    Parameters:
    input_dir(String) - path to the series of original frames.
    output_dir(String) - path where the interpolated images will be stored.
    model_path(String) - pretrained model path.

    Returns:
    None
    """
    image_names = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
    image_names.sort()
    print(f"Found {len(image_names)} images")
    
    os.makedirs(output_dir, exist_ok=True)
    
    interpolator = interpolator_lib.Interpolator(
            model_path=model_path,
            align=64,
            block_shape=[1, 1])
    
    img_count = 0
    img_index = 0
    image_2 = None
    while img_index < len(image_names)-1:
        print(f"Interpolating an image between {image_names[img_index]} and {image_names[img_index+1]}", end='\r', flush=True)

        frame_name1 = input_dir + '/' + image_names[img_index]
        frame_name2 = input_dir + '/' + image_names[img_index+1]

        image_1 = util.read_image(frame_name1)
        image_batch_1 = np.expand_dims(image_1, axis=0)

        image_2 = util.read_image(frame_name2)
        image_batch_2 = np.expand_dims(image_2, axis=0)

        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

        util.write_image(f"{output_dir}/{img_count:05}.png", image_1)
        util.write_image(f"{output_dir}/{(img_count+1):05}.png", mid_frame)
        img_index += 1
        img_count += 2

    util.write_image(f"{output_dir}/{img_count:05}.png", image_2)

if __name__ == "__main__":
    args = parse_args()

    temp_dir = "./temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    move_images(args.input_dir, temp_dir)

    cycle_count = 0
    while cycle_count < args.cycles:
        print(f"\n\nCycle number {cycle_count+1}\n")
        interpolate_imgs(temp_dir, args.output_dir, args.model_path)

        shutil.rmtree(temp_dir)
        if cycle_count < args.cycles - 1:
            os.makedirs(temp_dir)
            move_images(args.output_dir, temp_dir)
            shutil.rmtree(args.output_dir)

        cycle_count += 1