"""
This script uses the FILM model to create interpolated images of all the real images in a time series.
It interpolates an image in the seires by taking i and i+2 image and interpolating the i+1 image of that series.
"""
import os
import argparse

import numpy as np
import interpolator as interpolator_lib
import util

def parse_args():
    parser = argparse.ArgumentParser(description='Video Interpolation')

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="./interpolated_images")
    parser.add_argument('--model_path', type=str, required=True)

    return parser.parse_args()

def interpolate_imgs(input_dir, output_dir, model_path):
    """
    Creates a series of interpolated images by taking i and i+2 image in a seires and interpolating the i+1 image.
    It is expected that the series of the images in the input_dir match the order when the images are arranged in the
    lexicographical order.

    Parameters:
    input_dir(String) - path to the series of original frames.
    output_dir(String) - path where the interpolated images will be stored.
    model_path(String) - pre-trained model path.

    Returns:
    None
    """
    image_names = [f for f in os.listdir(input_dir) if f.endswith(('.png'))]
    image_names.sort()
    print(f"Found {len(image_names)} images. Creating interpolated images of every image except the first and the last image.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    interpolator = interpolator_lib.Interpolator(
            model_path=model_path,
            align=64,
            block_shape=[1, 1])
    
    img_index = 0
    while img_index < len(image_names)-2:
        print(f"Interpolating {image_names[img_index+1]}", end='\r', flush=True)

        frame_name1 = input_dir + '/' + image_names[img_index]
        frame_name2 = input_dir + '/' + image_names[img_index+2]

        image_1 = util.read_image(frame_name1)
        image_batch_1 = np.expand_dims(image_1, axis=0)

        image_2 = util.read_image(frame_name2)
        image_batch_2 = np.expand_dims(image_2, axis=0)

        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

        util.write_image(f"{output_dir}/{image_names[img_index+1]}", mid_frame)
        img_index += 1

if __name__ == "__main__":
    args = parse_args()
    interpolate_imgs(args.input_dir, args.output_dir, args.model_path)