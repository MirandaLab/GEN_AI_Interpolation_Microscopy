# Frame interpolation using RIFE

## Installation instructions using conda

If you have an older `rife` environment you can remove it with `conda env remove -n rife` before creating a new one.

1. Open an command prompt which has `conda` for **python 3** in the path
2. Create a new environment with `conda create --name rife python=3.11`.
3. To activate this new environment, run `conda activate rife`
4. Install all the packages under requirement.txt, `pip install -r requirements.txt`

## Interpolate images to check interpolation quality

The following steps create interpolated images of every original image except the first and the last image in the series. An interpolated image i is created using the i-1 and i+1 image as the input in the series.

1. Activate rife conda environment `conda activate rife`.
2. Run the following command by replacing `<input>` with the path to the images, `<output>` with the path of where the interpolated images will be stored and `<model>` with the path to pretrained model. `output_dir` and `model` are optional parameters with default values "./interpolated_images" and "./train_log" respectively. For other pretrained models refer [RIFE](https://github.com/hzwer/ECCV2022-RIFE?tab=readme-ov-file#cli-usage).
~~~sh
python interpolate_series.py --input_dir "<input>" --output_dir "<output>" --model_dir "<model>"
~~~
Note all the images are required to be in *PNG* format.

## Interpolate images between the series
1. Activate rife conda environment `conda activate rife`.
2. Run the following command by replacing `<input>` with the path to the images, `<output>` with the path of where the interpolated images will be stored, `<model>` with the path to pretrained model and `"<number of cycles>"` with the number of interpolation cycles to perfrom. `output_dir` and `model` are optional parameters with default values "./interpolated_images" and "./train_log" respectively. For other pretrained models refer [RIFE](https://github.com/hzwer/ECCV2022-RIFE?tab=readme-ov-file#cli-usage).

    Note: If number of interpolation cycle is n for m images. The resultant series has (2^n)*(m-1) + 1.
~~~sh
python interpolate_between_series.py --input_dir "<input>" --output_dir "<output>" --model_dir "<model>" --cycles "<number of cycles>"
~~~
Note all the images are required to be in *PNG* format.