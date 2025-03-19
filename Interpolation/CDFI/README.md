# Frame interpolation using CDFI

## Installation instructions using conda

If you have an older `cdfi` environment you can remove it with `conda env remove -n cdfi` before creating a new one.

1. Open an command prompt which has `conda` for **python 3** in the path
2. Create a new environment with `conda create --name cdfi python=3.8`
3. To activate this new environment, run `conda activate cdfi`
4. Install torch and torchvision via pytorch's distribution, `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html`
5. Install all the packages under requirement.txt, `pip install -r requirements.txt` 

## Interpolate images to check interpolation quality

The following steps create interpolated images of every original image except the first and the last image in the series. An interpolated image i is created using the i-1 and i+1 image as the input in the series.

1. Activate cdfi conda environment `conda activate cdfi`.
2. Run the following command by replacing `<input>` with the path to the images and `<output>` with the path of where the interpolated images will be stored. `output_dir` is set to "./interpolated_images" by default.
~~~sh
python interpolate_series.py --input_dir "<input>" --output_dir "<output>"
~~~

Note all the images are required to be in *PNG* format.

## Interpolate images between the series
1. Activate cdfi conda environment `conda activate cdfi`.
2. Run the following command by replacing `<input>` with the path to the images, `<output>` with the path of where the interpolated images will be stored and `"<number of cycles>"` with the number of interpolation cycles to perfrom. Default number of cycles is 2.
Note: If number of interpolation cycle is n for m images. The resultant series has (2^n)*(m-1) + 1.
~~~sh
python interpolate_between_series.py --input_dir "<input>" --output_dir "<output>" --cycles "<number of cycles>"
~~~
Note all the images are required to be in *PNG* format.