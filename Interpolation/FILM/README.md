# Frame interpolation using FILM

## Installation instructions using conda

If you have an older `film` environment you can remove it with `conda env remove -n film` before creating a new one.

1. Open an command prompt which has `conda` for **python 3** in the path
2. Create a new environment with `conda create --name film pip python=3.9`
3. To activate this new environment, run `conda activate film`
4. Install all the packages under requirement.txt, `pip install -r requirements.txt` 
5. Install ffmpeg using from conda-forge, `conda install -c conda-forge ffmpeg`
6. Download the pretrained model from [here](https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy). The downloaded folder should have the following structure,
```
<pretrained_models>/
├── film_net/
│   ├── L1/
│   ├── Style/
│   ├── VGG/
├── vgg/
│   ├── imagenet-vgg-verydeep-19.mats
```

## Interpolate images to check interpolation quality

The following steps create interpolated images of every original image except the first and the last image in the series. An interpolated image i is created using the i-1 and i+1 image as the input in the series.

1. Activate film conda environment `conda activate film`.
2. Run the following command by replacing `<input>` with the path to the images, `<output>` with the path of where the interpolated images will be stored and `<model_path>` with the path to the downloaded folder. `output_dir` is set to "./interpolated_images" by default.
~~~sh
python interpolate_series.py --input_dir "<input>" --output_dir "<output>" --model_path "<pretrained_models>\film_net\Style\saved_model"
~~~

Note all the images are required to be in *PNG* format.

## Interpolate images between the series
1. Activate cdfi conda environment `conda activate cdfi`.
2. Run the following command by replacing `<input>` with the path to the images, `<output>` with the path of where the interpolated images will be stored, `<model_path>` with the path to the downloaded folder and `"<number of cycles>"` with the number of interpolation cycles to perfrom. Default number of cycles is 2.
Note: If number of interpolation cycle is n for m images. The resultant series has (2^n)*(m-1) + 1.
~~~sh
python interpolate_between_series.py --input_dir "<input>" --output_dir "<output>" --model_path "<pretrained_models>\film_net\Style\saved_model" --cycles "<number of cycles>"
~~~
Note all the images are required to be in *PNG* format.