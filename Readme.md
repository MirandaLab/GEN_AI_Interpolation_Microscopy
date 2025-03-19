# CUDA setup
## Install CUDA Toolkit
1. Go to [https://developer.nvidia.com/cuda-11.2.1-download-archive](https://developer.nvidia.com/cuda-11.2.1-download-archive) and select your `Windows`.
2. Download and install `CUDA Tookit 11.2.1`.
3. Additional CUDA installation information available [here](https://docs.nvidia.com/cuda/archive/11.2.2/cuda-installation-guide-microsoft-windows/index.html).

## Install cuDNN
1. Go to [https://developer.nvidia.com/rdp/cudnn-download](https://developer.nvidia.com/rdp/cudnn-download).
2. Create a user profile (if needed) and login.
3. Select `cuDNN v8.1.0 (January 26th, 2021), for CUDA 11.0,11.1 and 11.2`.
4. Download [cuDNN Library for Widnows (x86)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip). 
5. Extract the contents of the zipped folder (it contains a folder named `cuda`) into `<INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\`. `<INSTALL_PATH>` points to the installation directory specified during CUDA Toolkit installation. By default, `<INSTAL_PATH> = C:\Program Files`.

## Environment Setup
Add the following paths to your 'Advanced System Settings' > 'Environment Variables ...' > Edit 'Path', and add:
1. <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin
2. <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\libnvvp
3. <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include
4. <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\extras\CUPTI\lib64
5. <INSTALL_PATH>\NVIDIA GPU Computing Toolkit\CUDA\v11.2\cuda\bin

# Interpolating images
After the CUDA setup interpolate images using [CDFI](Interpolation/CDFI/README.md), [FILM](Interpolation/FILM/README.md) or [RIFE](Interpolation/RIFE/README.md).
