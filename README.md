﻿# AdaptiveMeanFilter
# Adaptive Mean Filter

## Project Description
This repository contains Python scripts that perform image denoising using adaptive mean filters. The provided script implements various adaptive filtering techniques to process noisy images and improve their quality. The repository includes detailed implementations of these techniques and provides example usage on sample images.

## Files
- `Hikmet_Terzioglu_161101071_HW5.py`: Python script implementing the adaptive mean filtering techniques.
- `noisyImage_Gaussian.jpg`: Sample noisy image used for testing the script.
- `lena_grayscale_hq.jpg`: Sample grayscale image used for comparison.
- `rapor.pdf`: Project report describing the implementation and results of the adaptive mean filtering techniques.

## Python Script Descriptions

### `Hikmet_Terzioglu_161101071_HW5.py`
This Python script contains functions to apply various adaptive mean filtering techniques to noisy images. The main functions are `question_1()` and `question_2()`, each performing different adaptive filtering tasks.

#### Key Features:
- **`question_1()`**: Applies an adaptive mean filter to a noisy image and compares the results with OpenCV's built-in filtering functions.
- **`question_2()`**: Implements a more advanced adaptive mean filtering technique, comparing the output with different kernel sizes.

#### Example Workflow

1. **Applying Adaptive Mean Filter (question_1)**
   - The function reads a noisy image, applies an adaptive mean filter, and compares the output with OpenCV's blur and GaussianBlur functions.
   - Displays the Peak Signal-to-Noise Ratio (PSNR) values to evaluate the filter performance.

2. **Advanced Adaptive Mean Filter (question_2)**
   - The function reads a noisy image and applies an advanced adaptive mean filter with varying kernel sizes.
   - Displays the PSNR values for different kernel sizes to evaluate the filter performance.

### Example Usage
To run the script and perform the adaptive mean filtering tasks, execute the following commands:

Detailed Function Descriptions
# question_1()
This function applies an adaptive mean filter to a given noisy image and compares the results with OpenCV's blur and GaussianBlur functions.

Steps:
Read the noisy image using OpenCV.
Normalize the image and add padding for the filter kernel.
Apply the adaptive mean filter by calculating the local variance and mean.
Compare the output with OpenCV's blur and GaussianBlur functions.
Display the PSNR values for each filter.


# question_2()
This function implements an advanced adaptive mean filter with varying kernel sizes.

Steps:
Read the noisy image using OpenCV.
Add padding to the image for the filter kernel.
Apply the adaptive mean filter with different kernel sizes.
Display the PSNR values for each kernel size.
