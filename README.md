# Image Watermarking Tool

A robust digital image watermarking tool that embeds and verifies invisible watermarks using a combination of Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT). The tool adapts watermark strength based on image characteristics to maintain image quality while keeping watermark durability. The block based embedding and verification enables to see which part of the image was modified.

This tool should be viewed as a test project demonstrating the potential use of watermarking to protect against tampering and keep attribution to the original publisher of the image.
The standard image transformations (resizing and compression) that the imnage suffers trough social media and other media sharing pipelines should not erase the signature, making it suitable for online use.

## Features

- Invisible watermarking that preserves image quality
- Automatic image resizing for consistent processing
- Block-based watermark embedding for localized tampering detection
- Adaptive watermark strength based on image luminance and texture
- Detailed verification reporting with visual block correlation grid
- Support for both color and grayscale images


## Requirements

- Python 3.x
- Required packages:
  - numpy
  - Pillow
  - PyWavelets
  - OpenCV
  - scipy

## Installation

1. Clone this repository
2. Install required packages:
```bash
pip install numpy Pillow PyWavelets opencv-python scipy
```

## Usage

### Command Line Interface

The tool provides two main operations: signing (watermarking) and verification.

#### Signing an Image

```bash
python ImgSignBlock.py sign -i input.jpg -o output.jpg -k your_secret_key
```

This will:
- Create a watermarked version of the image at `output.jpg`
- Generate a signature file at `output.jpg.sig`

#### Verifying an Image

```bash
python ImgSignBlock.py verify -i input.jpg -k your_secret_key -s signature.sig
```

### Command Line Arguments

- `sign`: Sign an image
- `verify`: Verify the signature
- `-i`, `--input`: Path to input image (required)
- `-o`, `--output`: Output path for signed image (required for sign action)
- `-k`, `--key`: Secret key for signing/verification (required)
- `-s`, `--signature`: Path to signature file (required for verify action)

## Verification Output

The verification process provides comprehensive results including:

- Signature validity check
- Overall image status (AUTHENTIC/MODIFIED/UNSIGNED)
- Statistical analysis:
  - Average correlation
  - Standard deviation
  - Number of valid blocks
- Visual correlation grid showing:
  - ██ - Authentic blocks (correlation > 0.6)
  - ▓▓ - Suspicious blocks (0.4-0.6)
  - ░░ - Modified blocks (< 0.4)

## Technical Details

### Watermarking Process

1. **Image Preprocessing**
   - Converts color images to YUV color space and uses Y (luminance) channel
   - Resizes images maintaining aspect ratio (max dimension: 2048px)

2. **Watermark Embedding**
   - Applies DWT to decompose image into frequency sub-bands
   - Divides each sub-band into 4x4 grid of blocks
   - Applies DCT to each block
   - Embeds watermark in mid-frequency DCT coefficients
   - Follows (HVS) Human visual system model
   - Adjusts watermark strength based on:
     - Block luminance (darker/brighter areas)
     - Texture complexity (flat vs. detailed areas)

3. **Verification**
   - Extracts watermark using same process
   - Calculates correlation between extracted and expected watermark
   - Provides block-by-block analysis for tampering detection

### Parameters

Key parameters that affect watermark behavior:
- `max_dimension`: Determines the max image resolution. (Default 2048). Adjusted for most WEB purposes, but can be changed in code
- `strength`: Overall watermark embedding strength (default: 3)
- `threshold`: Correlation threshold for verification (default: 0.3)
- `grid_size`: Number of blocks per dimension (4x4 grid)
- `block_size`: Size of DCT blocks (8x8)

### TODO
- Further optimise the parameters for a better balance between robustnes and visibility
- Experiment with a single band (HH?) embedding for greater sensitivity to modification.
- Implement a more sophisticated perceptual mask (Variations inside DCT blocks)
- Use multiple arrays to capture the whole middle frequency DCT zone

### REFERENCES

- Alomoush, Khashan, Alrosan, Attar et al. Digital image watermarking using discrete cosine transformation based linear modulation. 
  - Journal of Cloud Computing 12, 96 (2023).
- Xin Li, Xingjun Wang, Anqi Chen, Linghao Xiao et al. A Simplified and Robust DCT-based Watermarking Algorithm 
  - International Journal of Electronics and Communications, March 2017
- Ali Al-Haj. Combined DWT-DCT digital image watermarking. 
  - Journal of Computer Science 3(9)(9), September 2007
- Verma, A. Image-Watermarking-using-DCT. 
  - GitHub repository (2020). [Link](https://github.com/arooshiverma/Image-Watermarking-using-DCT)
- Jadavpur University, DWT-DCT-Digital-Image-Watermarking. 
  - GitHub repository (2021). [Link](https://github.com/diptamath/DWT-DCT-Digital-Image-Watermarking)


