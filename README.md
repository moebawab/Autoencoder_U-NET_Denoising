# X-Ray Image Denoising with Self-Supervised U-Net

This repository provides three **self-supervised Residual U-Net models** for denoising grayscale X-ray images using:

* **Gaussian noise**
* **Poisson noise**
* **Salt-and-pepper noise**

The models are trained using the original images as the clean target. Synthetic noise is generated automatically during training, so paired noisy/clean images are not required.

---

## 1. Installation

Install the required dependencies:

```bash
pip install torch torchvision opencv-python numpy tqdm pytorch-msssim
```

A CUDA-enabled GPU is recommended but not required. The code automatically uses CUDA when available and otherwise falls back to CPU.

---

## 2. Prepare the Images

Create a folder called: "baseline/"

and place your grayscale X-ray images inside it.

Example:

```text
project/
│
├── baseline/
│   ├── image_001.png
│   ├── image_002.png
│   ├── image_003.png
│   └── ...
│
├── gaussian_denoising.py
├── poisson_denoising.py
└── salt_and_pepper_denoising.py
```


Supported formats:
.png
.jpg
.jpeg
.bmp
.tif
.tiff

Images are automatically converted to grayscale and resized to **296 × 296**.

---

## 3. Run Gaussian Denoising

Run: python gaussian_denoising.py 

The script:

1. Loads images from `baseline/`.
2. Generates Gaussian noise during training.
3. Trains the Residual U-Net.
4. Saves the trained model.
5. Applies the model to all images in `baseline/`.

Outputs: gaussian/

The denoised images are saved in: gaussian/


---

## 4. Run Poisson Denoising

Run: python poisson_denoising.py

The script:

1. Loads images from `baseline/`.
2. Generates Poisson noise during training.
3. Trains the Residual U-Net.
4. Saves the trained model.
5. Applies the model to all images in `baseline/`.


Outputs: poisson/ 

The Poisson noise strength can be adjusted using:

```python
vals = 255
```

Lower values produce stronger simulated Poisson noise.

---

## 5. Run Salt-and-Pepper Denoising

Run: python salt_and_pepper_denoising.py

The script:

1. Loads images from `baseline/`.
2. Generates Salt and Pepper noise during training.
3. Trains the Residual U-Net.
4. Saves the trained model.
5. Applies the model to all images in `baseline/`.


Outputs: salt_and_pepper/


The noise probability can be changed using:

```python
prob = 0.02
```

For example:

```python
prob = 0.05
```

will generate stronger salt-and-pepper noise.

---

## 6. Training Parameters

The main training parameters are defined directly in each script:

```python
batch_size = 8
learning_rate = 1e-4
```

The number of training epochs is controlled by:

```python
for epoch in range(1, 3):
```

Change this value according to your experiment, for example:
    for epoch in range(1, 51): --> for 50 epochs.

The training loss combines **L1 loss and SSIM**:

```text
Loss = L1 + (1 - SSIM)
```

---

## 7. Output Structure

After running all three models:

```text
project/
├── baseline/
│   ├── image_001.png
│   └── image_002.png
│
├── gaussian/
│   ├── image_001.png
│   └── image_002.png
│
├── poisson/
│   ├── image_001.png
│   └── image_002.png
│
├── salt_and_pepper/
│   ├── image_001.png
│   └── image_002.png
│
├── residual_unet_selfsupervised_gaussian.pth
├── residual_unet_selfsupervised_poisson.pth
└── residual_unet_selfsupervised_saltpepper.pth
```

Original filenames are preserved, allowing the denoised images to be directly matched with the corresponding images in `baseline/`.

---

## 8. Summary

Each script follows the same workflow:

```text
Original X-ray
      ↓
Synthetic noise
      ↓
Residual U-Net
      ↓
Denoised X-ray
```
-------------------------------------------------------------------
The three scripts differ only in the type of synthetic noise used for training.

**Note:** The current implementation is intended for research purposes. 
