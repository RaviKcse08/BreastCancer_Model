# data/transforms.py

from torchvision import transforms
import pydicom
from PIL import Image
import numpy as np

# Standard ImageNet normalization values (good for pre-trained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.05, contrast=0.05), # Subtle for medical
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def get_val_transforms(image_size):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

# Utility to read DICOM and convert to PIL Image
def dicom_to_pil(dicom_path):
    dicom_data = pydicom.dcmread(dicom_path)
    image_data = dicom_data.pixel_array

    # Handle MONOCHROME1 photometric interpretation (inverted grayscale)
    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        image_data = np.amax(image_data) - image_data

    # Normalize to 0-255 range and convert to uint8 for PIL
    image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min()) * 255.0
    return Image.fromarray(image_data.astype(np.uint8)).convert("RGB")