import torch
import torchvision.transforms as T
from PIL import Image
import albumentations as A
import numpy as np
import cv2
from innitius_enhance_shadow_light_specular import (
    add_uneven_illumination,
    add_random_shadows,
    add_streak_reflections,
    add_elliptical_reflections
)

alb_transforms = [
    ("RandomBrightnessContrast", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.2, p=1.0)),
    ("GaussianBlur", A.Blur(blur_limit=(3, 5), p=1.0)),
    ("GaussNoise", A.GaussNoise(std_range=(0.05, 0.1), p=1.0)),
    ("MotionBlur", A.MotionBlur(blur_limit=(3, 5), p=1.0))
]

custom_transforms = [
    ("UnevenIllumination", add_uneven_illumination),
    ("RandomShadows", add_random_shadows),
    ("StreakReflections", add_streak_reflections),
    ("EllipticalReflections", add_elliptical_reflections),
]

todas = alb_transforms + custom_transforms
nombre_transformaciones = [n for n, _ in todas]

def aplicar_transformacion(tensor_img, nombre_transformacion, target_size=None):
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    for nombre, transformacion in todas:
        if nombre == nombre_transformacion:
            if isinstance(transformacion, A.BasicTransform):
                img_np = A.Compose([transformacion])(image=img_np)['image']
            else:
                img_np = transformacion(img_np)
            break

    if target_size is not None:
        img_np = cv2.resize(img_np, target_size)

    tensor_resultado = T.ToTensor()(Image.fromarray(img_np)).unsqueeze(0)
    return tensor_resultado



def apply_median_filter(image, ksize=3):
    return cv2.medianBlur(image, ksize)


def clahe_lab(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.012, tileGridSize=(3, 3))
    l_clahe = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)


def apply_sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def reduce_saturation(image, scale=0.9):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def image_enhancement(image_np):
    if image_np is None:
        print("Error: La imagen es None")
        return None
    denoised = apply_median_filter(image_np)
    contrast_corrected = clahe_lab(denoised)
    reduced_saturation = reduce_saturation(contrast_corrected, scale=0.9)
    sharpened = apply_sharpening(reduced_saturation)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)