# --- Librerías estándar ---
import os
import time
import random

# --- Numéricas y científicas ---
import numpy as np
import cv2

# --- Deep Learning ---
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms as T

# --- Imagen y visualización ---
from PIL import Image
import albumentations as A

# --- Métricas ---
import piq
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim
from skimage.measure import shannon_entropy

# --- Visualización web ---
import streamlit as st


def add_uneven_illumination(image, intensity=None, center=None, sigma=None):
    if intensity is None:
        intensity = random.uniform(0.5, 1.5)
    if center is None:
        h, w = image.shape[:2]
        center = (random.randint(0, w), random.randint(0, h))
    if sigma is None:
        h, w = image.shape[:2]
        max_dim = max(h, w)
        sigma = random.uniform(max_dim * 0.1, max_dim * 0.3)

    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    gaussian = np.exp(-((X - center[0])**2 + (Y - center[1])**2) / (2 * sigma**2))
    gaussian = (gaussian - gaussian.min()) / (gaussian.max() - gaussian.min())
    illumination = 1 + (gaussian * intensity)
    min_illumination = 0.6
    illumination = min_illumination + (1 - min_illumination) * illumination
    result = image.copy().astype(float)
    result = result * illumination[:, :, np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)

def add_random_shadows(image, num_shadows=None, shadow_intensity=None):
    if num_shadows is None:
        num_shadows = random.randint(1, 3)
    if shadow_intensity is None:
        shadow_intensity = random.uniform(0.4, 0.8)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_shadows):
        gradient_type = random.choice(['linear', 'radial'])

        if gradient_type == 'linear':
            angle = random.uniform(0, 2 * np.pi)
            x = np.linspace(0, w - 1, w)
            y = np.linspace(0, h - 1, h)
            X, Y = np.meshgrid(x, y)
            gradient = (X * np.cos(angle) + Y * np.sin(angle)) / np.sqrt(w ** 2 + h ** 2)
            gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)
        else:
            center = (random.randint(-w // 2, w * 3 // 2), random.randint(-h // 2, h * 3 // 2))
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            gradient = dist_from_center / np.sqrt(w ** 2 + h ** 2)
            gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=w // 8)

        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        shadow_mask = 1 - (gradient * (1 - shadow_intensity))
        result *= shadow_mask[:, :, np.newaxis]

    return np.clip(result, 0, 255).astype(np.uint8)

def add_elliptical_reflections(image, num_reflections=None, intensity=None):
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        reflection_mask = np.zeros((h, w), dtype=np.float32)
        center = (random.randint(0, w), random.randint(0, h))
        major_axis = random.randint(w // 20, w // 5)
        minor_axis = random.randint(w // 30, major_axis)
        angle = random.uniform(0, 180)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(temp_mask, center, (major_axis, minor_axis), angle, 0, 360, 255, -1)
        reflection_mask = cv2.GaussianBlur(temp_mask.astype(float) / 255.0, (0, 0), sigmaX=random.uniform(5, 15))
        reflection_mask = reflection_mask / reflection_mask.max() if reflection_mask.max() > 0 else reflection_mask

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            result[:, :, c] += contribution

    return np.clip(result, 0, 255).astype(np.uint8)

def add_streak_reflections(image, num_reflections=None, intensity=None):
    if num_reflections is None:
        num_reflections = random.randint(1, 3)
    if intensity is None:
        intensity = random.uniform(0.3, 0.7)

    result = image.copy().astype(float)
    h, w = image.shape[:2]

    for _ in range(num_reflections):
        start = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        ctrl = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        end = (random.randint(-w // 4, w * 5 // 4), random.randint(-h // 4, h * 5 // 4))
        pts = np.array([start, ctrl, end], dtype=np.float32)
        curve = np.array([(1 - t)**2 * pts[0] + 2 * (1 - t) * t * pts[1] + t**2 * pts[2] for t in np.linspace(0, 1, 100)], dtype=np.int32)

        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.polylines(temp_mask, [curve], False, 255, random.randint(5, 20))
        reflection_mask = cv2.GaussianBlur(temp_mask.astype(float) / 255.0, (0, 0), sigmaX=random.uniform(5, 15))
        reflection_mask = reflection_mask / reflection_mask.max() if reflection_mask.max() > 0 else reflection_mask

        reflection_color = np.array([random.uniform(0.8, 1.0) for _ in range(3)])
        for c in range(3):
            channel = result[:, :, c]
            contribution = (255 - channel) * reflection_mask * intensity * reflection_color[c]
            result[:, :, c] += contribution

    return np.clip(result, 0, 255).astype(np.uint8)

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
nombre_transformaciones = [name for name, _ in todas]

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(
            True))  # self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=2, dilation=2), nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(
            True))  # self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=2, dilation=2), nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(256))

        # Decoder
        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, 3, padding=1),
                                  nn.Sigmoid())  # Change the output channels to 64

        self.residual = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # salida residual
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.enc1(x)  # (B, 32, H, W)
        x2 = self.pool1(x1)  # (B, 32, H/2, W/2)
        x3 = self.enc2(x2)  # (B, 64, H/2, W/2)
        x4 = self.pool2(x3)  # (B, 64, H/4, W/4)
        x5 = self.enc3(x4)  # (B, 128, H/4, W/4)

        y = self.dec1(x5)  # (B, 64, H/4, W/4)
        y = self.up1(y)  # (B, 64, H/2, W/2)
        y = torch.cat([y, x3], dim=1)  # (B, 128, H/2, W/2)
        y = self.dec2(y)  # (B, 32, H/2, W/2)
        y = self.up2(y)  # (B, 32, H, W)
        y = torch.cat([y, x1], dim=1)  # (B, 64, H, W)
        y = self.dec3(y)  # (B, 3, H, W)
        residual = self.tanh(self.residual(y))  # Now 'y' has 64 channels
        out = torch.clamp(x + residual, 0.0, 1.0)  # asegúrate que los valores estén entre 0 y 1
        return out

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)
modelo = ConvAutoencoder().to(dispositivo)
modelo.eval()
# ---------------------------
# FUNCTIONS
# ---------------------------

def cargar_imagen(img_file, size=(256, 256)):
    if isinstance(img_file, Image.Image):
        imagen = img_file.convert('RGB')
    else:
        imagen = Image.open(img_file).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

    imagen_tensor = transform(imagen).unsqueeze(0)
    return imagen_tensor

def tensor_a_pil(tensor):
    tensor = tensor.squeeze(0).cpu().clamp(0, 1)
    return T.ToPILImage()(tensor)

def aplicar_transformacion_especifica(tensor_img, nombre_transformacion):
    img_np = tensor_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)

    for nombre, transformacion in todas:
        if nombre == nombre_transformacion:
            if isinstance(transformacion, A.BasicTransform):
                img_np = A.Compose([transformacion])(image=img_np)['image']
            else:
                img_np = transformacion(img_np)
            break

    tensor_resultado = T.ToTensor()(Image.fromarray(img_np)).unsqueeze(0)
    return tensor_resultado

 #METRICS WHIT REFERENCE
def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True, data_range=1.0)
    return score

def normalize_metrics(psnr, mse, lpips_score, ssim_score):
    psnr_norm = 1 if psnr >= 30 else max(0, min(1, (psnr - 15) / 15))
    mse_norm = 1 if mse <= 0.2 else max(0, min(1, 1 - (mse - 0.2) / 0.8))
    lpips_norm = 1 if lpips_score <= 0.1 else max(0, min(1, 1 - (lpips_score - 0.1) / 0.9))
    ssim_norm = 1 if ssim_score >= 0.9 else max(0, min(1, (ssim_score - 0.3) / 0.6))
    return psnr_norm, mse_norm, lpips_norm, ssim_norm

def calcular_metricas_normalizadas(img1_pil, img2_pil, img1_tensor, img2_tensor):
    img1_np = np.array(img1_pil.resize((320, 320))).astype(np.float32) / 255.0
    img2_np = np.array(img2_pil.resize((320, 320))).astype(np.float32) / 255.0

    if img1_np.ndim == 2:
        img1_np = np.stack([img1_np] * 3, axis=-1)
    if img2_np.ndim == 2:
        img2_np = np.stack([img2_np] * 3, axis=-1)

    ssim_raw = calculate_ssim(img1_np, img2_np)
    psnr_raw = psnr_sk(img1_np, img2_np, data_range=1.0)
    mse_raw = calculate_mse(img1_np, img2_np)

    def normalizar_lpips(t): return (t * 2) - 1

    lpips_raw = lpips_fn(
        normalizar_lpips(img1_tensor).to(dispositivo),
        normalizar_lpips(img2_tensor).to(dispositivo)
    ).item()

    return normalize_metrics(psnr_raw, mse_raw, lpips_raw, ssim_raw)

 #METRICS WHITOUT REFERENCE
def calculate_entropy(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    entropy = shannon_entropy(img)
    entropy_normalized = np.clip(entropy * 100 / 8, 0, 100)  # 8 bits = max entropy 8
    return entropy_normalized

def calculate_sharpness(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()

    mean_intensity = np.mean(img)
    normalized_sharpness = variance / (mean_intensity + 1e-5)
    sharpness_normalized = np.clip(normalized_sharpness * 100, 0, 100)

    return sharpness_normalized

def calculate_contrast(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = img.std()
    contrast_normalized = np.clip(contrast * 100 / 255, 0, 100)
    return contrast_normalized
# Función para calcular el colorido
def calculate_colorfulness(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(img_rgb)
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    rg_std = np.std(rg)
    yb_std = np.std(yb)
    colorfulness = np.sqrt(rg_mean ** 2 + yb_mean ** 2) + 0.3 * np.sqrt(rg_std ** 2 + yb_std ** 2)
    colorfulness_normalized = np.clip(colorfulness * 100 / 255, 0, 100)
    return colorfulness_normalized

#simulation of neural networks
def calcular_brisque(img):
    # Versión simplificada normalizada igual que en normalize_metrics
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).float()
    elif isinstance(img, torch.Tensor):
        img_tensor = img.float()
    else:
        raise TypeError("La imagen debe ser un numpy array o un tensor de PyTorch.")

    if img_tensor.ndim == 3:
        if img_tensor.shape[0] not in [1, 3] and img_tensor.shape[2] in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)
    elif img_tensor.ndim == 2:
        img_tensor = img_tensor.unsqueeze(0)

    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.0

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    img_tensor = img_tensor.to(dispositivo)

    score = piq.brisque(img_tensor, data_range=1.0).item()

    # Normalización: ideal <= 0.25, decrece linealmente hasta 1
    if score <= 0.25:
        return 1
    else:
        return max(0, 1 - (score - 0.25) / (1.0 - 0.25))

def niqe(image, max_niqe=100):
    if not isinstance(image, np.ndarray):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape
    block_size = 32
    scores = []

    natural_mean = 127
    natural_std = 50

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image_gray[i:i + block_size, j:j + block_size]
            mu = np.mean(block)
            sigma = np.std(block)
            score = np.sqrt((mu - natural_mean) ** 2 + (sigma - natural_std) ** 2)
            scores.append(score)

    mean_score = np.mean(scores)
    score_clipped = np.clip(mean_score, 0, max_niqe)
    score_pct = (1 - score_clipped / max_niqe) * 100

    # Normalización: ideal <= 0.25, decae linealmente hasta 8.0 (ajustado a escala 0-1)
    niqe_val = score_clipped / max_niqe * 8  # normalizo a rango similar que normalize_metrics
    if niqe_val <= 0.25:
        return 1
    else:
        return max(0, 1 - (niqe_val - 0.25) / (8.0 - 0.25))

def nima(image):
    if not isinstance(image, np.ndarray):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contraste = np.std(image_gray) / 128 * 10
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    nitidez = np.var(laplacian)
    nitidez = min(nitidez / 1000 * 10, 10)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturacion = np.mean(image_hsv[:, :, 1]) / 255 * 10

    nima_score = 0.4 * contraste + 0.4 * nitidez + 0.2 * saturacion
    nima_score = np.clip(nima_score, 0, 10)

    # Normalización ideal >= 8, sube linealmente desde 0 a 1 entre 0 y 8
    return min(nima_score / 8, 1)

def normalize_metrics2(metrics):
    normalized = {}

    # Entropía: ideal >= 7
    e = metrics["Entropía"]
    normalized["Entropía"] = 1 if e >= 7 else max(0, e / 7)

    # Contraste: ideal entre 40 y 70, decae fuera de ese rango
    c = metrics["Contraste"]
    if 40 <= c <= 70:
        normalized["Contraste"] = 1
    elif c < 40:
        normalized["Contraste"] = max(0, c / 40)  # proporcional desde 0 a 40
    else:  # c > 70
        normalized["Contraste"] = max(0, 1 - (c - 70) / 30)  # decae de 70 a 100

    # Nitidez: ideal entre 60 y 70, decae fuera
    n = metrics["Nitidez"]
    if 60 <= n <= 70:
        normalized["Nitidez"] = 1
    elif n < 60:
        normalized["Nitidez"] = max(0, n / 60)
    else:  # n > 70
        normalized["Nitidez"] = max(0, 1 - (n - 70) / 30)

    # Colorido: ideal entre 45 y 50
    col = metrics["Colorido"]
    if 45 <= col <= 50:
        normalized["Colorido"] = 1
    elif col < 45:
        normalized["Colorido"] = max(0, col / 45)
    else:  # col > 50
        normalized["Colorido"] = max(0, 1 - (col - 50) / 50)

    # Brisque: ideal <= 0.25, decrece suavemente a 0 cuando > 1.0
    b = metrics["Brisque"]
    if b <= 0.25:
        normalized["Brisque"] = 1
    else:
        normalized["Brisque"] = max(0, 1 - (b - 0.25) / (1.0 - 0.25))

    # Niqe: ideal <= 0.25, decae hasta 0 en 8.0
    ni = metrics["Niqe"]
    if ni <= 0.25:
        normalized["Niqe"] = 1
    else:
        normalized["Niqe"] = max(0, 1 - (ni - 0.25) / (8.0 - 0.25))

    # Nima: ideal >= 8, sube desde 0 a 1 entre 0 y 8
    nima_val = metrics["Nima"]
    normalized["Nima"] = min(nima_val / 8, 1)

    return normalized

def calculate_1image_metrics(image):
    entropy = calculate_entropy(image)
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    colorfulness = calculate_colorfulness(image)

    # Asegúrate de tener definidas o importar estas funciones:
    valor_brisque = calcular_brisque(image)
    valor_niqe = niqe(image)
    valor_nima = nima(image)

    metrics = {
        "Entropía": entropy,
        "Nitidez": sharpness,
        "Contraste": contrast,
        "Colorido": colorfulness,
        "Brisque": valor_brisque,
        "Niqe": valor_niqe,
        "Nima": valor_nima
    }

    normalized = normalize_metrics2(metrics)
    return normalized


#TRADITIONAL IMAGE ENHACEMENT PIPELINE
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

def image_enhancement_pipeline(image_np):
    # Esperamos una imagen en formato NumPy (H, W, C)
    if image_np is None:
        print("Error: La imagen es None")
        return None
    denoised = apply_median_filter(image_np)
    contrast_corrected = clahe_lab(denoised)
    reduced_saturation = reduce_saturation(contrast_corrected, scale=0.9)
    sharpened = apply_sharpening(reduced_saturation)
    normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
    return normalized


# ---------------------------
# EXPLANATION OF TRANSFORMATIONS
# ---------------------------

explicaciones_degradaciones = {
    "RandomBrightnessContrast": (
        "Esta degradación simula variaciones en el brillo y contraste de la imagen. "
        "Estas alteraciones pueden originarse por fluctuaciones en la intensidad o dirección de la fuente de luz "
        "integrada en la sonda, o por condiciones anatómicas internas, como la presencia de fluidos o tejidos que absorben o dispersan la luz, "
        "modificando el rango dinámico captado por el sensor y afectando la visibilidad de estructuras."
    ),
    "GaussianBlur": (
        "El desenfoque gaussiano reproduce la pérdida de nitidez que se produce cuando la imagen capturada no está bien enfocada. "
        "Esto puede ser causado por lentes sucias o empañadas, imprecisiones en el sistema óptico de la sonda, "
        "o fallos en el mecanismo de enfoque automático, resultando en una imagen menos definida."
    ),
    "GaussNoise": (
        "Este ruido simula interferencias electrónicas o ruido inherente al sensor de imagen, "
        "especialmente perceptible en condiciones de baja iluminación o cuando se aumenta la ganancia para mejorar la señal. "
        "Puede generar un efecto de granulado que dificulta la interpretación clínica."
    ),
    "MotionBlur": (
        "El desenfoque por movimiento ocurre cuando la sonda o el tejido examinado se desplazan durante la captura, "
        "lo que puede suceder por temblores del operador, movimientos involuntarios del paciente o desplazamientos fisiológicos, "
        "provocando una imagen borrosa que afecta la detección precisa de detalles."
    ),
    "UnevenIllumination": (
        "Esta degradación simula una iluminación desigual en la imagen, generando áreas más oscuras o sobreexpuestas. "
        "Se produce por una colocación subóptima de la fuente luminosa en la sonda o por la orientación incorrecta de esta, "
        "lo que limita la homogeneidad de la luz y dificulta la visualización uniforme de la escena."
    ),
    "RandomShadows": (
        "Simula sombras parciales que se generan cuando estructuras anatómicas, como pliegues, vasos sanguíneos o acumulaciones de fluidos, "
        "bloquean parcial o totalmente la luz, causando zonas de baja luminosidad que pueden ocultar detalles importantes."
    ),
    "StreakReflections": (
        "Imita reflejos lineales provocados por superficies húmedas o brillantes dentro del campo visual, "
        "como la mucosa o líquidos corporales, que generan destellos al reflejar la luz. "
        "Estos reflejos pueden superponerse a la imagen clínica y complicar la interpretación."
    ),
    "EllipticalReflections": (
        "Representa reflejos especulares con forma elíptica típicos de superficies convexas o ciertas geometrías internas, "
        "como la pared uterina o estructuras ováricas. Estos reflejos muy brillantes pueden saturar zonas de la imagen "
        "y dificultar la diferenciación de tejidos."
    )
}
