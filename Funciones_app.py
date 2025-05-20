import torch
import torchvision.transforms as T
from PIL import Image
import albumentations as A
import piq
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim
from innitius_enhance_shadow_light_specular import (
    add_uneven_illumination,
    add_random_shadows,
    add_streak_reflections,
    add_elliptical_reflections
)
from skimage.measure import shannon_entropy
import numpy as np
import cv2
import lpips
import torch
import torchvision.transforms as transforms

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)

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

# ---------------------------
# FUNCIONES AUXILIARES
# ---------------------------

def calcular_brisque(img):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(img, torch.Tensor):
        img_tensor = img.float()
    else:
        img_tensor = torch.from_numpy(img).float()

    if img_tensor.ndim == 3:
        if img_tensor.shape[0] not in [1, 3]:
            img_tensor = img_tensor.permute(2, 0, 1)
    if img_tensor.max() > 1.0:
        img_tensor = img_tensor / 255.

    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)

    score = piq.brisque(img_tensor.to(dispositivo), data_range=1.0).item()

    # Normalizamos a 0-100%
    score_norm = 100 - np.clip(score, 0, 100)  # Invertido: menor BRISQUE es mejor calidad
    return score_norm


def niqe(image, max_niqe=100):
    """
    Estimación manual de NIQE (No-Reference Image Quality Estimation).
    Devuelve el valor NIQE y su equivalencia como porcentaje de degradación (0% = perfecto, 100% = muy malo).

    Args:
        image (np.ndarray o tensor): Imagen en formato HxWxC o tensor CHW.
        max_niqe (float): Valor NIQE máximo razonable para normalización (ajustable). Por defecto: 100.

    Returns:
        tuple: (niqe_score, niqe_percentage)
    """
    # Convertir a numpy array con forma HxWxC y tipo uint8 si es tensor
    if not isinstance(image, np.ndarray):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    # Pasar a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape
    block_size = 32
    scores = []

    # Estadísticas naturales aproximadas
    natural_mean = 127
    natural_std = 50

    # Calcular score por bloque
    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            block = image_gray[i:i + block_size, j:j + block_size]
            mu = np.mean(block)
            sigma = np.std(block)
            score = np.sqrt((mu - natural_mean) ** 2 + (sigma - natural_std) ** 2)
            scores.append(score)

    niqe_score = np.mean(scores)
    niqe_percent = min((niqe_score / max_niqe) * 100, 100.0)

    return niqe_percent


def nima(image):
    """
    Calcula un score de calidad tipo NIMA donde 10 representa 100% de calidad percibida.

    Parámetros:
        image: Imagen en formato NumPy array o tensor de PyTorch.

    Retorna:
        float: Porcentaje de calidad entre 0% y 100%, donde 10 equivale a 100%.
    """
    # Convertir tensor de PyTorch a NumPy si es necesario
    if not isinstance(image, np.ndarray):
        image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

    # Convertir a escala de grises
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Contraste normalizado
    contraste = np.std(image_gray) / 128 * 10

    # Nitidez (varianza del laplaciano) normalizada
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    nitidez = np.var(laplacian)
    nitidez = min(nitidez / 1000 * 10, 10)

    # Saturación (canal S del HSV)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturacion = np.mean(image_hsv[:, :, 1]) / 255 * 10

    # Score ponderado (máximo 10)
    nima_score = 0.4 * contraste + 0.4 * nitidez + 0.2 * saturacion
    nima_score = np.clip(nima_score, 0, 10)

    # Convertir a porcentaje (donde 10 = 100%)
    return nima_score * 10
def cargar_imagen_cv2(img_np):
    img = Image.fromarray(img_np)
    return cargar_imagen(img)


def cargar_imagen(img_file, size=(256, 256)):
    # Verificamos si ya es una imagen PIL o no
    if isinstance(img_file, Image.Image):
        imagen = img_file.convert('RGB')
    else:
        imagen = Image.open(img_file).convert('RGB')

    # Definimos las transformaciones
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()  # Convierte a tensor y normaliza a [0, 1]
    ])

    imagen_tensor = transform(imagen).unsqueeze(0)  # Añade dimensión de batch
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

def calcular_metricas(img1_pil, img2_pil, img1_tensor, img2_tensor):
    img1_pil = img1_pil.convert('RGB')
    img2_pil = img2_pil.convert('RGB')

    img1_np = np.array(img1_pil.resize((320, 320))).astype(np.float32) / 255.0
    img2_np = np.array(img2_pil.resize((320, 320))).astype(np.float32) / 255.0

    if img1_np.ndim == 2:
        img1_np = np.stack([img1_np] * 3, axis=-1)
    if img2_np.ndim == 2:
        img2_np = np.stack([img2_np] * 3, axis=-1)

    # SSIM: 0-1 → %
    ssim_val = calculate_ssim(img1_np, img2_np) * 100

    # PSNR: 0-50 dB → %
    psnr_raw = psnr_sk(img1_np, img2_np, data_range=1.0)
    psnr_val = np.clip(psnr_raw * 2, 0, 100)  # 50 dB = 100%

    # MSE: 0-inf → %, invertido
    mse_raw = calculate_mse(img1_np, img2_np)
    mse_val = np.clip((mse_raw / 0.1) * 100, 0, 100)  # 0.1 es un límite razonable

    # LPIPS: 0-1 → %, invertido
    def normalizar_lpips(t):
        return (t * 2) - 1

    lpips_raw = lpips_fn(normalizar_lpips(img1_tensor).to(dispositivo),
                         normalizar_lpips(img2_tensor).to(dispositivo)).item()
    lpips_val = np.clip((lpips_raw) * 100, 0, 100)

    return ssim_val, psnr_val, mse_val, lpips_val

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


def calculate_mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def calculate_ssim(img1, img2):
    if len(img1.shape) == 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if len(img2.shape) == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(img1, img2, full=True, data_range=1.0)
    return score


def calculate_1image_metrics(image):
    entropy = calculate_entropy(image)
    sharpness = calculate_sharpness(image)
    contrast = calculate_contrast(image)
    colorfulness = calculate_colorfulness(image)
    return entropy, sharpness, contrast, colorfulness


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
# TRANSFORMACIONES
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
