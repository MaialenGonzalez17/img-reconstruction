import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from PIL import Image
import albumentations as A
import time
import piq
import torchvision.transforms.functional as TF
import cv2
from torchvision.transforms import ToTensor
from skimage.metrics import structural_similarity as ssim_sk, peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
import lpips
from innitius_enhance_shadow_light_specular import (
    add_uneven_illumination,
    add_random_shadows,
    add_streak_reflections,
    add_elliptical_reflections
)
import plotly.graph_objects as go
import pandas as pd
from skimage.measure import shannon_entropy

import torch.nn as nn
import torchvision.models as models

# ---------------------------
# TRANSFORMACIONES
# ---------------------------
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

todas = alb_transforms + custom_transforms
nombre_transformaciones = [n for n, _ in todas]

# ---------------------------
# MODELO AUTOENCODER
# ---------------------------
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(True))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(True))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(True))

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, padding=1), nn.ReLU(True))
        self.up1 = nn.Upsample(scale_factor=2)
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(128 + 128, 64, 3, padding=1), nn.ReLU(True))
        self.up2 = nn.Upsample(scale_factor=2)
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(64 + 64, 64, 3, padding=1), nn.Sigmoid())
        self.residual = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.pool1(x1)
        x3 = self.enc2(x2)
        x4 = self.pool2(x3)
        x5 = self.enc3(x4)

        y = self.dec1(x5)
        y = self.up1(y)
        y = torch.cat([y, x3], dim=1)
        y = self.dec2(y)
        y = self.up2(y)
        y = torch.cat([y, x1], dim=1)
        y = self.dec3(y)
        residual = self.tanh(self.residual(y))
        out = torch.clamp(x + residual, 0.0, 1.0)
        return out

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


import numpy as np
import cv2


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


def cargar_imagen(img_file):
    imagen = Image.open(img_file).convert('RGB')
    transform = T.Compose([
        T.Resize((320, 320)),
        T.ToTensor()
    ])
    return transform(imagen).unsqueeze(0)

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


# ---------------------------
# STREAMLIT UI SETUP
# ---------------------------

st.set_page_config(page_title="Restaurador de Imágenes", layout="wide")

# --- Estilos CSS para tablas grisáceas y tamaños de imagen ---
st.markdown("""
    <style>
    /* Fondo gris claro para tablas */
    table.dataframe tbody tr {
        background-color: #f5f5f5;
    }
    /* Alternar filas para mejor legibilidad */
    table.dataframe tbody tr:nth-child(even) {
        background-color: #e0e0e0;
    }
    /* Encabezados tabla */
    table.dataframe thead th {
        background-color: #b0b0b0;
        color: black;
        font-weight: bold;
    }
    /* Ajustar tamaño imágenes */
    .image-container img {
        max-width: 250px;
        height: auto;
    }
    /* Cabecera */
    header {
        display: flex;
        align-items: center;
        padding: 10px 20px;
        background-color: #0a3d62;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    header img {
        height: 50px;
        margin-right: 15px;
    }
    header h1 {
        font-size: 28px;
        margin: 0;
    }
    /* Pie de página */
    footer {
        margin-top: 50px;
        padding: 15px 20px;
        text-align: center;
        font-size: 12px;
        color: #888888;
        border-top: 1px solid #cccccc;
        font-family: 'Arial', sans-serif;
    }
    </style>
    
""", unsafe_allow_html=True)

# Cabecera con logo y título
st.markdown("""
<header>
     <h1>🧽 Restaurador de Imágenes </h1>
 
</header>
    <br><br>
""", unsafe_allow_html=True)

with st.expander("**Conocimientos básicos**"):
    st.markdown("""
**¿Cómo funciona el autoencoder?**  
Un autoencoder es una red neuronal entrenada para aprender una representación comprimida de una imagen (codificación), y luego reconstruirla lo más fielmente posible (decodificación).  
En este proyecto, se utiliza para mejorar imágenes médicas degradadas: toma una imagen con artefactos o problemas y trata de reconstruir su versión más clara y precisa posible, aprendiendo a eliminar dichas distorsiones durante el entrenamiento.

**¿Qué métricas se usan para evaluar la calidad de la restauración?**  
Se emplean métricas con referencia (comparan la imagen restaurada con la original) y métricas sin referencia (evalúan la imagen restaurada por sí misma).  

- **Con referencia:**  
  - **SSIM (Structural Similarity Index):** mide la similitud estructural entre dos imágenes.  
  - **PSNR (Peak Signal-to-Noise Ratio):** mide la relación entre la señal original y el ruido introducido.  
  - **MSE (Mean Squared Error):** calcula el error promedio entre los píxeles de ambas imágenes.  
  - **LPIPS (Learned Perceptual Image Patch Similarity):** evalúa la similitud perceptual entre imágenes usando redes neuronales.

- **Sin referencia:**  
  - **Entropía:** mide la cantidad de información visual o desorden.  
  - **Nitidez (Sharpness):** evalúa el nivel de detalle en la imagen.  
  - **Contraste:** mide la diferencia de intensidad entre los píxeles.  
  - **Colorido (Colorfulness):** estima la viveza y diversidad de colores presentes.

**Comparabilidad de métricas:**  
Para facilitar la comparación y visualización, todas las métricas se han transformado y escalado a un rango común del 0 al 100%.
    """)

st.markdown("""
Este sistema permite **eliminar artefactos** en las imágenes de una sonda transvaginal mediante una red neuronal.

1. Sube una imagen.
2. Para cada tipo de degradación que puede producirse en una sonda, se mostrará: la imagen original, la imagen degradada y la versión restaurada por el modelo.
3. Cada degradación incluirá una breve explicación de su causa, junto con una sección dedicada a las métricas de calidad correspondientes a esa imagen.
""")

archivo_subido = st.file_uploader("📤 Sube tu imagen aquí:", type=["png", "jpg", "jpeg"])

# nombre_seleccionado = st.selectbox("🛠️ Elige una degradación para aplicar:", nombre_transformaciones)

if archivo_subido:
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)
    modelo = ConvAutoencoder().to(dispositivo)
    
    checkpoint = torch.load("autoencoder/Buenos/checkpoint.pt", map_location=dispositivo)
    modelo.load_state_dict(checkpoint['model_state_dict'])
    modelo.eval()

    img_tensor = cargar_imagen(archivo_subido)
    original = tensor_a_pil(img_tensor)

    for i, (nombre, transformacion) in enumerate(todas):
        st.markdown(f"---\n### 🛠️ Degradación: {nombre}")

        # Aplicar la transformación
        img_degradada_tensor = aplicar_transformacion_especifica(img_tensor, nombre)
        img_degradada_pil = tensor_a_pil(img_degradada_tensor)

        with torch.no_grad():
            img_restaurada_tensor = modelo(img_degradada_tensor.to(dispositivo))
        img_restaurada_pil = tensor_a_pil(img_restaurada_tensor)

        # Mostrar imágenes en tres columnas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Original**")
            st.image(original, width=400)
        with col2:
            st.markdown("**Degradada**")
            st.image(img_degradada_pil, width=400)
        with col3:
            st.markdown("**Restaurada**")
            st.image(img_restaurada_pil, width=400)

        # Cálculo métricas con referencia
        ssim_degradada, psnr_degradada, mse_degradada, lpips_degradada = calcular_metricas(
            original, img_degradada_pil, img_tensor, img_degradada_tensor)
        ssim_restaurada, psnr_restaurada, mse_restaurada, lpips_restaurada = calcular_metricas(
            original, img_restaurada_pil, img_tensor, img_restaurada_tensor)

        # Cálculo métricas sin referencia para las 3 imágenes
        original_np = np.array(original)
        degradada_np = np.array(img_degradada_pil)
        restaurada_np = np.array(img_restaurada_pil)

        ent_orig, nitidez_orig, contr_orig, color_orig = calculate_1image_metrics(original_np)
        ent_deg, nitidez_deg, contr_deg, color_deg = calculate_1image_metrics(degradada_np)
        ent_res, nitidez_res, contr_res, color_res = calculate_1image_metrics(restaurada_np)

        # **Aquí añadimos las métricas manuales de BRISQUE, NIQE y NIMA**
        brisque_orig = calcular_brisque(original_np)
        niqe_orig = niqe(original_np)
        nima_orig = nima(original_np)

        brisque_deg = calcular_brisque(degradada_np)
        niqe_deg = niqe(degradada_np)
        nima_deg = nima(degradada_np)

        brisque_res = calcular_brisque(restaurada_np)
        niqe_res = niqe(restaurada_np)
        nima_res = nima(restaurada_np)



        with st.expander("**Causas**"):
            explicacion = explicaciones_degradaciones.get(nombre, "No se encontró una explicación específica para esta degradación.")
            st.markdown(explicacion)

        # Al mostrar las gráficas, define keys dinámicas:
        key_ref = f"grafica_con_ref_{i}"
        key_ideal_ref = f"ideal_con_ref_{i}"
        key_sin_ref = f"grafica_sin_ref_{i}"
        key_ideal_sin_ref = f"ideal_sin_ref_{i}"

        with st.expander("**Métricas de calidad**"):
            col1, col2 = st.columns(2)

            # ----- MÉTRICAS CON REFERENCIA -----
            with col1:
                st.markdown("**Métricas con referencia**")

                categories_con_ref = ["SSIM", "PSNR", "MSE", "LPIPS"]
                valores_deg = [ssim_degradada, psnr_degradada, mse_degradada, lpips_degradada]
                valores_res = [ssim_restaurada, psnr_restaurada, mse_restaurada, lpips_restaurada]

                fig_ref = go.Figure()
                fig_ref.add_trace(
                    go.Scatterpolar(r=valores_deg, theta=categories_con_ref, fill='toself', name='Degradada'))
                fig_ref.add_trace(
                    go.Scatterpolar(r=valores_res, theta=categories_con_ref, fill='toself', name='Restaurada'))
                fig_ref.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                      title="Original vs Degradada/Restaurada (Con Referencia)")

                # Columna doble: gráfica y patrón ideal
                col_ref_1, col_ref_2 = st.columns([4, 1])
                with col_ref_1:
                    st.plotly_chart(fig_ref, use_container_width=True, key=key_ref)

                    # Comparar SSIM y MSE para decidir si la restauración mejoró la calidad
                    mejor_mse = mse_restaurada > mse_degradada
                    mejor_ssim = ssim_restaurada > ssim_degradada

                    if mejor_ssim or mejor_mse:
                        st.success("La restauración ha mejorado la calidad de la imagen ✅")
                    else:
                        st.error("La restauración no ha mejorado la calidad de la imagen ❌")

                with col_ref_2:
                    st.markdown("**Ideal**")
                    fig_ideal_ref = go.Figure()
                    fig_ideal_ref.add_trace(go.Scatterpolar(
                        r=[100, 70, 0, 0],  # Todas las métricas al 100%
                        theta=categories_con_ref,
                        fill='toself',
                        name='Ideal',
                        line_color="#9cc7fa  "  # Azul clarito, segundo color de Plotly
                    ))
                    fig_ideal_ref.update_layout(
                        polar=dict(radialaxis=dict(visible=False)),
                        showlegend=False,
                        width=100,
                        height=100,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_ideal_ref, key=key_ideal_ref)

            # ----- MÉTRICAS SIN REFERENCIA -----
            with col2:
                st.markdown("**Métricas sin referencia**")

                categories_sin_ref = ["Entropía", "Nitidez", "Contraste", "Colorido", "Brisque", "Niqe", "Nima"]
                fig_sin_ref = go.Figure()
                fig_sin_ref.add_trace(go.Scatterpolar(
                    r=[ent_orig, nitidez_orig, contr_orig, color_orig, brisque_orig, niqe_orig, nima_orig],
                    theta=categories_sin_ref, fill='toself', name='Original'))
                fig_sin_ref.add_trace(go.Scatterpolar(
                    r=[ent_deg, nitidez_deg, contr_deg, color_deg, brisque_deg, niqe_deg, nima_deg],
                    theta=categories_sin_ref, fill='toself', name='Degradada'))
                fig_sin_ref.add_trace(go.Scatterpolar(
                    r=[ent_res, nitidez_res, contr_res, color_res, brisque_res, niqe_res, nima_res],
                    theta=categories_sin_ref, fill='toself', name='Restaurada'))
                fig_sin_ref.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                          title="Original vs Degradada/Restaurada (Sin Referencia)")

                # Columna doble: gráfica y patrón ideal
                col_sinref_1, col_sinref_2 = st.columns([4, 1])
                with col_sinref_1:
                    st.plotly_chart(fig_sin_ref, use_container_width=True, key=key_sin_ref)
                with col_sinref_2:
                    st.markdown("**Ideal**")
                    fig_ideal_sinref = go.Figure()
                    fig_ideal_sinref.add_trace(go.Scatterpolar(
                        r=[100, 60, 50, 50, 100, 0, 100],  # Valores ideales arbitrarios para las nuevas métricas
                        theta=categories_sin_ref,
                        fill='toself',
                        name='Ideal',
                        line_color="#ff4c4c"  # Azul clarito
                    ))
                    fig_ideal_sinref.update_layout(
                        polar=dict(radialaxis=dict(visible=False)),
                        showlegend=False,
                        width=100,
                        height=100,
                        margin=dict(l=10, r=10, t=10, b=10)
                    )
                    st.plotly_chart(fig_ideal_sinref, key=key_ideal_sin_ref)

    # Pie de página simple
    st.markdown(
        """
        <hr>
        <p style='font-size:0.8em; text-align:center; color:gray;'>
            © 2025 Proyecto de mejora de imágenes. Todos los derechos reservados.
        </p>
        """,
        unsafe_allow_html=True
    )