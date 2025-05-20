import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import lpips
import plotly.graph_objects as go
import numpy as np
from Funciones_app import (calcular_brisque, nima, niqe, cargar_imagen, tensor_a_pil, aplicar_transformacion_especifica,
                           calcular_metricas, calculate_1image_metrics,
                           explicaciones_degradaciones, todas)
from Funciones_app import image_enhancement_pipeline
import cv2
# Estilos CSS (solo para método Tradicional, puedes añadir una condición para que se aplique solo ahí)
estilos_css_tradicional = """
    <style>
    table.dataframe tbody tr {background-color: #f5f5f5;}
    table.dataframe tbody tr:nth-child(even) {background-color: #e0e0e0;}
    table.dataframe thead th {
        background-color: #b0b0b0; color: black; font-weight: bold;
    }
    .image-container img {
        max-width: 250px; height: auto;
    }
    </style>
"""
st.set_page_config(page_title="Restaurador de Imágenes", layout="wide")

cabecera_html = """
<style>
/* Estilos generales para el header */
header {
    margin-top:10px
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #B0E0E6; /* blanco con transparencia */
    backdrop-filter: blur(6px);
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    padding: 10px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    font-size: 1.5rem;
    font-weight: 600;
    color: #333;
    z-index: 9999;
    transition: top 0.3s ease;
    
}
header h1 {
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Icono relacionado con calidad/imágenes */
header h1 span.icon {
    font-size: 1.8rem;
}

/* Menú hamburguesa */
.menu-hamburger {
    position: relative;
    cursor: pointer;
    width: 30px;
    height: 24px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    z-index: 10001;
}

.menu-hamburger div.bar {
    height: 3px;
    background-color: #333;
    border-radius: 2px;
    transition: all 0.3s ease;
}

/* Menú desplegable */
.dropdown-menu {
    position: absolute;
    top: 36px;
    right: 0;
    background: #fff;
    border: 1px solid #ddd;
    border-radius: 8px;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    display: none;
    flex-direction: column;
    min-width: 160px;
    font-size: 0.95rem;
    overflow: hidden;
    z-index: 10000;
}

.dropdown-menu a {
    padding: 12px 16px;
    color: #333;
    text-decoration: none;
    transition: background-color 0.2s ease;
    white-space: nowrap;
}

.dropdown-menu a:hover {
    background-color: #f5f5f5;
}

/* Mostrar menú al pasar el ratón o si está activo */
.menu-hamburger:hover .dropdown-menu,
.menu-hamburger.active .dropdown-menu {
    display: flex;
}
</style>

<header id="page-header">
    <h1>
        <img src="https://cdn-icons-png.flaticon.com/512/8601/8601448.png" alt="Restaurar Icono" style="height: 1.2em;">
        Restaurador de Imágenes
    </h1>
    <div class="menu-hamburger" id="menuToggle" aria-label="Menú" tabindex="0">
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="bar"></div>
        <div class="dropdown-menu" role="menu" aria-hidden="true">
            <a href="#funcionamiento" role="menuitem">Funcionamiento</a>
            <a href="#saber-mas" role="menuitem">Resultados</a>
            <a href="#funcionamiento" role="menuitem">Compartir</a>
        </div>
    </div>
</header>

<script>
// Manejo de clic para abrir/cerrar menú
const menu = document.getElementById('menuToggle');

menu.addEventListener('click', function (e) {
    e.stopPropagation();
    this.classList.toggle('active');
});

// Cerrar si se hace clic fuera del menú
document.addEventListener('click', function (e) {
    if (!menu.contains(e.target)) {
        menu.classList.remove('active');
    }
});
</script>
<script>
// Detectar scroll para ocultar o mostrar header
let lastScrollTop = 0;
const header = document.getElementById('page-header');

window.addEventListener('scroll', function(){
    let scrollTop = window.pageYOffset || document.documentElement.scrollTop;
    if(scrollTop > lastScrollTop && scrollTop > 50){
        // Scroll hacia abajo - ocultar header
        header.style.top = '-70px';
    } else {
        // Scroll hacia arriba - mostrar header
        header.style.top = '0';
    }
    lastScrollTop = scrollTop <= 0 ? 0 : scrollTop; // Evita scroll negativo
});
</script>
"""
st.markdown(cabecera_html, unsafe_allow_html=True)
""""""
# Ahora sí el selector de método justo debajo
metodo = st.radio("Elige el método de restauración:", ("IA", "Tradicional"))

if metodo == "Tradicional":

    st.markdown(estilos_css_tradicional, unsafe_allow_html=True)

    with st.expander("**Conocimientos básicos**"):
        st.markdown(""" **¿Cómo funciona el método tradicional?**  
   Aplica una mejora de calidad utilizando una serie de filtros clásicos manualmente seleccionados. El proceso de mejora de imagen comienza con la aplicación de un filtro de mediana, que elimina el ruido preservando los bordes. A continuación, se realiza una corrección de contraste utilizando la técnica CLAHE, la cual mejora localmente el contraste en el canal de luminancia del espacio de color LAB. Seguidamente, se reduce la saturación de la imagen para atenuar colores excesivos y lograr una apariencia más natural. Después, se incrementa la nitidez 
   mediante un filtro de convolución que realza los bordes. Finalmente, se aplica una normalización para ajustar los valores de intensidad al rango visible de 0 a 255.

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
    Este sistema permite **eliminar artefactos** en las imágenes de una sonda transvaginal mediante filtros.

    1. Sube una imagen.
    2. Para cada tipo de degradación que puede producirse en una sonda, se mostrará: la imagen original, la imagen degradada y la versión restaurada por el modelo.
    3. Cada degradación incluirá una breve explicación de su causa, junto con una sección dedicada a las métricas de calidad correspondientes a esa imagen.
    """)

    archivo_subido = st.file_uploader("📤 Sube tu imagen aquí:",
                                      type=["png", "jpg", "jpeg"])

    if archivo_subido:

        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)
        import cv2

        # Convertir archivo subido a tensor
        img_tensor = cargar_imagen(archivo_subido)
        original = tensor_a_pil(img_tensor)

        for i, (nombre, transformacion) in enumerate(todas):
            st.markdown(f"---\n### 🛠️ Degradación: {nombre}")

            # Aplicar transformación
            img_degradada_tensor = aplicar_transformacion_especifica(img_tensor, nombre)
            img_degradada_pil = tensor_a_pil(img_degradada_tensor)

            # Convertir a NumPy (H, W, C) para OpenCV
            img_degradada_np = img_degradada_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            img_degradada_np = (img_degradada_np * 255).astype(np.uint8)

            # Restaurar usando pipeline clásico
            normalized = image_enhancement_pipeline(img_degradada_np)

            # Convertir restaurada a tensor
            img_restaurada_tensor = torch.tensor(normalized / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(
                0).to(dispositivo)
            img_restaurada_pil = Image.fromarray(normalized)

            # Mostrar en columnas
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

            # Cálculo de métricas
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
                explicacion = explicaciones_degradaciones.get(nombre,
                                                              "No se encontró una explicación específica para esta degradación.")
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
                        mejor_mse = mse_restaurada < mse_degradada
                        mejor_ssim = ssim_restaurada > ssim_degradada

                        if mejor_mse:
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
                            r=[100, 70, 50, 50, 100, 0, 100],  # Valores ideales arbitrarios para las nuevas métricas
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

elif metodo == "IA":
    st.markdown(estilos_css_tradicional, unsafe_allow_html=True)


    # ---------------------------
    # MODELO AUTOENCODER
    # ---------------------------
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

    archivo_subido = st.file_uploader("📤 Sube tu imagen aquí:",
                                      type=["png", "jpg", "jpeg"])

    if archivo_subido:

        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lpips_fn = lpips.LPIPS(net='alex').to(dispositivo)
        modelo = ConvAutoencoder().to(dispositivo)

        checkpoint = torch.load("checkpoint.pt", map_location=dispositivo)
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
                explicacion = explicaciones_degradaciones.get(nombre,
                                                              "No se encontró una explicación específica para esta degradación.")
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
                            r=[100, 70, 50, 50, 100, 0, 100],  # Valores ideales arbitrarios para las nuevas métricas
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

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: gray;
        text-align: center;
        font-size: 0.75em;
        padding: 5px 0;
        border-top: 1px solid #e7e7e7;
        font-family: Arial, sans-serif;
        transition: transform 0.3s ease;
        z-index: 1000;
    }
    </style>

    <div class="footer" id="footer">
        <div>📍 Calle Ategorrieta 123, Salamanca</div>
        <div>📞 Contacto: +34 600 123 456 | mluis234@gmail.com</div>
        <div>© 2025 Proyecto de mejora de imágenes. Todos los derechos reservados.</div>
    </div>

    <script>
    let lastScrollTop = 0;
    const footer = document.getElementById('footer');

    window.addEventListener('scroll', function() {
        let st = window.pageYOffset || document.documentElement.scrollTop;
        if (st > lastScrollTop){
            // Scroll hacia abajo - ocultar footer
            footer.style.transform = 'translateY(100%)';
        } else {
            // Scroll hacia arriba - mostrar footer
            footer.style.transform = 'translateY(100%)';
        }
        lastScrollTop = st <= 0 ? 0 : st; // Para evitar valores negativos
    });
    </script>
    """,
    unsafe_allow_html=True
)
