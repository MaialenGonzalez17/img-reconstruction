import streamlit as st
from PIL import Image
import torch, lpips, cv2, time, numpy as np
import plotly.graph_objects as go
import torchvision.transforms as transforms
from Funciones_app import (cargar_imagen, tensor_a_pil, aplicar_transformacion_especifica,
                           calcular_metricas_normalizadas, calculate_1image_metrics,
                           explicaciones_degradaciones, todas, image_enhancement_pipeline, dispositivo, ConvAutoencoder)
from design import cabecera_html, estilos_css_tradicional, mostrar_conocimientos_basicos, mostrar_footer, mostrar_conocimientos_autoencoder

st.set_page_config(page_title="Restaurador de Imágenes", layout="wide")
st.markdown(cabecera_html, unsafe_allow_html=True)

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelo_ae = ConvAutoencoder()
modelo_ae.to(dispositivo)

checkpoint_path = "checkpoint.pt"
checkpoint = torch.load(checkpoint_path, map_location=dispositivo)

modelo_ae.load_state_dict(checkpoint['model_state_dict'])
modelo_ae.eval()


metodo = st.radio("Elige el método de restauración:", ("IA", "Tradicional"))

pil_to_tensor = transforms.ToTensor()
def mostrar_metricas(original, degradada_pil, degradada_tensor, restaurada_pil, restaurada_tensor, i):
    original_tensor = pil_to_tensor(original).unsqueeze(0).to(dispositivo)

    degradada_tensor = degradada_tensor.unsqueeze(0).to(dispositivo) if degradada_tensor.dim() == 3 else degradada_tensor.to(dispositivo)
    restaurada_tensor = restaurada_tensor.unsqueeze(0).to(dispositivo) if restaurada_tensor.dim() == 3 else restaurada_tensor.to(dispositivo)

    ssim_deg, psnr_deg, mse_deg, lpips_deg = calcular_metricas_normalizadas(original, degradada_pil, original_tensor, degradada_tensor)
    ssim_res, psnr_res, mse_res, lpips_res = calcular_metricas_normalizadas(original, restaurada_pil, original_tensor, restaurada_tensor)

    metrics_orig = calculate_1image_metrics(np.array(original))
    metrics_deg = calculate_1image_metrics(np.array(degradada_pil))
    metrics_res = calculate_1image_metrics(np.array(restaurada_pil))

    with st.expander("**Causas**"):
        st.markdown(explicaciones_degradaciones.get(todas[i][0], "No se encontró explicación."))

    with st.expander("**Métricas de calidad**"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Métricas con referencia**")
            categorias_ref = ["SSIM", "PSNR", "MSE", "LPIPS"]
            fig_ref = go.Figure()
            fig_ref.add_trace(
                go.Scatterpolar(r=[ssim_deg, psnr_deg, mse_deg, lpips_deg], theta=categorias_ref, fill='toself',
                                name='Degradada'))
            fig_ref.add_trace(
                go.Scatterpolar(r=[ssim_res, psnr_res, mse_res, lpips_res], theta=categorias_ref, fill='toself',
                                name='Restaurada'))
            fig_ref.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                  title="Original vs Degradada/Restaurada (Con Referencia)")
            st.plotly_chart(fig_ref, use_container_width=True, key=f"ref_chart_{i}")

            if mse_res > mse_deg or ssim_res > ssim_deg:
                st.success("La restauración ha mejorado la calidad ✅")
            else:
                st.error("La restauración no ha mejorado la calidad ❌")
        with col2:
            st.markdown("**Métricas sin referencia**")
            categorias_sin_ref = ["Entropía", "Nitidez", "Contraste", "Colorido", "Brisque", "Niqe", "Nima"]
            fig_sin_ref = go.Figure()
            fig_sin_ref.add_trace(go.Scatterpolar(
                r=[metrics_orig["Entropía"], metrics_orig["Nitidez"], metrics_orig["Contraste"],
                   metrics_orig["Colorido"], metrics_orig["Brisque"], metrics_orig["Niqe"], metrics_orig["Nima"]],
                theta=categorias_sin_ref, fill='toself', name='Original'))
            fig_sin_ref.add_trace(go.Scatterpolar(
                r=[metrics_deg["Entropía"], metrics_deg["Nitidez"], metrics_deg["Contraste"],
                   metrics_deg["Colorido"], metrics_deg["Brisque"], metrics_deg["Niqe"], metrics_deg["Nima"]],
                theta=categorias_sin_ref, fill='toself', name='Degradada'))
            fig_sin_ref.add_trace(go.Scatterpolar(
                r=[metrics_res["Entropía"], metrics_res["Nitidez"], metrics_res["Contraste"],
                   metrics_res["Colorido"], metrics_res["Brisque"], metrics_res["Niqe"], metrics_res["Nima"]],
                theta=categorias_sin_ref, fill='toself', name='Restaurada'))
            fig_sin_ref.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True,
                                      title="Original vs Degradada/Restaurada (Sin Referencia)")
            st.plotly_chart(fig_sin_ref, use_container_width=True, key=f"sinref_chart_{i}")


def procesamiento_imagen_tradicional(imagen):
    img_tensor = cargar_imagen(imagen)
    original = tensor_a_pil(img_tensor)
    for i, (nombre, _) in enumerate(todas):
        st.markdown(f"---\n### 🛠️ Degradación: {nombre}")
        img_degradada_tensor = aplicar_transformacion_especifica(img_tensor, nombre)
        img_degradada_pil = tensor_a_pil(img_degradada_tensor)

        img_degradada_np = (img_degradada_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        restaurada_np = image_enhancement_pipeline(img_degradada_np)
        img_restaurada_pil = Image.fromarray(restaurada_np)
        img_restaurada_tensor = torch.tensor(restaurada_np / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(
            0).to(dispositivo)

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Original**");
        c1.image(original, width=400)
        c2.markdown("**Degradada**");
        c2.image(img_degradada_pil, width=400)
        c3.markdown("**Restaurada**");
        c3.image(img_restaurada_pil, width=400)

        mostrar_metricas(original, img_degradada_pil, img_degradada_tensor, img_restaurada_pil, img_restaurada_tensor,
                         i)


def procesamiento_imagen_ia(imagen, modelo_ae):
    img_tensor = cargar_imagen(imagen)  # Suponemos tensor con shape (C, H, W)
    original = tensor_a_pil(img_tensor)

    for i, (nombre, _) in enumerate(todas):
        st.markdown(f"---\n### 🛠️ Degradación: {nombre}")

        # Aplica la transformación degradada, resultado tensor (C, H, W)
        img_degradada_tensor = aplicar_transformacion_especifica(img_tensor, nombre)
        img_degradada_pil = tensor_a_pil(img_degradada_tensor)

        # Mover tensor a dispositivo (GPU/CPU)
        img_degradada_tensor = img_degradada_tensor.to(dispositivo)

        # Debug forma antes de pasar al modelo
        print(f"Forma antes del modelo para '{nombre}': {img_degradada_tensor.shape}")

        # Ajustar dimensiones para que sea (B, C, H, W)
        if img_degradada_tensor.dim() == 3:
            entrada = img_degradada_tensor.unsqueeze(0)  # Añade batch
        elif img_degradada_tensor.dim() == 5:
            # Si tienes dimensiones extras con tamaño 1, las eliminas
            entrada = img_degradada_tensor.squeeze(1).squeeze(0).unsqueeze(0)
        else:
            entrada = img_degradada_tensor  # Ya está en la forma correcta

        # Inferencia sin gradiente
        with torch.no_grad():
            restaurada_tensor = modelo_ae(entrada)

        # Elimina la dimensión batch para seguir trabajando
        restaurada_tensor = restaurada_tensor.squeeze(0).cpu()
        restaurada_pil = tensor_a_pil(restaurada_tensor)

        c1, c2, c3 = st.columns(3)
        c1.markdown("**Original**")
        c1.image(original, width=400)
        c2.markdown("**Degradada**")
        c2.image(img_degradada_pil, width=400)
        c3.markdown("**Restaurada (IA)**")
        c3.image(restaurada_pil, width=400)

        mostrar_metricas(original, img_degradada_pil, img_degradada_tensor.cpu(), restaurada_pil, restaurada_tensor, i)


def procesamiento_video_tradicional(video_path):
    cap = cv2.VideoCapture(video_path)
    st.markdown(f"### 🎞️ Total de frames detectados: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    placeholders = []
    for nombre, _ in todas:
        st.markdown(f"---\n### 🛠️ Degradación: {nombre}")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Original**");
        ph_orig = c1.empty()
        c2.markdown("**Degradada**");
        ph_deg = c2.empty()
        c3.markdown("**Restaurada**");
        ph_res = c3.empty()
        placeholders.append((ph_orig, ph_deg, ph_res))

    max_frames = 100
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        img_tensor = cargar_imagen(image_pil)
        original_pil = tensor_a_pil(img_tensor)

        for i, (nombre, _) in enumerate(todas):
            ph_orig, ph_deg, ph_res = placeholders[i]
            img_deg_tensor = aplicar_transformacion_especifica(img_tensor, nombre)
            img_deg_pil = tensor_a_pil(img_deg_tensor)
            img_deg_np = (img_deg_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_res_np = image_enhancement_pipeline(img_deg_np)
            img_res_pil = Image.fromarray(img_res_np)
            ph_orig.image(original_pil, use_container_width=True)
            ph_deg.image(img_deg_pil, use_container_width=True)
            ph_res.image(img_res_pil, use_container_width=True)

        time.sleep(0.07)
        frame_count += 1
    cap.release()


def procesamiento_video_ia(video_path, modelo_ae):
    cap = cv2.VideoCapture(video_path)
    st.markdown(f"### 🎞️ Total de frames detectados: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
    placeholders = []
    for nombre, _ in todas:
        st.markdown(f"---\n### 🛠️ Degradación: {nombre}")
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Original**");
        ph_orig = c1.empty()
        c2.markdown("**Degradada**");
        ph_deg = c2.empty()
        c3.markdown("**Restaurada (IA)**");
        ph_res = c3.empty()
        placeholders.append((ph_orig, ph_deg, ph_res))

    def restore_with_model(img_tensor, modelo):
        modelo.eval()
        with torch.no_grad():
            # Solo añadir batch si no existe
            if img_tensor.dim() == 3:  # (C, H, W)
                input_batch = img_tensor.unsqueeze(0).to(dispositivo)  # (1, C, H, W)
            elif img_tensor.dim() == 4:  # ya tiene batch
                input_batch = img_tensor.to(dispositivo)
            else:
                raise ValueError(f"Dimensiones inesperadas: {img_tensor.shape}")

            output_batch = modelo(input_batch)
            output = output_batch.squeeze(0).cpu().numpy()
            output = np.transpose(output, (1, 2, 0))  # (C,H,W) -> (H,W,C)
            output = (output * 255).clip(0, 255).astype(np.uint8)
        return output

    max_frames = 100
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)
        img_tensor = cargar_imagen(image_pil)  # Tensor CxHxW, valores normalizados [0,1]

        original_pil = tensor_a_pil(img_tensor)

        for i, (nombre, _) in enumerate(todas):
            ph_orig, ph_deg, ph_res = placeholders[i]

            img_deg_tensor = aplicar_transformacion_especifica(img_tensor, nombre).to(dispositivo)

            # 1. Inferencia con el modelo (tensor a NumPy o PIL)
            img_restaurada_np = restore_with_model(img_deg_tensor, modelo_ae)  # función que te di antes

            # 2. Pipeline clásica sobre imagen restaurada (NumPy)
            img_mejorada_np = image_enhancement_pipeline(img_restaurada_np)

            # 3. Convertir img_deg_tensor a PIL para mostrar
            img_deg_pil = tensor_a_pil(img_deg_tensor.cpu())

            # 4. Convertir la imagen mejorada (NumPy) a PIL para mostrar
            restaurada_pil = Image.fromarray(img_mejorada_np)

            # Mostrar en Streamlit
            ph_orig.image(original_pil, use_container_width=True)
            ph_deg.image(img_deg_pil, use_container_width=True)
            ph_res.image(restaurada_pil, use_container_width=True)

        time.sleep(0.07)
        frame_count += 1

    cap.release()


if metodo == "Tradicional":
    st.markdown(estilos_css_tradicional, unsafe_allow_html=True)
    mostrar_conocimientos_basicos()
    opcion = st.radio("📷 Elige una opción para la imagen o vídeo:",
                      ("Subir imagen", "Capturar desde cámara (imagen)", "Subir video"))
    if opcion in ("Subir imagen", "Capturar desde cámara (imagen)"):
        entrada = st.file_uploader("📤 Sube tu imagen aquí:",
                                   type=["png", "jpg", "jpeg"]) if opcion == "Subir imagen" else st.camera_input(
            "📸 Toma una foto")
        if entrada:
            imagen = Image.open(entrada).convert("RGB")
            procesamiento_imagen_tradicional(imagen)
    else:
        video_file = st.file_uploader("Sube un video grabado desde tu navegador", type=["mp4", "webm"])
        if video_file:
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            procesamiento_video_tradicional("temp_video.mp4")

elif metodo == "IA":
    st.markdown(estilos_css_tradicional, unsafe_allow_html=True)
    mostrar_conocimientos_autoencoder()
    opcion = st.radio("📷 Elige una opción para la imagen o vídeo:",
                      ("Subir imagen", "Capturar desde cámara (imagen)", "Subir video"))

    # No vuelvas a crear el modelo aquí, usa la instancia creada previamente.

    if opcion in ("Subir imagen", "Capturar desde cámara (imagen)"):
        entrada = st.file_uploader("📤 Sube tu imagen aquí:",
                                   type=["png", "jpg", "jpeg"]) if opcion == "Subir imagen" else st.camera_input(
            "📸 Toma una foto")
        if entrada:
            imagen = Image.open(entrada).convert("RGB")
            procesamiento_imagen_ia(imagen, modelo_ae)  # Pasa la instancia, no llamas a modelo_ae()
    else:
        video_file = st.file_uploader("Sube un video grabado desde tu navegador", type=["mp4", "webm"])
        if video_file:
            with open("temp_video.mp4", "wb") as f:
                f.write(video_file.read())
            procesamiento_video_ia("temp_video.mp4", modelo_ae)

mostrar_footer()