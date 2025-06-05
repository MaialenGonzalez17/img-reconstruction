import streamlit as st
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

def mostrar_conocimientos_basicos():
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

def mostrar_conocimientos_autoencoder():
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


def mostrar_footer():
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
            z-index: 1000;
        }
        </style>

        <div class="footer">
            <div>📍 Calle Ategorrieta 123, Salamanca</div>
            <div>📞 Contacto: +34 600 123 456 | mluis234@gmail.com</div>
            <div>© 2025 Proyecto de mejora de imágenes. Todos los derechos reservados.</div>
        </div>
        """,
        unsafe_allow_html=True
    )