import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# Función para la ecualización local del histograma
def local_histogram_equalization(img, window_size):
    """
    Aplica la ecualización local del histograma a una imagen utilizando una ventana de tamaño especificado.

    Args:
        img (numpy.ndarray): La imagen en escala de grises.
        window_size (int): El tamaño de la ventana para la ecualización local.

    Returns:
        numpy.ndarray: La imagen ecualizada.
    """
    # Crear una copia de la imagen para evitar modificar la original
    img = img.copy()
    # Aumentar el borde de la imagen para manejar los bordes
    pad_size = window_size // 2
    padded_img = cv2.copyMakeBorder(img, pad_size, pad_size, pad_size, pad_size, borderType=cv2.BORDER_REPLICATE)

    # Crear una imagen de salida vacía
    equalized_img = np.zeros_like(img)

    # Mover la ventana a través de la imagen
    for i in range(pad_size, padded_img.shape[0] - pad_size):
        for j in range(pad_size, padded_img.shape[1] - pad_size):
            # Extraer la región local (ventana)
            local_window = padded_img[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]

            # Aplicar la ecualización del histograma a la región local
            equalized_pixel = cv2.equalizeHist(local_window)[pad_size, pad_size]

            # Establecer el píxel ecualizado en la imagen de salida
            equalized_img[i-pad_size, j-pad_size] = equalized_pixel

    return equalized_img

# Cargar la imagen en escala de grises
image_path = r'Imagenes de Entrada\Imagen_con_detalles_escondidos.tif'
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Aplicar ecualización local del histograma con diferentes tamaños de ventana
window_size_small = 15
window_size_medium = 30
window_size_large = 45

equalized_small = local_histogram_equalization(img, window_size_small)
equalized_medium = local_histogram_equalization(img, window_size_medium)
equalized_large = local_histogram_equalization(img, window_size_large)

# Mostrar los resultados y sus respectivos histogramas
plt.figure(figsize=(18, 10))

# Imagen original y su histograma
plt.subplot(4, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.hist(img.flatten(), 256, [0, 256])
plt.title('Histograma Imagen Original')

# Ecualización local con ventana pequeña y su histograma
plt.subplot(4, 2, 3)
plt.imshow(equalized_small, cmap='gray')
plt.title(f'Ecualización Local (Ventana: {window_size_small})')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.hist(equalized_small.flatten(), 256, [0, 256])
plt.title(f'Histograma Ecualización (Ventana: {window_size_small})')

# Ecualización local con ventana mediana y su histograma
plt.subplot(4, 2, 5)
plt.imshow(equalized_medium, cmap='gray')
plt.title(f'Ecualización Local (Ventana: {window_size_medium})')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.hist(equalized_medium.flatten(), 256, [0, 256])
plt.title(f'Histograma Ecualización (Ventana: {window_size_medium})')

# Ecualización local con ventana grande y su histograma
plt.subplot(4, 2, 7)
plt.imshow(equalized_large, cmap='gray')
plt.title(f'Ecualización Local (Ventana: {window_size_large})')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.hist(equalized_large.flatten(), 256, [0, 256])
plt.title(f'Histograma Ecualización (Ventana: {window_size_large})')

plt.tight_layout()
plt.show()