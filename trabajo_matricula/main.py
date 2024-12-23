# PAQUETES NECESARIOS
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Funcion que recibe una imagen y la convierte a formato RGB
"""Leer la imagen de entrada y pasarla a RGB->(Formato de imagen que usa OpenCv).
Al usar matplotlib, tendremos que usar una variación de RGB, que es BGR."""
def convertir_a_rgb(ruta_imagen):
    img = cv.imread(ruta_imagen)  # ESTA propiedad lee la imagen desde una ruta de archivo
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # Conversión a RGB, donde la propiedad indica que el archivo BGR proviene de RGB
    return img

# CONVERTIR A ESPACIO DE COLOR HSV:
def convertir_a_hsv(img):  # Convierte de RGB a HSV
    HSVimage = cv.cvtColor(img, cv.COLOR_RGB2HSV)  # Conversión de RGB a HSV
    Vimage = cv.split(HSVimage)[2]  # Extrae el canal V (valor -> brillo de la imagen)
    return HSVimage, Vimage
    """HSV->H(tono), S(saturación), V(brillo).
    Canal V: Extraer el canal de brillo y así identificar mejor las zonas claras y las zonas oscuras."""
    """
    HSV = [0, 1, 2]
    """

# UMBRALIZACIÓN
def umbralizar(Vimage):  # Recibe el canal V de la imagen en HSV
    """Pasar del color -> imagen binaria (2 valores: negro y blanco -> generamos el color con escalas de grises)"""
    th, imThr = cv.threshold(Vimage, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    """
    CV.THRESH_BINARY_INV -> Genera un binario invertido, donde las zonas claras se convierten a oscuras y las oscuras a claras.
    OTSU -> Calcula automáticamente el mejor valor de umbral para separar los píxeles claros y oscuros.
    imThr -> Imagen resultante en blanco y negro.
    """
    return imThr

# ETIQUETADO DE COMPONENTES CONECTADOS
def etiquetar_componentes(imThr):  # Recibe la imagen binaria
    """Identificar y etiquetar áreas conectadas (regiones blancas) en la imagen binaria."""
    _, imLabels = cv.connectedComponents(imThr)  # Etiqueta regiones conectadas en la imagen
    imLabels = np.uint8(imLabels)  # Convertir etiquetas al formato uint8 para visualización
    """
    COMPONENTES CONECTADOS -> Identifica y etiqueta áreas conectadas (regiones blancas), ya que es una imagen binaria.
    imLabels -> Imagen donde cada región conectada tiene un número único.
    uint8 -> Tamaño que adquiere la imagen (tamaño mínimo).
    """
    return imLabels

# Ejemplo de uso:
ruta = 'img/7153JWD.JPG'  # Ruta de la imagen
imagen_rgb = convertir_a_rgb(ruta)  # Convertir la imagen a RGB
imagen_hsv, canal_v = convertir_a_hsv(imagen_rgb)  # Convertir la imagen RGB a HSV y obtener el canal V
imagen_umbralizada = umbralizar(canal_v)  # Umbralizar el canal V de la imagen HSV
imagen_etiquetada = etiquetar_componentes(imagen_umbralizada)  # Etiquetar componentes conectados en la imagen binaria

# Mostrar las imágenes con matplotlib
plt.figure(figsize=(20, 10))

# Mostrar la imagen HSV completa
plt.subplot(1, 3, 1)
plt.imshow(imagen_hsv)
plt.title("Imagen en espacio HSV")

# Mostrar la imagen umbralizada en escala de grises
plt.subplot(1, 3, 2)
plt.imshow(imagen_umbralizada, cmap='gray')
plt.title("Imagen Umbralizada")

# Mostrar la imagen etiquetada
plt.subplot(1, 3, 3)
plt.imshow(imagen_etiquetada, cmap='nipy_spectral')  # Usar un colormap para destacar las etiquetas
plt.title("Componentes Conectados Etiquetados")

plt.show()


