import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.fft import fft2, ifft2, fftshift, ifftshift



# extensión (padding) configurable de la imagen:
def padding(img, pad, mode="reflect", cval=0.0):
    img = np.asarray(img, dtype=np.float64)
    widths = ((pad, pad), (pad, pad))
    return np.pad(img, widths, mode=mode, constant_values=cval) 

# FFT 2D:
def transformada(img):
    return fftshift(fft2(img))

def inversa(img):
    return ifft2(ifftshift(img))


# construcción del filtro notch:

    # se nos pide:
    # ideal y Gaussiano. 
    # permitir definir uno o más pares simétricos de notches
    # parametrizar centros y su ancho.

def filtro_notch(forma, notches, ancho, tipo='ideal'): #(notches son tuplas que representan los centros)
    
    # Dimensiones dl espectro
    P, Q = forma

    # Centro del espectro
    cy, cx = P // 2, Q // 2

    # Utilizamos una grilla para recorrer
    y, x = np.mgrid[0:P, 0:Q]
    H = np.ones((P, Q))

    for u0, v0 in notches:
            # Distancia a los dos centros simetricos
            D1 = np.sqrt((x - (cx + u0))**2 + (y - (cy + v0))**2)
            D2 = np.sqrt((x - (cx - u0))**2 + (y - (cy - v0))**2)
    
            # Ambos tipos son los pasa altos vistos en clase, centrados en las frecuencias de los notches.
            if tipo == 'ideal':
                # tiramos cero donde la distancia es menor o igual al ancho
                H[(D1 <= ancho) | (D2 <= ancho)] = 0
    
            elif tipo == 'gaussiano':
                # Notch Gaussiano, donde la desviacion estandar es el ancho.
                H1 = 1 - np.exp(-D1**2 / (2 * ancho**2))
                H2 = 1 - np.exp(-D2**2 / (2 * ancho**2))
                H *= H1 * H2 # Multiplicamos los filtros xy y obtenemos el filtro completo
    
            else:
                raise ValueError("Tipo debe ser 'ideal' o 'gaussiano'")

    return H


# multiplicación del espectro complejo por la función de transferencia:
def aplicar_filtro(espectro, filtro):
    return espectro * filtro

# transformada inversa: 
def reconstruir(espc_filtrado):
    return np.abs(inversa(espc_filtrado).real)

# y recorte al tamaño original:
def recortar(img, forma_original, pad):
    N, M = forma_original
    return img[pad:pad+N, pad:pad+M]

#Función que encapsula todo el proceso:
def filtrar_notch(img, notches, ancho, tipo='ideal', pad=0, modo='constant'):
    # Extensión de la imagen
    img_ext = padding(img, pad, mode=modo)
    espectro = transformada(img_ext)
    filtro = filtro_notch(espectro.shape, notches, ancho, tipo)
    espc_filtrado = aplicar_filtro(espectro, filtro)
    img_recuperada = reconstruir(espc_filtrado)
    img_final = recortar(img_recuperada, img.shape, pad)
    return img_final, img_recuperada, espc_filtrado, filtro, espectro, img_ext

def decibeles(img):
    return np.log1p(np.abs(img))
#Test
img = data.camera()
notches = [(100, 0),(200, 0), (300, 0), (400, 0), (500,0)]

# Parametros
ancho = 50
N, M = img.shape
pad = 2 * N

# Filtrado
img_filtrada, img_recuperada, espc_filtrado, filtro, espectro, img_ext = filtrar_notch(
    img,
    notches,
    ancho,
    tipo='ideal',
    pad=pad,
    modo='constant'
)


fig3, axes3 = plt.subplots(1, 3, figsize=(15, 5))

axes3[0].imshow(img, cmap='gray')
axes3[0].set_title('Imagen original')

axes3[1].imshow(img_recuperada, cmap='gray')
axes3[1].set_title('Imagen recuperada')

axes3[2].imshow(img_filtrada, cmap='gray')
axes3[2].set_title('Imagen filtrada')

for ax in axes3:
    ax.axis('off')

plt.tight_layout()
plt.show()