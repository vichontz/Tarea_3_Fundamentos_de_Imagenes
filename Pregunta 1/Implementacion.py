import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from scipy.fft import fft2, ifft2, fftshift, ifftshift

img = data.camera()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
#Implemente una cadena completa de procesamiento: 

# extensión (padding) configurable de la imagen:
def padding(img, pad, mode="reflect", cval=0.0):
    img = np.asarray(img, dtype=np.float64)
    widths = ((pad, pad), (pad, pad))
    return np.pad(img, widths, mode=mode, cval=cval)  

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
        if tipo == 'ideal':

        elif tipo == 'gaussiano':

        else:
            raise ValueError("Tipo debe ser 'ideal' o 'gaussiano'")

    return H


# multiplicación del espectro complejo por la función de transferencia:

# transformada inversa: 

# y recorte al tamaño original:
