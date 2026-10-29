import numpy as np
from funciones import fft2, ifft2
import numpy as np
import imageio.v3 as iio
# Implementacion Parte A: Filtro de Wiener simplificado 

# Definiciones iniciales
RUTA_X = 'imagenes propuestas/cameraman.jpg'
RUTA_H = 'imagenes propuestas/psf.png'
SIGMA_RUIDO = 0.01      # desviación estándar del ruido
SEMILLA = 0


# El enunciado dice que estan normalizadas pero porsiacaso las re-normalizamos.
def cargar_y_normalizar(ruta_img, ruta_psf):
   
    # Cargar imagen
    x = iio.imread(ruta_img)
    if x.ndim == 3:
        x = x[..., :3].mean(axis=2)
    x = x.astype(np.float64)
    x = (x - x.min()) / (x.max() - x.min())  # Normalizar en [0, 1]

    # Cargar PSF
    h = iio.imread(ruta_psf)
    if h.ndim == 3:
        h = h[..., :3].mean(axis=2)
    h = h.astype(np.float64)
    h = h / np.sum(h)  # Normalizamos Energiá
    return x, h


# OTF
def construir_H(h, shape):
    kh, kw = h.shape
    hp = np.zeros(shape)
    hp[:kh, :kw] = h # Aumentamos la imagen con cero-padding
    hp = np.roll(hp, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return fft2(hp)

# Filtro de Wiener X̂ = H*·Y / (|H|² + K)
def wiener(Y, H, K):
    return np.conj(H) * Y / (np.abs(H) ** 2 + K)

def restaurar_wiener(y, H, K):
    return np.real(ifft2(wiener(fft2(y), H, K)))
