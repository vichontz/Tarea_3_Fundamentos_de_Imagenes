import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, ifftshift
import imageio.v3 as iio



# El enunciado dice que estan normalizadas pero porsiacaso las normalizamos.
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

