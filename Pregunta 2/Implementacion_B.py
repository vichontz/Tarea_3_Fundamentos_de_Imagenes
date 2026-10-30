
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import imageio.v3 as iio

#Reutilizamos estas
from Implementacion_A import cargar_y_normalizar, construir_H, degradar, RUTA_H, RUTA_X


# H Y H^T
def aplicar_H(x, H_freq):
    return np.real(ifft2(fft2(x) * H_freq))


def aplicar_Ht(z, H_freq):
    return np.real(ifft2(fft2(z) * np.conj(H_freq)))


def calcular_costo_tikhonov(x, y, H_freq, lam):
    residuo = aplicar_H(x, H_freq) - y
    return (0.5 * np.sum(residuo**2) + 0.5 * lam * np.sum(x**2))


def calcular_rmse(x, x_ideal):
    return float(np.sqrt(np.mean((x - x_ideal)**2)))


