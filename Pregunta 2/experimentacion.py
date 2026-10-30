import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import imageio.v3 as iio

from Implementacion_A import cargar_y_normalizar, construir_H, degradar, wiener, restaurar, rmse, curva_rmse 
from Implementacion_B import aplicar_H, aplicar_Ht, calcular_costo_tikhonov, tikhonov_gradiente


# Parametros
SIGMAS = [0.01, 0.03]
SEMILLA = 42
# Cargar imagen y PSF
x, h = cargar_y_normalizar()
H = construir_H(h, x.shape)
