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




print("ingrese el numero del experimento que desea ver (1, 2, 3, etc)")
accion = int(input())

if accion == 1:

#explore K en una escala suficientemente amplia, preferentemente logarítmica. 
#Para al menos dos niveles de ruido.
#grafique RMSE versus K.
#determine el valor óptimo 
#compare las reconstrucciones 
#correspondientes.


    Ks = np.logspace(-6, 0, 150) 
    resultados = {}
    plt.figure(figsize=(8, 5))


    for sigma in SIGMAS:

        # Generar observacion
        y = degradar(x, H, sigma=sigma, seed=SEMILLA)

        errores = []
        for K in Ks:
            x_hat = restaurar(y, H, K)
            errores.append(rmse(x, x_hat))
        errores = np.array(errores)

