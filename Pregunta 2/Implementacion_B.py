
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


# Decenso del gradiente
def tikhonov_gradiente(y, H_freq, lam, tau=None, max_iter=500, tol=1e-5, x_ideal=None):
    
    #Descenso de gradiente para: J(x) = 1/2 ||Hx-y||^2 + lam/2 ||x||^2
    #tau=None: usa un paso conservador 0.9/L. tol: tolerancia del cambio relativo entre iteraciones.

    # Constante de Lipschitz del gradiente
    L = np.max(np.abs(H_freq)**2) + lam

    # Paso 
    if tau is None:
        tau = 0.9 / L

    if tau <= 0 or tau >= 2.0 / L:
        raise ValueError(f"Se requiere 0 < tau < {2.0/L:.6g}")

    print(f"L = {L:.6f}")
    print(f"tau = {tau:.6f}")

    # Inicializacion
    x_k = np.copy(y)
    historial_costo = [calcular_costo_tikhonov(x_k, y, H_freq, lam)]
    historial_rmse = []

    if x_ideal is not None:
        historial_rmse.append(calcular_rmse(x_k, x_ideal))

    convergio = False

    for k in range(max_iter):

        # Gradiente en x^k
        residuo = aplicar_H(x_k, H_freq) - y
        gradiente = aplicar_Ht(residuo, H_freq) + lam*x_k

        # Actualizamos
        x_next = x_k - tau * gradiente

        # Registrar metricas en x^(k+1)
        costo = calcular_costo_tikhonov(x_next, y, H_freq, lam)
        historial_costo.append(costo)

        if x_ideal is not None: 
            historial_rmse.append(calcular_rmse(x_next, x_ideal))

        # Criterio de detencion
        cambio_relativo = ( np.linalg.norm(x_next - x_k) / (np.linalg.norm(x_k) + 1e-12))
        x_k = x_next

        if cambio_relativo < tol:
            convergio = True
            print(f"Convergencia en iteracion {k+1}")
            break

    if not convergio:
        print(f"Se alcanzo max_iter = {max_iter}")
    return ( x_k, np.array(historial_costo), np.array(historial_rmse), convergio )

