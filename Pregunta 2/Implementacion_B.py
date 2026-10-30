
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

        # Criterio de detención
        cambio_relativo = ( np.linalg.norm(x_next - x_k) / (np.linalg.norm(x_k) + 1e-12))
        x_k = x_next

        if cambio_relativo < tol:
            convergio = True
            print(f"Convergencia en iteracion {k+1}")
            break

    if not convergio:
        print(f"Se alcanzo max_iter = {max_iter}")
    return ( x_k, np.array(historial_costo), np.array(historial_rmse), convergio )


#Testeo 

if __name__ == "__main__":

    x, h = cargar_y_normalizar(RUTA_X, RUTA_H)
    H = construir_H(h, x.shape)

    # Imagen degradada
    sigma_ruido = 0.02
    rng = np.random.default_rng(42)

    y_limpio = aplicar_H(x, H)
    ruido = rng.normal(0, sigma_ruido, size=x.shape)
    y = y_limpio + ruido

    # Parametros
    lam = 0.05
    tau = None       # Automatico: 0.9/L
    max_iter = 500
    tol = 1e-5

    # Descenso de gradiente
    x_tikh, costos, rmses, convergio = tikhonov_gradiente(
        y=y,
        H_freq=H,
        lam=lam,
        tau=tau,
        max_iter=max_iter,
        tol=tol,
        x_ideal=x
    )

    print(f"RMSE degradada: {calcular_rmse(y, x):.6f}")
    print(f"RMSE restaurada: {calcular_rmse(x_tikh, x):.6f}")
    print(f"Costo final: {costos[-1]:.6f}")

    # Imagenes
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    ax[0].imshow(x, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Original")

    ax[1].imshow(y, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title("Degradada")

    ax[2].imshow(x_tikh, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f"Tikhonov (lambda={lam})")

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    # Costo:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(costos)), costos)
    plt.xlabel("Iteracion")
    plt.ylabel("J(x)")
    plt.title("Evolucion de la funcion de costo")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # RMSE: 
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(len(rmses)), rmses)
    plt.xlabel("Iteracion")
    plt.ylabel("RMSE")
    plt.title("Evolucion del RMSE")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
