import numpy as np
from scipy.fft import fft2, ifft2
import imageio.v3 as iio
import matplotlib.pyplot as plt
# Implementacion Parte A: Filtro de Wiener simplificado 

# Definiciones iniciales
RUTA_X = 'imagenes propuestas/cameraman.jpg'
RUTA_H = 'imagenes propuestas/psf.png'
SIGMA_RUIDO = 0.01      # desviación estándar del ruido
SEMILLA = 42


# El enunciado dice que estan normalizadas pero porsiacaso las cargamos y re-normalizamos.
def cargar_y_normalizar(ruta_img=RUTA_X, ruta_psf=RUTA_H):
   
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
    h = h / np.sum(h)  # Normalizar energía (suma = 1)

    return x, h

def construir_H(h, shape):

    kh, kw = h.shape
    h_padded = np.zeros(shape, dtype=np.float64)
    h_padded[:kh, :kw] = h
    h_centered = np.roll(h_padded, shift=(-(kh // 2), -(kw // 2)), axis=(0, 1))
    
    return fft2(h_centered)

def degradar(x, H, sigma=SIGMA_RUIDO, seed=SEMILLA): 
    rng = np.random.default_rng(seed)
    Y_limpio = np.real(ifft2(fft2(x) * H)) #convo circular
    ruido = rng.normal(0, sigma, size=x.shape)
    return Y_limpio + ruido
# Tanto cargar,construir H como degradar nos sirven para la  implementacion B


# Filtro de Wiener X̂ = H*·Y / (|H|² + K)
def wiener(Y, H, K):
    return np.conj(H) * Y / (np.abs(H) ** 2 + K)

def restaurar(y, H, K):
    return np.real(ifft2(wiener(fft2(y), H, K)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def curva_rmse(x, y, H, Ks):
    return np.array([rmse(x, restaurar(y, H, K)) for K in Ks])



#Test para ver que si todo funciona
if __name__ == "__main__":
    x, h = cargar_y_normalizar(RUTA_X, RUTA_H)

    H = construir_H(h, x.shape)
    y = degradar(x, H)

    # Explorar distintos valores de K
    Ks = np.logspace(-6, 0, 100)
    errores = curva_rmse(x, y, H, Ks)

    idx = np.argmin(errores)
    K_opt = Ks[idx]
    x_hat = restaurar(y, H, K_opt)

    print(f"K óptimo: {K_opt:.6g}")
    print(f"RMSE mínimo: {errores[idx]:.6f}")
    print(f"RMSE degradada: {rmse(x, y):.6f}")

    # Visualización
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))

    ax[0].imshow(x, cmap='gray', vmin=0, vmax=1)
    ax[0].set_title("Original")

    ax[1].imshow(y, cmap='gray', vmin=0, vmax=1)
    ax[1].set_title("Degradada")

    ax[2].imshow(x_hat, cmap='gray', vmin=0, vmax=1)
    ax[2].set_title(f"Wiener (K={K_opt:.2e})")

    for a in ax:
        a.axis('off')

    plt.tight_layout()
    plt.show()

    # Curva RMSE
    plt.figure(figsize=(7, 4))
    plt.semilogx(Ks, errores)
    plt.axvline(K_opt, color='red', linestyle='--',
                label=f'K óptimo={K_opt:.2e}')
    plt.xlabel("K")
    plt.ylabel("RMSE")
    plt.title("RMSE vs K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
