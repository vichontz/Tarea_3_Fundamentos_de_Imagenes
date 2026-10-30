import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
import imageio.v3 as iio
from pathlib import Path
from Implementacion_A import cargar_y_normalizar, construir_H, degradar, wiener, restaurar, rmse, curva_rmse 
from Implementacion_B import aplicar_H, aplicar_Ht, calcular_costo_tikhonov, tikhonov_gradiente


# Parametros
output_dir = Path(__file__).resolve().parent / "figures_p2"
SIGMAS = [0.001, 0.01, 0.03, 0.07]
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


    Ks = np.logspace(-3.5, 2.5, 150) 
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

        # Mejor K va a ser aquel que minimice el error para cada sigma

        idx = np.argmin(errores)
        K_opt = Ks[idx]
        rmse_opt = errores[idx]

        # Reconstrimos
        x_opt = restaurar(y, H, K_opt)

        resultados[sigma] = {
            'y': y,
            'K_opt': K_opt,
            'RMSE_opt': rmse_opt,
            'x_opt': x_opt,
            'errores': errores}

        print(f"\nSigma = {sigma}")
        print(f"RMSE degradada = {rmse(x, y):.6f}")
        print(f"K optimo = {K_opt:.6g}")
        print(f"RMSE minimo = {rmse_opt:.6f}")

        plt.plot(Ks, errores, label=f'sigma = {sigma}')

    plt.xscale('log')
    plt.xlabel("K")
    plt.ylabel("RMSE")
    plt.title("Wiener: RMSE según K")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.savefig(output_dir / "RMSEvsK", dpi=200, bbox_inches="tight")
    #plt.show()


    fig, ax = plt.subplots(len(SIGMAS), 3, figsize=(15, 15))

    for i, sigma in enumerate(SIGMAS):

        r = resultados[sigma]

        imagenes = [
            (x, "Original"),
            (r['y'], f"Degradada, sigma={sigma}"),
            (r['x_opt'], f"Wiener, K={r['K_opt']:.2e}\nRMSE={r['RMSE_opt']:.4f}")
        ]

        for j, (img, titulo) in enumerate(imagenes):
            ax[i, j].imshow(img, cmap='gray', vmin=0, vmax=1)
            ax[i, j].set_title(titulo)
            ax[i, j].axis('off')

    plt.suptitle("Comparacion de reconstrucciones")
    plt.tight_layout()
    #fig.savefig(output_dir / "comp_reconst", dpi=200, bbox_inches="tight")
    #plt.show()

    y_inicio, y_fin = 47, 110
    x_inicio, x_fin = 17, 90
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    for eje, sigma in zip(ax.flat, SIGMAS):
        r = resultados[sigma]
        recorte = r['x_opt'][y_inicio:y_fin, x_inicio:x_fin]
        eje.imshow(recorte, cmap='gray', vmin=0, vmax=1)
        eje.set_title(f"Reconstruida, sigma={sigma}")
        eje.axis('off')

    plt.suptitle(f"Recorte ampliado: filas {y_inicio}:{y_fin}, columnas {x_inicio}:{x_fin}")
    plt.tight_layout()
    #fig.savefig(output_dir / "ring_rec", dpi=200, bbox_inches="tight")
    plt.show()



elif accion == 2:

    x, h = cargar_y_normalizar()

    #  PSF perturbada = desenfoque extra (suavizado gauss de sigma_p px) es decir, PSF más ancha
    sigma_p = 1.5
    m = int(np.ceil(4 * sigma_p))
    t = np.arange(-m, m + 1)
    g = np.exp(-t**2 / (2 * sigma_p**2))
    g /= g.sum()
    h_pert = np.array([np.convolve(fila, g, mode='same') for fila in h])
    h_pert = np.array([np.convolve(columna, g, mode='same') for columna in h_pert.T]).T
    h_pert /= h_pert.sum()

    # Construimos OTF's
    H = construir_H(h, x.shape)
    H_pert = construir_H(h_pert, x.shape)


    sigma_ruido = 0.01
    y = degradar(x, H, sigma=sigma_ruido, seed=42)

    # Buscamos K (solo para asegurar, aun q podriamos tomar los valores del exp 1)
    Ks = np.logspace(-6, 0, 150)
    errores = [rmse(x, restaurar(y, H, K)) for K in Ks]
    K_opt = Ks[np.argmin(errores)]

    # 
    x_correcta = restaurar(y, H, K_opt)
    x_incorrecta = restaurar(y, H_pert, K_opt)
    x_dif = np.abs(x_incorrecta - x_correcta)                       # efecto aislado de la perturbación

    # Recorte ampliado
    crop = (slice(25, 105), slice(95, 175))

    rm_c, rm_i = rmse(x, x_correcta), rmse(x, x_incorrecta)
    rc_c, rc_i = rmse(x[crop], x_correcta[crop]), rmse(x[crop], x_incorrecta[crop])
    print(f"sigma_ruido = {sigma_ruido},  sigma_p = {sigma_p},   K utilizado: {K_opt:.3e}")
    print(f"RMSE total   : correcta {rm_c:.5f},  perturbada {rm_i:.5f},  aumento {100*(rm_i/rm_c-1):.1f} %")
    print(f"RMSE recorte : correcta {rc_c:.5f},  perturbada {rc_i:.5f},  aumento {100*(rc_i/rc_c-1):.1f} %")

    # Graficos
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    imagenes = [x_correcta, x_incorrecta, x_dif]
    titulos = [f'PSF correcta\nRMSE={rm_c:.4f}',
               f'PSF perturbada (sigma_p={sigma_p})\nRMSE={rm_i:.4f}',
               '|perturbada - correcta|']
    for k, (a, img, titulo) in enumerate(zip(ax, imagenes, titulos)):
        a.imshow(img, cmap='gray', vmin=0, vmax=(1 if k < 2 else 0.3))   # escala fija; la diferencia usa 0-0.3
        a.set_title(titulo)
        a.axis('off')
    plt.suptitle(f'Wiener con K={K_opt:.2e}', y=1.05)
    plt.tight_layout()
    #plt.savefig(output_dir / "alterada", dpi=200, bbox_inches="tight")
    plt.show()

    # 8. Comparación de un recorte ampliado
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
    for k, (a, img, titulo) in enumerate(zip(ax, imagenes, titulos)):
        a.imshow(img[crop], cmap='gray', vmin=0, vmax=(1 if k < 2 else 0.3), interpolation='nearest')
        a.set_title(titulo)
        a.axis('off')
    plt.suptitle(f'Recorte', y=1.05)
    plt.tight_layout()
    #fig.savefig(output_dir / "alterada_rec", dpi=200, bbox_inches="tight")
    plt.show()