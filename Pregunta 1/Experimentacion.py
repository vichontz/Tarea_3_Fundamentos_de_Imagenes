import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import data
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
from Implementacion import padding, transformada, inversa, filtrar_notch, decibeles
import imageio.v3 as iio
output_dir = Path(__file__).resolve().parent / "figures_p1"



# Como debemos trabajar con una imagen de ruido periodico, pero aun no la tenemos, vamos a generar una imagen de prueba con ruido periodico.
img = data.camera()
img_float = img.astype(np.float64)


#   ------Definiciones iniciales------      #
ancho = 50
N, M = img.shape
pad = 3*N

# Ruido periodico
y, x = np.mgrid[0:N, 0:M]
frecuencias = [(35, 22), (72, -48)]
ruido = np.zeros_like(img_float)
for u0, v0 in frecuencias:
    ruido += 18 * np.cos(2 * np.pi * (u0*x/M + v0*y/N))
img_ruidosa = img_float + ruido
#   ------------------------------------    #

print("ingrese el numero del experimento que desea ver (1, 2, 3, etc)")
accion = int(input())

if accion == 1:

    #1. Muestre el espectro de magnitud de la imagen, identifique las componentes que atribuye al ruido periódico 
    #y justifique las posiciones seleccionadas para los notches. 


    # El padding cambia la cantidad de bins por ciclo en el espectro.
    escala_x = (M + 2 * pad) / M
    escala_y = (N + 2 * pad) / N

    notches = [
        (round(u0 * escala_x), round(v0 * escala_y))
        for u0, v0 in frecuencias
    ]

    
    
    # Filtrado
    img_filtrada, img_recuperada, espc_filtrado, filtro, espectro, img_ext = filtrar_notch(
        img_ruidosa,
        notches,
        ancho,
        tipo='ideal',
        pad=pad,
        modo='constant'
    )
    # Visualizacion
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagen original")
    ax[0].axis('off')

    ax[1].imshow(img_ruidosa, cmap='gray')
    ax[1].set_title("Imagen ruidosa")
    ax[1].axis('off')

    ax[2].imshow(decibeles(espectro), cmap='gray')
    ax[2].set_title("Espectro de magnitud")
    ax[2].axis('off')

    # Marcar las frecuencias conocidas
    A, L = espectro.shape
    cy, cx = A//2, L//2

    for u0, v0 in frecuencias:
        for u, v in [(u0, v0), (-u0, -v0)]:
            px = cx + round(u * escala_x)
            py = cy + round(v * escala_y)
            ax[2].add_patch(Circle(
                (px, py),
                radius=ancho,
                edgecolor='red',
                facecolor='none',
                linewidth=1
            ))
            
    plt.tight_layout()

    output_dir.mkdir(exist_ok=True)
    #fig.savefig(output_dir / "r_periodico_y_espectro", dpi=200, bbox_inches="tight")

    plt.show()


elif accion == 2:
    # 2. Para ambas familias de filtro, explore sistemáticamente varios anchos de notch.
    # Analice conjuntamente ruido residual, pérdida de detalle y artefactos.
    mode = input(' Ingrese el modo de padding \n' 'constant, edge, linear_ramp, maximum, mean, median, minimum, reflect, symmetric, wrap:\n')
    familia = input('Ingrese el tipo de notch \n''ideal, gaussiano:')
    
    # Parametros
    anchos = [5, 10, 20, 40, 100]
    pad = int(1 * N)

    notches = [
        (round(u0 * (M + 2 * pad) / M), round(v0 * (N + 2 * pad) / N))
        for u0, v0 in frecuencias
    ]

    # en primer lugar, se calculan todos los casos y se guardan
    resultados = []
    for ancho in anchos:
        img_filtrada, img_recuperada, espc_filtrado, filtro, espectro, img_ext = filtrar_notch(
            img_ruidosa,
            notches,
            ancho,
            tipo=familia,
            pad=pad,
            modo=mode
        )
        diferencia = (img_ruidosa - img_filtrada)
        resultados.append((ancho, img_filtrada, espc_filtrado, espectro, diferencia))

    # límites comunes para todo el barrido
    A = max(np.abs(r[4]).max() for r in resultados)            # diferencia: [-A, A]
    esp_ref = decibeles(resultados[0][3])                       # espectro original (igual en todos)
    esp_ref = esp_ref[np.isfinite(esp_ref)]
    vmin_esp, vmax_esp = esp_ref.min(), esp_ref.max()           # espectros: mismo rango antes/después

    # graficos
    for ancho, img_filtrada, espc_filtrado, espectro, diferencia in resultados:

        fig2, ax2 = plt.subplots(1, 2, figsize=(15, 5))

        ax2[0].imshow(decibeles(espectro), cmap='gray', vmin=vmin_esp, vmax=vmax_esp)       
        ax2[0].set_title('Espectro original')

        ax2[1].imshow(decibeles(espc_filtrado), cmap='gray', vmin=vmin_esp, vmax=vmax_esp)  
        ax2[1].set_title(f'Espectro filtrado (ideal, ancho={ancho})')                            

        for ax in ax2:
            ax.axis('off')

        plt.tight_layout()
        #fig2.savefig(output_dir / f"gaus_espc_ancho {ancho}", dpi=200, bbox_inches="tight")
        
        #plt.show()             # [ descomentar para ver ]

        fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))

        ax3[0].imshow(img_ruidosa, cmap='gray', vmin=0, vmax=255)                           
        ax3[0].set_title('Imagen con ruido')

        ax3[1].imshow(img_filtrada, cmap='gray', vmin=0, vmax=255)                          
        ax3[1].set_title('Imagen filtrada')

        im = ax3[2].imshow(diferencia, cmap='gray', vmin=-192.4, vmax=192.4)                        
        ax3[2].set_title(f'Diferencia (escala común {192.4:.1f})')                                                                     

        for ax in ax3:
            ax.axis('off')

        plt.tight_layout()
        #fig3.savefig(output_dir / f"gaus_dif_ancho{ancho}", dpi=200, bbox_inches="tight")

        #plt.show()             # [ descomentar para ver ]

    

