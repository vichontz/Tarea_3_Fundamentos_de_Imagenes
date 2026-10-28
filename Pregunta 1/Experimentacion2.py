import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import data
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
from Implementacion import padding, transformada, inversa, filtrar_notch, decibeles
import imageio.v3 as iio
import matplotlib.patches as patches
output_dir = Path(__file__).resolve().parent / "figures_p1"

#-------------------------------------------------------------------------------------------
# Este archivo (experimentacion2.py) corresponde a lo realizado con la imagen propuesta,
# el archivo experimentacion.py corresponde a lo realizado con una imagen y ruido arbitrario.
#-------------------------------------------------------------------------------------------

ruta = "imagenes propuestas/P1.CR2"
img_prop = iio.imread(ruta)
if img_prop.ndim == 3:
    img_prop = img_prop[..., :3].mean(axis=2)
img_prop = img_prop.astype(np.float64)

img_float = img_prop
img = img_prop
img_ruidosa = img

#   ------Definiciones iniciales------      #
ancho = 50
N, M = img.shape
pad = 0

# Ruido periodico
y, x = np.mgrid[0:N, 0:M]
frecuencias = [(35, 22), (72, -48)]
#   ------------------------------------    #

print("ingrese el numero del experimento que desea ver (1, 2, 3, etc)")
accion = int(input())

if accion == 1:

    

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
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Imagen original")
    ax[0].axis('off')

    ax[1].imshow(decibeles(espectro), cmap='gray')
    ax[1].set_title("Espectro de magnitud")
    ax[1].axis('off')

    # Marcar las frecuencias conocidas
    A, L = espectro.shape
    cy, cx = A//2, L//2

    #for u0, v0 in frecuencias:
    #    for u, v in [(u0, v0), (-u0, -v0)]:
    #       px = cx + round(u * escala_x)
    #        py = cy + round(v * escala_y)
    #        ax[1].add_patch(Circle(
    #            (px, py),
    #            radius=ancho,
    #            edgecolor='red',
    #            facecolor='none',
    #            linewidth=1
    #        ))
            
    plt.tight_layout()

    output_dir.mkdir(exist_ok=True)
    #fig.savefig(output_dir / "r_per_y_espc_prop", dpi=200, bbox_inches="tight")

    plt.show()
   

elif accion == 2:
    # 2. Para ambas familias de filtro, explore sistemáticamente varios anchos de notch.
    # Analice conjuntamente ruido residual, pérdida de detalle y artefactos.
    mode = 'constant'
    familia = input('Ingrese el tipo de notch \n''ideal, gaussiano:')
    
    # Parametros
    anchos = [1,3,5,10,20,40]
    pad = int(0 * N)
    
    # el centro esta en (1128,752) por lo que las frecuencias (71,752) (212,752) (353,752) deben ser despazadas.
    frecuencias = [(1128-71,0), (1128-212,0), (1128-353,0)]
    notches = frecuencias

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
        diferencia = np.abs(img_ruidosa.astype(float) - img_filtrada.astype(float))
        resultados.append((ancho, img_filtrada, espc_filtrado, espectro, diferencia))

    # límites comunes para todo el barrido
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
        #fig2.savefig(output_dir / f"prop_{familia}_espc_{ancho}", dpi=200, bbox_inches="tight")
        #plt.show()             # [ descomentar para ver ]

        fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))

        ax3[0].imshow(img_ruidosa, cmap='gray', vmin=0, vmax=255)                           
        ax3[0].set_title('Imagen con ruido')

        ax3[1].imshow(img_filtrada, cmap='gray', vmin=0, vmax=255)                          
        ax3[1].set_title('Imagen filtrada')

        im = ax3[2].imshow(diferencia, cmap='gray', vmin=vmin_esp, vmax=vmax_esp)                        
        ax3[2].set_title(f'Diferencia (escala común {vmin_esp:.1f} , {vmax_esp:.1f})')                                                                     

        for ax in ax3:
            ax.axis('off')

        plt.tight_layout()
        #fig3.savefig(output_dir / f"prop_{familia}_dif_{ancho}", dpi=200, bbox_inches="tight")
        #plt.show()             # [ descomentar para ver ]

elif accion == 3:
    # 3. Compare el filtrado sin padding con la estrategia de extensión propuesta.
    # Explore al menos dos tamaños de extensión y analice los bordes mediante recortes ampliados.
    
    familia = input('Ingrese el tipo de notch (ideal, gaussiano): ')

    # --- Parámetros ---
    ancho_orig = 3  # tamaño del notcj

    # Frecuencias 
    frecuencias = [(M // 2 - 71, 0), (M // 2 - 212, 0), (M // 2 - 353, 0)]
    
    # Configuraciones a evaluar: (Modo de padding, Cantidad de píxeles, Título)
    configs = [
        
        ('constant', 0,   'Sin padding (Pad = 0)'),
        ('constant', 128, 'Constant (Pad = 128)'),

        ('reflect' , 128, 'Reflect (Pad = 128)'),
        ('reflect' , 256, 'Reflect (Pad = 256)'),
        ('reflect' , 512, 'Reflect (Pad = 512)'),
        ('reflect' , 1024,'Reflect (Pad = 1024)')
    ]

    

    z = 200  # Tamaño del recorte


    for modo, pad, titulo in configs:
        
        # Escalamos las frecuencias
        escala_x = (M + 2 * pad) / M
        escala_y = (N + 2 * pad) / N
        
        notches = [(round(u0 * escala_x), round(v0 * escala_y)) for u0, v0 in frecuencias]
        ancho = ancho_orig * escala_x
        
        # Aplicar el filtro
        img_filtrada, _, _, _, _, _ = filtrar_notch(
            img_ruidosa, notches, ancho, tipo=familia, pad=pad, modo=modo
        )
        
        # Calcular diferencia
        diferencia = np.abs(img_ruidosa.astype(float) - img_filtrada.astype(float))
        
        # Extraer los recortes de la esquina superior izquierda
        recorte_filt = img_filtrada[0:z, 0:z]
        recorte_dif  = diferencia[0:z, 0:z]


        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{titulo} | Filtro: {familia}", fontsize=16)

        # Subplot 1: Imagen completa
        ax[0].imshow(img_filtrada, cmap='gray', vmin=0, vmax=255)
        ax[0].set_title("Imagen Completa")
        ax[0].axis('off')

        # Subplot 2: Recorte ampliado (Zoom)
        ax[1].imshow(recorte_filt, cmap='gray', vmin=0, vmax=255)
        ax[1].set_title(f"Zoom Borde ({z}x{z} px)")
        ax[1].axis('off')

        # Subplot 3: Diferencia en el recorte (Artefactos/Ruido)
        ax[2].imshow(recorte_dif, cmap='gray', vmin= 0, vmax=10)
        ax[2].set_title("Diferencia en el Borde")
        ax[2].axis('off')

        plt.tight_layout()
        #fig.savefig(output_dir / f"{modo}_{pad}_{familia}.png", dpi=200, bbox_inches="tight")
        
        #plt.show()         # [ descomentar para ver ]


        