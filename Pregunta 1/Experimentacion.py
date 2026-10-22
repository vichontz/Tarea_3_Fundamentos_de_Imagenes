import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from skimage import data
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from pathlib import Path
from Implementacion import padding, transformada, inversa, filtrar_notch, decibeles

output_dir = Path(__file__).resolve().parent / "figures_p1"

#1. Muestre el espectro de magnitud de la imagen, identifique las componentes que atribuye al ruido periódico 
#y justifique las posiciones seleccionadas para los notches. 

# Como debemos trabajar con una imagen de ruido periodico, pero aun no la tenemos, vamos a generar una imagen de prueba con ruido periodico.
img = data.camera()
img_float = img.astype(np.float64)

# Parametros
ancho = 50
N, M = img.shape
pad = N



# Ruido periodico
y, x = np.mgrid[0:N, 0:M]
frecuencias = [(35, 22), (72, -48)]
ruido = np.zeros_like(img_float)

for u0, v0 in frecuencias:
    ruido += 18 * np.cos(2 * np.pi * (u0*x/M + v0*y/N))

img_ruidosa = img_float + ruido

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
fig.savefig(output_dir / "r_periodico_y_espectro", dpi=200, bbox_inches="tight")

plt.show()

# 2. Para ambas familias de filtro, explore sistemáticamente varios anchos de notch.
# Analice conjuntamente ruido residual, pérdida de detalle y artefactos.

diferencia = img_ruidosa - img_filtrada

fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))

ax2[0].imshow(decibeles(espectro), cmap='gray')
ax2[0].set_title('Espectro original')

ax2[1].imshow(decibeles(espc_filtrado), cmap='gray')
ax2[1].set_title('Espectro filtrado')

ax2[2].imshow(img_filtrada, cmap='gray')
ax2[2].set_title('Imagen Filtrada')

for ax in ax2:
    ax.axis('off')

plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5))


ax3[0].imshow(img_ruidosa, cmap='gray')
ax3[0].set_title('Imagen con ruido')

ax3[1].imshow(img_filtrada, cmap='gray')
ax3[1].set_title('Imagen filtrada')

ax3[1].imshow(diferencia, cmap='gray')
ax3[1].set_title('diferencia')

for ax in ax3:
    ax.axis('off')

plt.tight_layout()
plt.show()
