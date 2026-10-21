import numpy as np
import matplotlib.pyplot as plt
from skimage import data

img = data.camera()
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
#Implemente una cadena completa de procesamiento: 

# extensión (padding) configurable de la imagen:
def padding(img, pad, mode="reflect", cval=0.0):
    img = np.asarray(img, dtype=np.float64)
    widths = ((pad, pad), (pad, pad))
    return np.pad(img, widths, mode=mode)  

# FFT 2D:

# construcción del filtro notch:

# multiplicación del espectro complejo por la función de transferencia:

# transformada inversa: 

# y recorte al tamaño original:

