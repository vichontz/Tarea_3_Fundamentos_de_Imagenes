
#@title 0.1 Librerías e imágenes de prueba

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from scipy import fft as sfft
from skimage import data, color, transform, img_as_float

plt.rcParams.update({'figure.dpi': 90, 'image.cmap': 'gray', 'axes.titlesize': 10, 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})


def preparar_imagen(im, lado=256):
    """Escala de grises, recorte cuadrado central, tamaño lado×lado y rango [0, 1]."""
    im = img_as_float(np.asarray(im))
    if im.ndim == 3:
        im = color.rgb2gray(im[..., :3])
    alto, ancho = im.shape
    s = min(alto, ancho)
    im = im[(alto - s) // 2:(alto - s) // 2 + s, (ancho - s) // 2:(ancho - s) // 2 + s]
    im = transform.resize(im, (lado, lado), anti_aliasing=True)
    return (im - im.min()) / (im.max() - im.min() + 1e-12)


IMAGENES = {
    'Camarógrafo': preparar_imagen(data.camera()),
    'Astronauta': preparar_imagen(data.astronaut()),
    'Luna': preparar_imagen(data.moon()),
    'Café': preparar_imagen(data.coffee()),
}

fig, ax = plt.subplots(1, len(IMAGENES), figsize=(3 * len(IMAGENES), 3.2))
for a, (nombre, im) in zip(ax, IMAGENES.items()):
    a.imshow(im, vmin=0, vmax=1)
    a.set_title(f'{nombre}  {im.shape[0]}×{im.shape[1]}')
    a.axis('off')
plt.tight_layout()
plt.show()


#@title 1.1 Definición de herramientas

# ── Transformadas y frecuencias ─────────────────────────────────────────────
fft2 = lambda x: sfft.fft2(x, workers=-1)
ifft2 = lambda X: sfft.ifft2(X, workers=-1)


def radio_frecuencia(shape):
    """r = √(u² + v²), con u, v enteros en el orden de la FFT (sin centrar)."""
    M, N = shape
    U, V = np.meshgrid(np.fft.fftfreq(M) * M, np.fft.fftfreq(N) * N, indexing='ij')
    return np.sqrt(U**2 + V**2)


# ── PSF y OTF ───────────────────────────────────────────────────────────────
def psf_promedio(n):
    return np.ones((n, n)) / n**2


def psf_gaussiana(sigma):
    r = int(np.ceil(4 * sigma))
    x = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, x)
    h = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return h / h.sum()


def psf_movimiento(largo, angulo):
    """Desenfoque lineal de 'largo' píxeles en la dirección 'angulo' (grados)."""
    c = largo // 2 + 1
    h = np.zeros((2 * c + 1, 2 * c + 1))
    t = np.linspace(-(largo - 1) / 2, (largo - 1) / 2, 20 * largo)
    a = np.deg2rad(angulo)
    x, y = c + t * np.cos(a), c - t * np.sin(a)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    dx, dy = x - x0, y - y0
    for xx, yy, w in [(x0, y0, (1 - dx) * (1 - dy)), (x0 + 1, y0, dx * (1 - dy)),
                      (x0, y0 + 1, (1 - dx) * dy), (x0 + 1, y0 + 1, dx * dy)]:
        np.add.at(h, (yy, xx), w)          # interpolación bilineal
    return h / h.sum()


def psf2otf(h, shape):
    """OTF H(u,v) de la PSF h para imágenes de tamaño 'shape' (convolución circular).
    Se rellena con ceros y se desplaza el centro de h al pixel (0, 0)."""
    kh, kw = h.shape
    if kh > shape[0] or kw > shape[1]:
        raise ValueError(f'La PSF ({kh}×{kw}) es más grande que la imagen {shape}.')
    hp = np.zeros(shape)
    hp[:kh, :kw] = h
    return fft2(np.roll(hp, (-(kh // 2), -(kw // 2)), axis=(0, 1)))


TIPOS_PSF = ['Promedio', 'Gaussiana', 'Movimiento']


def otf_de(shape, psf='Gaussiana', n=9, sigma_psf=2.0, largo=15, angulo=30):
    """OTF H(u,v) para el tipo de PSF y parámetros indicados."""
    h = {'Promedio': lambda: psf_promedio(n),
         'Gaussiana': lambda: psf_gaussiana(sigma_psf),
         'Movimiento': lambda: psf_movimiento(largo, angulo)}[psf]()
    return psf2otf(h, shape)


def describir_psf(psf, n, sigma_psf, largo, angulo):
    return {'Promedio': f'promedio {n}×{n}', 'Gaussiana': f'gaussiana σ={sigma_psf:.2f}',
            'Movimiento': f'movimiento {largo} px, {angulo}°'}[psf]


def psf_centrada(H, tam=41):
    """PSF espacial (centrada) a partir de la OTF, recortada a tam×tam."""
    h = np.fft.fftshift(np.real(ifft2(H)))
    cy, cx = h.shape[0] // 2, h.shape[1] // 2
    r = min(tam // 2, cy, cx)
    return h[cy - r:cy + r + 1, cx - r:cx + r + 1]


# ── Degradación y restauración ──────────────────────────────────────────────
def convolucionar(f, H):
    """h * f (convolución circular) usando la OTF H."""
    return np.real(ifft2(fft2(f) * H))


def aplicar_filtro(G, W):
    """Restauración lineal en frecuencia: f̂ = F⁻¹{ W · G }."""
    return np.real(ifft2(G * W))


def ruido_gaussiano(g, sigma, semilla=0):
    return g + sigma * np.random.default_rng(semilla).standard_normal(g.shape)


def ruido_poisson(g, pico, semilla=0):
    """Conteo de fotones: 'pico' fotones esperados donde g = 1. Se reescala a [0, 1]."""
    return np.random.default_rng(semilla).poisson(np.clip(g, 0, None) * pico) / pico


def ruido_sal_pimienta(g, densidad, semilla=0):
    m = np.random.default_rng(semilla).random(g.shape)
    out = g.copy()
    out[m < densidad / 2] = 0.0
    out[(m >= densidad / 2) & (m < densidad)] = 1.0
    return out


def degradar(f, H, sigma=0.0, semilla=0):
    """g = h * f + η, con η gaussiano de desviación estándar sigma. Retorna (g, η)."""
    gb = convolucionar(f, H)
    g = ruido_gaussiano(gb, sigma, semilla)
    return g, g - gb


def simular(imagen, sigma, **psf):
    """Original f, OTF H, degradada g, su espectro G y el ruido η (lo que usan los paneles)."""
    f = IMAGENES[imagen]
    H = otf_de(f.shape, **psf)
    g, n = degradar(f, H, sigma)
    return f, H, g, fft2(g), n


# ── Métricas ────────────────────────────────────────────────────────────────
def psnr(f, fh, rango=1.0):
    mse = np.mean((f - fh) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(rango**2 / mse)


def error_relativo(f, fh):
    return float(np.linalg.norm(f - fh) / np.linalg.norm(f))


def barrido(f, G, construir_W, valores):
    """PSNR de la restauración f̂ = F⁻¹{W(v)·G} para cada valor v del parámetro."""
    return np.array([psnr(f, aplicar_filtro(G, construir_W(v))) for v in valores])


# ── Visualización ───────────────────────────────────────────────────────────
def mostrar(ax, im, titulo='', vmin=0, vmax=1, cmap='gray', barra=False):
    """Muestra una imagen. Con vmin=None se escala entre su mínimo y su máximo."""
    if vmin is None:
        vmin, vmax = np.min(im), np.max(im)
    h = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(titulo)
    ax.axis('off')
    if barra:
        plt.colorbar(h, ax=ax, fraction=0.046, pad=0.03)
    return h


def espectro_log(X):
    """log10 |X| centrado (para visualizar espectros y filtros)."""
    return np.log10(np.abs(np.fft.fftshift(X)) + 1e-20)


def perfil_radial(A):
    """Promedio de A (en el orden de la FFT) sobre anillos de radio r = √(u²+v²)."""
    r = np.round(radio_frecuencia(A.shape)).astype(int).ravel()
    suma = np.bincount(r, weights=np.asarray(A, float).ravel())
    cuenta = np.bincount(r)
    lim = min(A.shape) // 2
    return np.arange(lim + 1), (suma / np.maximum(cuenta, 1))[:lim + 1]


def graficar_psnr(ax, valores, psnrs, actual=None, nombre='parámetro', log=True,
                  etiqueta=None, color=None, ls='-', lw=1.8):
    """Curva PSNR vs parámetro, con ★ en el óptimo y una línea en el valor actual."""
    linea, = ax.plot(valores, psnrs, ls, color=color, label=etiqueta, lw=lw)
    i = int(np.nanargmax(psnrs))
    v_opt, p_opt = valores[i], psnrs[i]
    ax.plot(v_opt, p_opt, '*', color=linea.get_color(), ms=13, mec='k', mew=0.6, zorder=5)
    if actual is not None:
        ax.axvline(actual, color='0.35', ls='--', lw=1)
    if log:
        ax.set_xscale('log')
    ax.set_xlabel(nombre)
    ax.set_ylabel('PSNR [dB]')
    ax.grid(alpha=0.3, which='both')
    return v_opt, p_opt


def ajustar_ylim(ax, rango=30):
    """Evita que valores muy negativos de PSNR aplasten el gráfico."""
    ys = []
    for linea in ax.get_lines():
        x = np.atleast_1d(np.asarray(linea.get_xdata(), float))
        if x.size >= 2 and np.all(x == x[0]):
            continue                                   # líneas verticales (axvline)
        ys.append(np.atleast_1d(np.asarray(linea.get_ydata(), float)))
    ys = np.concatenate(ys) if ys else np.array([])
    ys = ys[np.isfinite(ys)]
    if ys.size:
        ax.set_ylim(max(ys.min(), ys.max() - rango) - 1, ys.max() + 1.5)


# ── Controles (ipywidgets) ──────────────────────────────────────────────────
def _comunes(desc):
    """Argumentos comunes de los controles (cada uno con su propio Layout)."""
    return dict(description=desc, style={'description_width': '135px'},
                layout=widgets.Layout(width='98%'))


def deslizador(desc, valor, minimo, maximo, paso, fmt='.3f'):
    return widgets.FloatSlider(value=valor, min=minimo, max=maximo, step=paso, readout_format=fmt,
                               continuous_update=False, **_comunes(desc))


def deslizador_ent(desc, valor, minimo, maximo, paso=1):
    return widgets.IntSlider(value=valor, min=minimo, max=maximo, step=paso,
                             continuous_update=False, **_comunes(desc))


def deslizador_log(desc, valor, exp_min, exp_max, paso=0.05, fmt='.1e'):
    return widgets.FloatLogSlider(value=valor, base=10, min=exp_min, max=exp_max, step=paso,
                                  readout_format=fmt, continuous_update=False, **_comunes(desc))


def menu(desc, opciones, valor=None):
    """Menú desplegable; si 'valor' no está entre las opciones se elige la primera."""
    opciones = list(opciones)
    return widgets.Dropdown(options=opciones, value=valor if valor in opciones else opciones[0],
                            **_comunes(desc))


def alternar(selector, mapa, controles):
    """Muestra solo los controles que corresponden a la opción elegida en 'selector'."""
    todos = {c for lista in mapa.values() for c in lista}

    def _actualizar(*_):
        visibles = mapa.get(selector.value, [])
        for clave in todos:
            controles[clave].layout.display = '' if clave in visibles else 'none'

    selector.observe(_actualizar, names='value')
    _actualizar()


def controles_psf(psf='Gaussiana', n=9, sigma_psf=2.0, largo=15, angulo=30,
                  n_max=25, sigma_max=6.0, largo_max=41):
    """Menú de PSF y sus parámetros (solo se muestran los del tipo elegido)."""
    c = {
        'psf': menu('PSF', TIPOS_PSF, psf),
        'n': deslizador_ent('Promedio: n', n, 3, n_max, 2),
        'sigma_psf': deslizador('Gaussiana: σ', sigma_psf, 0.5, sigma_max, 0.25, '.2f'),
        'largo': deslizador_ent('Movimiento: largo', largo, 3, largo_max, 2),
        'angulo': deslizador_ent('Movimiento: ángulo', angulo, 0, 180, 5),
    }
    alternar(c['psf'], {'Promedio': ['n'], 'Gaussiana': ['sigma_psf'],
                        'Movimiento': ['largo', 'angulo']}, c)
    return c


def panel(funcion, controles, columnas=3):
    """Muestra los controles en una grilla y redibuja la figura de 'funcion' cuando cambian."""
    def dibujar(**valores):
        funcion(**valores)
        plt.tight_layout()
        plt.show()

    ui = widgets.GridBox(list(controles.values()),
                         layout=widgets.Layout(grid_template_columns=f'repeat({columnas}, minmax(0, 1fr))',
                                               grid_gap='2px 18px', width='100%'))
    display(widgets.VBox([ui, widgets.interactive_output(dibujar, controles)]))