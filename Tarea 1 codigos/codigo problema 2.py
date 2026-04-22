import sys
import subprocess
import time

def asegurar_librerias():
    """Verifica e instala las librerías necesarias automáticamente."""
    librerias = ["numpy", "matplotlib"]
    for lib in librerias:
        try:
            __import__(lib)
        except ImportError:
            print(f"Librería '{lib}' no encontrada. Instalando...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--break-system-packages"])
asegurar_librerias()

#esta funcion es una funcion de seguridad que revisa si tenemos instaladas las librerias numpy y matpltlib
#en caso de no tener las librerias esta funcion la instala automaticamente antes de correr todo el codigo.

import numpy as np
import matplotlib.pyplot as plt

#inciso a)
#fijamos frecuencias en 50 y 120Hz, la tasa de muestreo (fs) en 1000Hz por un segundo.
#calculamos el numero total de puntos (N)
print("--- inciso a) Generacion de la señal ---")
f1, f2 = 50.0, 120.0
fs, T = 1000.0, 1.0
N = int(fs * T)
tn = np.linspace(0.0, T, N, endpoint=False)
xn = np.sin(2 * np.pi * f1 * tn) + 0.5 * np.sin(2 * np.pi * f2 * tn)

plt.figure(figsize=(10, 4))
plt.plot(tn[:100], xn[:100], 'b-', marker='o', markersize=3, label='Señal discreta $x_n$')
plt.title("inciso a) - Señal en el dominio del tiempo")
plt.xlabel("Tiempo $t_n$ [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.legend()

#inciso b) y c)
#calculo DFT: crea un arreglo vacio de numeros complejos X_k, luego usamos 2 ciclos for anidados para
#resolver la sumatoria de fourier (esto para el inciso b))
#calculamos la marnitud de los numeros complejos (np.abs), tomando la primera mitad de los datos para
#para evitar redundancia (esto para el inciso c))
print("\n--- incisos b) y c): DFT Manual y Espectro ---")
X_k = np.zeros(N, dtype=complex)
for k in range(N):
    suma = 0.0 + 0.0j
    for n in range(N):
        exponente = -1j * 2.0 * np.pi * k * n / N
        suma += xn[n] * np.exp(exponente)
    X_k[k] = suma

frecuencias = np.linspace(0, fs, N, endpoint=False)
magnitud_Xk = np.abs(X_k)
mitad = N // 2 

plt.figure(figsize=(10, 4))
plt.plot(frecuencias[:mitad], magnitud_Xk[:mitad], 'r-', label='Magnitud $|X_k|$')
plt.axvline(x=50, color='k', linestyle='--', alpha=0.5, label='f1 = 50 Hz')
plt.axvline(x=120, color='k', linestyle=':', alpha=0.5, label='f2 = 120 Hz')
plt.title("inciso c) - Espectro de Frecuencias (DFT directa)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Amplitud $|X_k|$")
plt.grid(True)
plt.legend()

#inciso d), e), f), g), h)
#para inciso d), e), h) definimos arreglos con diferentes tamaños de señal y listas vacias para guardas los tiempos
#para cada tamaño N, se genera una nueva señal.
#medicion DFT manual para inciso d): se encarga de medir el tiempo de la DFT manual, usamos for y np.sum para evitar que el PC se congele
#medicion FFT para inciso e): usa time.perf_counter() para un cronometro de alta precision para medir cuanto tarda el algoritmo np.fft.fft

print("\n--- incisos d), e), f): Tiempos de ejecución ---")
N_valores = [10**2, 10**3, 10**4, 10**5]
tiempos_dft = []
tiempos_fft = []

for N_val in N_valores:
    print(f"Procesando N = {N_val}...")
    t_val = np.linspace(0.0, 1.0, N_val, endpoint=False)
    x_val = np.sin(2 * np.pi * 50.0 * t_val) + 0.5 * np.sin(2 * np.pi * 120.0 * t_val)
    
    # FFT Numpy
    t0_fft = time.perf_counter()
    X_fft = np.fft.fft(x_val)
    tf_fft = time.perf_counter()
    tiempos_fft.append(tf_fft - t0_fft)
    
    # DFT Manual 
    t0_dft = time.perf_counter()
    X_dft = np.zeros(N_val, dtype=complex)
    constante = -1j * 2.0 * np.pi / N_val
    n_array = np.arange(N_val)
    for k in range(N_val):
        X_dft[k] = np.sum(x_val * np.exp(constante * k * n_array))
    tf_dft = time.perf_counter()
    tiempos_dft.append(tf_dft - t0_dft)

#graficos de tiempo
#ax1 f): grafica los tiempos en escala lineal, aqui el DFT manual se dispara hacia arriva, mediante una curva exponencial, mientras que la FFTse queda en 0
#ax2 g): grafica lo mismo que ax1, pero en escala de log log (usando set_xscale("log")), ahora las curvas se vuelven lineas rectas.
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(N_valores, tiempos_dft, 'o-', color='red', label='DFT Manual')
ax1.plot(N_valores, tiempos_fft, 's-', color='green', label='FFT Numpy')
ax1.set_title("inciso f) - Escala Lineal")
ax1.set_xlabel("Tamaño de la señal (N)")
ax1.set_ylabel("Tiempo [s]")
ax1.grid(True)
ax1.legend()

ax2.plot(N_valores, tiempos_dft, 'o-', color='red', label='DFT Manual')
ax2.plot(N_valores, tiempos_fft, 's-', color='green', label='FFT Numpy')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title("inciso g) - Escala Log-Log")
ax2.set_xlabel("N [Log]")
ax2.set_ylabel("Tiempo [s] [Log]")
ax2.grid(True, which="both", ls="--", alpha=0.5)
ax2.legend()

# para el inciso g) al aplicar logaritmo a los datos y usa np.polyfit para calcular la pendiente de la recta de la DFT manual.
# para el inciso h) se recorre los tiempos calculados, divide el de la DFT entre el de la FFT, y se detiene cuando encuentra el 1er tamaño 
#de señal N, donde la FFT es mas rapido

print("\n--- Resultados analiticos g) y h) ---")
log_N = np.log10(N_valores)
log_T_dft = np.log10(tiempos_dft)
exponente_dft = np.polyfit(log_N, log_T_dft, 1)[0]
print(f"inciso g) -> Exponente DFT manual: O(N^{exponente_dft:.2f})")

for i, N_val in enumerate(N_valores):
    ratio = tiempos_dft[i] / tiempos_fft[i]
    if ratio >= 100.0:
        print(f"inciso h) -> Para N = {N_val}, la FFT es {ratio:.1f} veces más rapida.")
        break

plt.tight_layout()

#usamos un solo plt.show() al final del todo, para que todo se muestre simultaneamente 
print("\nMostrando todos los graficos")
plt.show()