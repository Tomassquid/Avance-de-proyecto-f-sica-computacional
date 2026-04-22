import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

#parametros
N = 125                     #numero de particulas
m = 3.32e-27                #masa de h2
kB = 1.380649e-23           #constante de Boltzmann (J/K)
dt = 1e-12                  #paso de tiempo: 1 ps
# Valores iniciales
L_actual = 10e-6            # 10 micrometros
T_actual = 300.0            # 300K

#funciones de inicializacion (inciso 1 y 2)
def inicializar_posiciones(N, L):
    """inciso 1: Red cúbica con perturbaciones para evitar superposiciones."""
    n_por_lado = int(np.ceil(N ** (1/3)))
    espaciado = L / n_por_lado
    
    x = np.linspace(espaciado/2, L - espaciado/2, n_por_lado)
    y = np.linspace(espaciado/2, L - espaciado/2, n_por_lado)
    z = np.linspace(espaciado/2, L - espaciado/2, n_por_lado)
    
    X, Y, Z = np.meshgrid(x, y, z)
    posiciones = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T[:N]
    
    perturbacion = (np.random.rand(N, 3) - 0.5) * (espaciado * 0.1)
    posiciones += perturbacion
    return posiciones

def inicializar_velocidades(N, T, m, kB):
    """inciso 2: Velocidades Maxwell-Boltzmann reescaladas a T exacta."""
    sigma_v = np.sqrt(kB * T / m)
    velocidades = np.random.normal(loc=0.0, scale=sigma_v, size=(N, 3))
    
    #eliminamos el momento del centro de masa
    v_cm = np.mean(velocidades, axis=0)
    velocidades -= v_cm
    
    #reescalado termico
    energia_cinetica = 0.5 * m * np.sum(velocidades**2)
    T_medida = (2.0 / 3.0) * energia_cinetica / (N * kB)
    velocidades *= np.sqrt(T / T_medida)
    
    return velocidades

#variables de estado global
pos = inicializar_posiciones(N, L_actual)
vel = inicializar_velocidades(N, T_actual, m, kB)


#motor fisico (incisos 3 y 4)
def aplicar_rebotes(posiciones, velocidades, L):
    """inciso 4: Colisiones elásticas con las paredes de tamaño L variable."""
    for i in range(3):
        #rebote en pared inferior
        choca_min = posiciones[:, i] < 0
        posiciones[choca_min, i] = np.abs(posiciones[choca_min, i]) # Devolver a la caja
        velocidades[choca_min, i] *= -1
        
        #rebote en pared superior
        choca_max = posiciones[:, i] > L
        posiciones[choca_max, i] = 2*L - posiciones[choca_max, i]   # Devolver a la caja
        velocidades[choca_max, i] *= -1
        
    #chequeo de seguridad por si L cambia bruscamente
    posiciones = np.clip(posiciones, 0, L)
    return posiciones, velocidades

def paso_integracion(posiciones, velocidades, L):
    """inciso 3: Avance temporal (Ideal Gas para rendimiento en tiempo real)."""
    #como las fuerzas son 0 (Gas Ideal)
    posiciones += velocidades * dt
    posiciones, velocidades = aplicar_rebotes(posiciones, velocidades, L)
    return posiciones, velocidades

#configuracion grafica y sliders
fig = plt.figure(figsize=(10, 8))
plt.subplots_adjust(bottom=0.25) #dejamos un pequeño espacio abajo para los sliders

#eje principal 3D
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='teal', s=30, alpha=0.8, edgecolors='k')

ax.set_xlim(0, L_actual)
ax.set_ylim(0, L_actual)
ax.set_zlim(0, L_actual)
ax.set_title("Dinámica Molecular de $H_2$ - Gas Ideal")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

#slider de temperatura
ax_temp = plt.axes([0.15, 0.1, 0.65, 0.03])
slider_temp = Slider(ax_temp, 'Temperatura [K]', 10.0, 1000.0, valinit=T_actual)

def update_temp(val):
    global vel
    T_nueva = slider_temp.val
    #recalcular T actual y reescalar velocidades
    energia_cinetica = 0.5 * m * np.sum(vel**2)
    T_medida = (2.0 / 3.0) * energia_cinetica / (N * kB)
    vel *= np.sqrt(T_nueva / T_medida)

slider_temp.on_changed(update_temp)

#slider de tamaño de caja
ax_L = plt.axes([0.15, 0.05, 0.65, 0.03])
slider_L = Slider(ax_L, 'Lado L [$\mu$m]', 2.0, 20.0, valinit=L_actual*1e6)

def update_L(val):
    global L_actual, pos, vel
    L_actual = slider_L.val * 1e-6 #convertir de micrometros a metros
    
    #actualizamos los limites visuales del grafico
    ax.set_xlim(0, L_actual)
    ax.set_ylim(0, L_actual)
    ax.set_zlim(0, L_actual)
    
    #forzamos a las partículas que quedaron fuera a rebotar hacia adentro
    pos, vel = aplicar_rebotes(pos, vel, L_actual)

slider_L.on_changed(update_L)

#animacion
def actualizar_frame(frame):
    global pos, vel
    
    #multiples pasos por frame para que la visualizacion sea mucho mas fluida
    pasos_por_frame = 50
    for _ in range(pasos_por_frame):
        pos, vel = paso_integracion(pos, vel, L_actual)
        
    scatter._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scatter,

ani = FuncAnimation(fig, actualizar_frame, frames=200, interval=30, blit=False)

#graficos analiticos en tiempo real, creamos una segunda ventana (fig2) para no saturar
fig2 = plt.figure(figsize=(10, 8))
fig2.canvas.manager.set_window_title('Análisis Termodinámico en Tiempo Real')

ax_T = fig2.add_subplot(221)
ax_P = fig2.add_subplot(222)
ax_hist = fig2.add_subplot(212)

#historial de datos para la grafica
tiempos = []
temps = []
presiones = []
tiempo_actual_ps = 0.0

#configuramos líneas de T y P
line_T, = ax_T.plot([], [], 'r-', lw=2)
ax_T.set_title("Temperatura vs Tiempo")
ax_T.set_xlabel("Tiempo [ps]")
ax_T.set_ylabel("T [K]")

line_P, = ax_P.plot([], [], 'b-', lw=2)
ax_P.set_title("Presión vs Tiempo")
ax_P.set_xlabel("Tiempo [ps]")
ax_P.set_ylabel("P [Pa]")

#limite de velocidad para el histograma
v_max = np.sqrt(3 * kB * 1000.0 / m) * 2.5
v_bins = np.linspace(0, v_max, 30)

def actualizar_graficos(frame):
    global tiempo_actual_ps
    
    #calculamos propiedades macroscopicas instantaneas
    v_magnitudes = np.linalg.norm(vel, axis=1)
    energia_cinetica = 0.5 * m * np.sum(v_magnitudes**2)
    T_inst = (2.0 / 3.0) * energia_cinetica / (N * kB)
    
    #presion P = N*kB*T / V 
    V = L_actual**3
    P_inst = N * kB * T_inst / V
    
    #guardamos el historial
    tiempo_actual_ps += (dt * 50) * 1e12 
    tiempos.append(tiempo_actual_ps)
    temps.append(T_inst)
    presiones.append(P_inst)
    
    #mantenemos solo los ultimos 100 puntos en memoria para que el grafico avance
    if len(tiempos) > 100:
        tiempos.pop(0)
        temps.pop(0)
        presiones.pop(0)
        
    #actualizamos curvas de T(t) y P(t)
    line_T.set_data(tiempos, temps)
    ax_T.set_xlim(tiempos[0], tiempos[-1] + 1)
    ax_T.set_ylim(min(temps)*0.9, max(temps)*1.1 + 1)
    
    line_P.set_data(tiempos, presiones)
    ax_P.set_xlim(tiempos[0], tiempos[-1] + 1)
    ax_P.set_ylim(min(presiones)*0.9, max(presiones)*1.1 + 1)
    
    #histograma vs Maxwell-Boltzmann Teorica
    ax_hist.clear()
    #histograma de la simulacion
    ax_hist.hist(v_magnitudes, bins=v_bins, density=True, color='skyblue', 
                 edgecolor='black', alpha=0.7, label='Simulación Molecular')
    
    #curva teorica MB
    v_teo = np.linspace(0, v_max, 100)
    coef = 4 * np.pi * (m / (2 * np.pi * kB * T_inst))**(1.5)
    f_v = coef * v_teo**2 * np.exp(-m * v_teo**2 / (2 * kB * T_inst))
    
    ax_hist.plot(v_teo, f_v, 'r-', lw=2, label=f'Maxwell-Boltzmann Teórica (T={T_inst:.1f} K)')
    ax_hist.set_title("Distribución de Velocidades")
    ax_hist.set_xlabel("Velocidad [m/s]")
    ax_hist.set_ylabel("Densidad de Probabilidad $f(v)$")
    ax_hist.legend(loc='upper right')
    ax_hist.set_xlim(0, v_max)
    
    #mantenemos las etiquetas para no perderlas al hacer clear()
    fig2.tight_layout()
    return line_T, line_P

#creamos la segunda animacion
ani2 = FuncAnimation(fig2, actualizar_graficos, interval=100, blit=False)
#con este plt.sjow() final mostramos todas las ventanas a la vez.
plt.show()