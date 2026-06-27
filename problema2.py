import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Definimos las constantes fisicas que usaremos, todo en MeV y cm
mp  = 938.272    
me  = 0.511       
re  = 2.8179e-13  
NA  = 6.022e23    
K   = 0.307075    

#propiedades del agua 
Z_awa   = 10                     
A_awa   = 18.015                    
I_awa   = 75e-6                     
ro_awa = 1                           
Ne = ro_awa * (Z_awa / A_awa) * NA   


#funcion Bethe-Bloch calcula el poder de frenado -dE/dx para protones en agua, implementa la formula de Bethe-Bloch
#aqui Tmax es la energia maxima transferida a un electron libre en un choque
def funcion_bethebloch(T_arr):
    T_arr = np.asarray(T_arr, dtype=float) 
    gamma  = 1 + T_arr / mp
    beta2  = 1 - 1 / gamma**2
    Tmax = (2 * me * beta2 * gamma**2) / (1 + 2 * gamma * me / mp + (me / mp)**2)  #energia maxima transferida a un electron libre 
    log_arg = (2 * me * beta2 * gamma**2 * Tmax) / (I_awa**2)
    dEdx = (K * (Z_awa / A_awa) * (1 / beta2) * (0.5 * np.log(np.maximum(log_arg, 1e-30)) - beta2) * ro_awa)

    return np.maximum(dEdx, 0)

#grafico 1: Bethe-Bloch
fig, ax = plt.subplots(figsize=(7, 5))
T_plot = np.logspace(-1, np.log10(500), 1000)
ax.loglog(T_plot, funcion_bethebloch(T_plot), 'steelblue', linewidth=2)
ax.axvline(150, color='tomato', linestyle='--', alpha=0.75, label="150 MeV")
ax.set_xlabel("energia cinetica del proton [MeV]", fontsize=12)
ax.set_ylabel(r"$-dE/dx$ [MeV/cm]", fontsize=12)
ax.set_title("poder de frenado Bethe-Bloch protones en agua", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig("fig1_bethe_bloch.png", dpi=150)
plt.close()
print("Guardada: fig1_bethe_bloch.png")


#b) rango CSDA (Continuous Slowing Down Approximation)
#funcion CSDA calcula el rango de distancias que viaja un proton dentro del agua (considerando que pierde su energia de modo ideal)
def funcion_CSDA(E0, n_puntos=60000):
    rango_T = np.linspace(0.05, E0, n_puntos) #creamos rango de 0.05 hasta energia inical del proton, evitamos el cero para que no diverja
    dEdx_grilla = funcion_bethebloch(rango_T) #aplicamos ecuacion de blethe-bloch a las energias
    R = np.trapezoid(1 / dEdx_grilla, rango_T) #integramos 1/dEdx en energia con la regla del trapecio
    return R  #en cm


#tomamos los datos de referencia de la base de datos PSTAR del NIST: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
nist = {50: 2.207, 150: 15.76, 250: 38.00}   #todo en cm

for E0 in [50, 150, 250]:
    R_cal  = funcion_CSDA(E0)                                            
    R_nist = nist[E0]            
    error_rela = abs(R_cal - R_nist) / R_nist * 100   #calcula el error porcentual relativo   
    
print()

#Para el eje de CSDA, de la figura 2, necesitamos el array de rangos
E0_arr = np.linspace(5, 300, 80)
R_arr  = np.array([funcion_CSDA(E0) for E0 in E0_arr])

#grafico 2: rango CSDA vs NIST
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(E0_arr, R_arr * 10, 'steelblue', linewidth=2, label="Bethe-Bloch (este trabajo)")
E_nist = np.array([50, 150, 250])
R_nist = np.array([2.207, 15.76, 38])
ax.scatter(E_nist, R_nist * 10, color='tomato', zorder=5, s=80, label="NIST PSTAR")
ax.set_xlabel(r"$E_0$ [MeV]", fontsize=12)
ax.set_ylabel("Rango CSDA [mm]", fontsize=12)
ax.set_title("Rango CSDA de protones en agua", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig2_rango_csda.png", dpi=150)
plt.close()
print("Guardada: fig2_rango_csda.png")


#pre-calculo de tablas para acelerar la simulacion MC
T_tabla    = np.linspace(0.01, 600, 300000)      #generamos un vector con 300.000 valores de energia
dEdx_tabla = funcion_bethebloch(T_tabla)           #calculamos el poder de frenado (dE/dx) para las 300.000 energias
beta2_tabla = 1 - 1 / (1 + T_tabla / mp)**2  #calculamos el factor relativista beta al cuadrado para todas esas energias

bb_interp    = interp1d(T_tabla, dEdx_tabla,  kind='linear',fill_value=(0, 0), bounds_error=False) #interpolador para el poder de frenado
beta2_interp = interp1d(T_tabla, beta2_tabla, kind='linear',fill_value=(0, 1), bounds_error=False) #interpolador para el factor beta al cuadrado

#Motor Monte Carlo para simular el paso de protones por agua.
def simular(E0, N=10000, dx=0.01, z_max=22,straggling=False, semilla=42):
    np.random.seed(semilla) #fijamos la semilla aleatoria

    #creamos el fantoma del agua, para ello dividimos la profundidad maxima
    n_bins   = int(z_max / dx)
    z_bins   = np.linspace(0, z_max, n_bins + 1)
    z_centro = 0.5 * (z_bins[:-1] + z_bins[1:])     #representa el centro de cada tajada de agua
    dosis    = np.zeros(n_bins)                     #guardamos energia totasl depositada a cada tarjeta
    #todos los protones parten con la misma energia
    energias = np.full(N, float(E0))
    activos = np.ones(N, dtype=bool)   #true mientras el proton tiene energia

    #creamos ciclo espacial tajada por tajada
    for j in range(n_bins):
        
        #si ya no quedan protones con energia paramos la simulacion
        if not np.any(activos):
            break  

        T_act = energias[activos]    #tomamos solo la energia de los protones que siguen moviendose
        #perdida de enrgia
        dEdx_act  = bb_interp(T_act)
        dE_media  = dEdx_act * dx        #la energia media perdida en esta tajada

        #fenomeno del straggling
        if straggling:
            #calculamos varianza con la formula gaussiana de Bohr
            b2_act = beta2_interp(T_act)
            sigma2 = (4 * np.pi * re**2 * me**2 * Ne * (1 / np.maximum(b2_act, 1e-10)) * dx)
            sigma  = np.sqrt(np.maximum(sigma2, 0))
            ruido = np.random.normal(0, sigma)   #creamos ruido aleatorio a la perdida media
            dE    = dE_media + ruido               #agregamos el ruido
            dE_dep = np.clip(dE, 0, T_act)       #la energia depositada no puede ser mayor a lo que le queda
        else:
            dE_dep = dE_media #si el straggling se apaga, todos los protones pierden lo mismo, para el caso ideal

        #registramos y actualizamos la dosis
        dosis[j] += np.sum(dE_dep)                 #sumamos toda la energia que depositaron los protones activos
        energias[activos] -= dE_dep                   #restamos a los protones la energaa que perdieron
        activos = activos & (energias > 0.05)         #actualizamos la lista de activos

    #entregamos profundidad y la Dosis depositada, que son el eje x e y respectivamente
    return z_centro, dosis

#c) Simulacion sin straggling, con 10^4 protones y 150 MeV
#definimos parametros iniciales
E0    = 150
N     = 10000
dx    = 0.01    # 0.1 mm en cm
z_max = 22

print("=" * 58)
print("c)  Simulacion sin straggling  (E0=150 MeV, N=10000)")
print("=" * 58)

#ejecutamos la simulacion de modo ideal, llamamos simular()
z_c, dosis_sin = simular(E0, N=N, dx=dx, z_max=z_max, straggling=False, semilla=42)

i_pico_sin = np.argmax(dosis_sin)     #identificamos el pico de bragg -_-
z_pico_sin   = z_c[i_pico_sin]        #usamos indice para buscar en el arreglo espacial, identificamos la profundidad a la que ocurre el maximo
R_csda_150   = funcion_CSDA(E0)         #calculamos distancia teorica con la funcion_CSDA

#mostramos resultados
print(f"  Pico de Bragg (simulacion): {z_pico_sin*10:.2f} mm")
print(f"  R_CSDA (Bethe-Bloch):       {R_csda_150*10:.2f} mm")
print(f"  Diferencia:                 {abs(z_pico_sin - R_csda_150)*10:.2f} mm")
print()

#grafico 3: pico de Bragg sin straggling
dosis_sin_norm = dosis_sin / dosis_sin.max()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_c * 10, dosis_sin_norm, 'steelblue', linewidth=1.5, label="D(z)")
ax.axvline(z_pico_sin * 10, color='tomato', linestyle='--', alpha=0.8, label=f"Pico Bragg: {z_pico_sin*10:.1f} mm")
ax.axvline(R_csda_150 * 10, color='seagreen', linestyle=':', alpha=0.8, label=r"$R_{CSDA}$" + f": {R_csda_150*10:.1f} mm")
ax.set_xlabel("profundidad en agua [mm]", fontsize=12)
ax.set_ylabel("dosis normalizada [u.a.]", fontsize=12)
ax.set_title(f"pico de Bragg  —  sin straggling\n"r"($E_0$" + f"={E0} MeV,  N={N},  "r"$\Delta x$" + f"={dx*10:.1f} mm)", fontsize=11)
ax.legend(fontsize=11)
ax.set_xlim(0, z_max * 10.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fig3_bragg_sin_straggling.png", dpi=150)
plt.close()



#d) repetir con straggling y comparar

#simulacion realista con straggling
z_c, dosis_con = simular(E0, N=N, dx=dx, z_max=z_max, straggling=True, semilla=42)  #usamos funcion simular() con straggling activado
#buscamos el pico de bragg -.-
i_pico_con = np.argmax(dosis_con)                                                
z_pico_con   = z_c[i_pico_con]
print(f" pico de bragg (con straggling): {z_pico_con*10:.2f} mm")

#funcion para medir ensanchamiento, Ancho a mitad de altura (FWHM) del pico de Bragg
def ensanchamiento(z, dosis):
    idx_max = np.argmax(dosis)         #encontramos valor maximo de la dosis
    D_mitad = dosis[idx_max] / 2       #calculamos la mitad de la dosis

    idx_izq = idx_max                  #camina hacia la izquierda desde el pico hasta que la dosis caiga por debajo de la mitad
    while idx_izq > 0 and dosis[idx_izq] > D_mitad:
        idx_izq -= 1

    idx_der = idx_max                  #camina hacia la derecha desde el pico hasta que la dosis caiga por debajo de la mitad
    while idx_der < len(dosis) - 1 and dosis[idx_der] > D_mitad:
        idx_der += 1

    #entregamos la distancia entre esos dos puntos, el ancho
    return z[idx_der] - z[idx_izq]

#comparamos el caso ideal y el caso real
ensanchamiento_sin = ensanchamiento(z_c, dosis_sin)
ensanchamiento_con = ensanchamiento(z_c, dosis_con)

#conversion de desviacion estandar, usamos FWHM = 2.355 * sigma, despejando sigma. usamos un perfil gaussiano
sigma_R_sin = ensanchamiento_sin / 2.355
sigma_R_con = ensanchamiento_con / 2.355

#grafico 4: Comparacion con / sin straggling
dosis_con_norm = dosis_con / dosis_con.max()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

#panel izquierdo: perfil completo
ax = axes[0]
ax.plot(z_c * 10, dosis_sin_norm, 'steelblue', lw=1.5, label="Sin straggling")
ax.plot(z_c * 10, dosis_con_norm, 'tomato',    lw=1.5, label="Con straggling", alpha=0.85)
ax.set_xlabel("profundidad [mm]", fontsize=12)
ax.set_ylabel("dosis normalizada [u.a.]", fontsize=12)
ax.set_title("perfil de dosis completo", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(0, z_max * 10)
ax.grid(True, alpha=0.3)

#Panel derecho: zoom en la region del pico
ax = axes[1]
ax.plot(z_c * 10, dosis_sin_norm, 'steelblue', lw=1.5, label="Sin straggling")
ax.plot(z_c * 10, dosis_con_norm, 'tomato',    lw=1.5, label="Con straggling", alpha=0.85)

#marcamos el FWHM del pico con straggling
y_nivel = 0.5
ax.annotate("", xy=(z_pico_con * 10 + ensanchamiento_con * 10 / 2, y_nivel), xytext=(z_pico_con * 10 - ensanchamiento_con * 10 / 2, y_nivel), arrowprops=dict(arrowstyle='<->', color='tomato', lw=1.5))
ax.text(z_pico_con * 10, y_nivel + 0.04, f"FWHM = {ensanchamiento_con*10:.1f} mm", color='tomato', ha='center', fontsize=10)
ax.set_xlabel("profundidad [mm]", fontsize=12)
ax.set_ylabel("dosis normalizada [u.a.]", fontsize=12)
ax.set_title("zoom en la region del pico", fontsize=11)
ax.legend(fontsize=10)
ax.set_xlim(130, 185)
ax.grid(True, alpha=0.3)

fig.suptitle(f"efecto del energy straggling  "r"($E_0$" + f"={E0} MeV, N={N})", fontsize=12)
plt.tight_layout()
plt.savefig("fig4_straggling.png", dpi=150)
plt.close()



#zona de graficacion

#grafico 5: Panel resumen 2x2
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
T_resumen = np.logspace(-1, np.log10(500), 1000)
axs[0, 0].loglog(T_resumen, funcion_bethebloch(T_resumen), 'steelblue', lw=2)
axs[0, 0].axvline(150, color='tomato', ls='--', alpha=0.7, label="150 MeV")
axs[0, 0].set_xlabel("T [MeV]"); axs[0, 0].set_ylabel(r"$-dE/dx$ [MeV/cm]")
axs[0, 0].set_title("a) Bethe-Bloch"); axs[0, 0].grid(True, which='both', alpha=0.3)
axs[0, 0].legend(fontsize=10)

axs[0, 1].plot(E0_arr, R_arr * 10.0, 'steelblue', lw=2, label="Bethe-Bloch")
axs[0, 1].scatter(E_nist, R_nist * 10.0, color='tomato', s=70, zorder=5, label="NIST PSTAR")
axs[0, 1].set_xlabel(r"$E_0$ [MeV]"); axs[0, 1].set_ylabel("Rango CSDA [mm]")
axs[0, 1].set_title("b) rango CSDA"); axs[0, 1].grid(True, alpha=0.3)
axs[0, 1].legend(fontsize=10)

axs[1, 0].plot(z_c * 10.0, dosis_sin_norm, 'steelblue', lw=1.5)
axs[1, 0].axvline(z_pico_sin * 10.0, color='tomato', ls='--', alpha=0.8, label=f"Pico: {z_pico_sin*10:.1f} mm")
axs[1, 0].axvline(R_csda_150 * 10.0, color='seagreen', ls=':', alpha=0.8, label=r"$R_{CSDA}$")
axs[1, 0].set_xlabel("Profundidad [mm]"); axs[1, 0].set_ylabel("Dosis norm. [u.a.]")
axs[1, 0].set_title("c) Bragg sin straggling"); axs[1, 0].set_xlim(0, z_max * 10.0)
axs[1, 0].grid(True, alpha=0.3); axs[1, 0].legend(fontsize=10)

axs[1, 1].plot(z_c * 10.0, dosis_sin_norm, 'steelblue', lw=1.5, label="Sin straggling")
axs[1, 1].plot(z_c * 10.0, dosis_con_norm, 'tomato',    lw=1.5, alpha=0.85, label="Con straggling")
axs[1, 1].set_xlabel("profundidad [mm]"); axs[1, 1].set_ylabel("Dosis norm. [u.a.]")
axs[1, 1].set_title("d) efecto straggling (zoom)"); axs[1, 1].set_xlim(130, 185)
axs[1, 1].grid(True, alpha=0.3); axs[1, 1].legend(fontsize=10)

fig.suptitle(r"simulacion Monte Carlo  $-$  Terapia con protones  "f"($E_0={E0}$ MeV, N={N})", fontsize=13)
plt.tight_layout()
plt.savefig("fig5_resumen.png", dpi=150, bbox_inches='tight')
plt.close()

