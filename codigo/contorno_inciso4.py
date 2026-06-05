import numpy as np
import matplotlib.pyplot as plt

#llamamos a los cripts que ya hicismos
from densidad_solar_inciso1 import funcion_intepolacion
from parametros import matriz_pmns, theta_13, theta_23, delta_cp, delta_m31
from evolucion import rk4_adaptativo


#definimos un hamiltoniano vacio, en donde podamos cambiar el angulo theta_12 y la masa delta_m21 en cada iteracion.
def h_dinamico(E_MeV, theta_12_dinamico, delta_m21_dinamico):
    E_eV = E_MeV * 1e6
    
    #paso 1: creamos una matriz de masa con las masa dinamicas
    M2 = np.diag([0, delta_m21_dinamico, delta_m31])
    
    #paso 2: creamos la matriz PMNS con el angulo theta_12 dinomico
    U = matriz_pmns(theta_12_dinamico, theta_13, theta_23, delta_cp)
    
    #paso 3: construimos el hamiltoniano H0
    H0 = (1 / (2 * E_eV)) * (U @ M2 @ np.conjugate(U).T)
    
    return H0

#para hacer el mapa creamos una cuadricula
def mapa_contorno():
    #condiciones iniciales
    R_sol = 6.96e8 * 5.06e6 
    x_i = 0
    x_f = R_sol
    psi_i = np.array([1 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    
    #definimos tolerancia
    h_i = R_sol / 5000
    tolerancia = 1e-3 
    
    #fijamos la energia en 10 MeV, ya que a estas energias el efecto MSW es mas fuerte
    E_MeV = 10

    print("cargando perfil solar")
    _, r_datos, rho_datos = funcion_intepolacion()

    #elegimos 20 valores para el angulo y 20 valores para las masas, de modo que tendremos
    #una cuadricula de 20x20
    n_puntos = 20 
    
    #angulo theta_12 va desde 10° hasta 80°, pero los pasamos en radianes para trabajar
    rango_theta = np.linspace(np.radians(10), np.radians(80), n_puntos)
    
    #diferencia de masa, usamos escala logaritmica
    rango_deltam21 = np.logspace(-6, -3, n_puntos)
    
    #definimos una matriz vacia, en donde anotaremos la probabilidad de cada uno de los viajes
    prob_supervivencia = np.zeros((n_puntos, n_puntos))

    print(f"\niniciando barrido de {n_puntos}x{n_puntos} ({n_puntos**2} simulaciones)")
    
    #usamos 2 ciclos for para recorrer toda la cuadricula, uno para filas y otro para columnas
    for i, dm2 in enumerate(rango_deltam21):
        for j, t12 in enumerate(rango_theta):
            
            #paso 1: planteamos las reglas para H0
            H0_actual = h_dinamico(E_MeV, t12, dm2)
            
            #paso 2: usamos el RK4 para disparar al neutrino a travez del sol
            x_hist, psi_historial = rk4_adaptativo(
                x_i, x_f, psi_i, h_i, tolerancia, 
                r_datos, rho_datos, H0_actual, R_sol
            )
            
            #paso 3: extraemos el neutrino que salio y lo guardamos en la cuadricula
            prob_e = np.abs(psi_historial[-1][0])**2
            prob_supervivencia[i, j] = prob_e
            
        #avisamos en la consola cada vez que termina una fila, siendo 20 neutrinos en total
        print(f"fila de masa {i+1}/{n_puntos} completada")

    print("\nsimulaciones completadas.generando mapa de contorno")
    plt.figure(figsize=(10, 7))
    
    #pasamos radianes a grados para el grafico
    Theta_grados, DeltaM = np.meshgrid(np.degrees(rango_theta), rango_deltam21)
    
    #dibujamos el mapa de color con 20 niveles de color de modo que los colores frios dan probabilidades bajas y colores calidos pprob altas
    mapa = plt.contourf(Theta_grados, DeltaM, prob_supervivencia, levels=20, cmap='viridis')
    plt.colorbar(mapa, label=r'Probabilidad de Supervivencia $P(\nu_e \rightarrow \nu_e)$')
    
    #marcamos el punto real (valor real de parametros.py) con la estrella joestar
    plt.plot(33.44, 7.53e-5, 'r*', markersize=15, markeredgecolor='black', label="valores reale")
    
    plt.yscale('log')
    plt.title(f"mapa de Contorno de efecto MSW (neutrino de {E_MeV} MeV)")
    plt.xlabel(r"angulo de mezcla $\theta_{12}$ (grados)")
    plt.ylabel(r"diferencia de masa $\Delta m_{21}^2$ ($eV^2$)")
    plt.legend()
    plt.grid(alpha=0.3, ls='--')
    plt.tight_layout()
    plt.show()

#ejecutamos el script
if __name__ == "__main__":
    mapa_contorno()