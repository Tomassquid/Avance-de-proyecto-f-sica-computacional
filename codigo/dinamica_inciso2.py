import numpy as np
import matplotlib.pyplot as plt

#importamos script que creamos previamente
from densidad_solar_inciso1 import funcion_intepolacion
from parametros import hamiltoniano_vacio
from evolucion import rk4_adaptativo

def función_viaje():
    #usamos unidades naturales (hbar = c = 1), 1 metro = 5.06e6 eV^-1
    #definimos el radio del sol = 6.96e8 metros
    R_sol = 6.96e8 * 5.06e6 
    
    #elegimos una energia de prueba dentro del rango [0.1, 15] MeV
    energia_MeV = 10
    
    print("1) cargando perfil de densidad solar")
    _, r_datos, rho_datos = funcion_intepolacion()
    
    print("2) construyendo el hamiltoniano en el vacio")
    H0 = hamiltoniano_vacio(energia_MeV)
    
    #definimos condiciones iniciales del neutrino, este nace en el centro del sol
    x_i = 0
    x_f = R_sol
    
    #el neutrino recien generado es electronico
    #el vector es [nu_e, nu_mu, nu_tau]
    psi_i = np.array([1 + 0j, 0 + 0j, 0 + 0j], dtype=complex)
    
    #configuramos RK4 adaptativo, de modo que el paso inicial es una fraccion del radio del sol
    h_i = R_sol / 1000
    tolerancia = 1e-3 #error aceptado por paso, relajamos este para que no dure 2 horas en ejecutarse
    

    #ejecutamos RK4, le damos intrucciones de comienzo y final con x_i y x_f, le damos
    #el estado cuantico inicial del neutrino electronico, sin olvidarnos de la tolerancia. esto
    #nos entrega 2 listas, una con las posiciones  con x_historial Y una matriz psi_historial
    print(f"3) iniciando RK4 para un neutrino de {energia_MeV} MeV")
    x_historial, psi_historial = rk4_adaptativo(
        x_i, 
        x_f, 
        psi_i, 
        h_i, 
        tolerancia, 
        r_datos, 
        rho_datos, 
        H0, 
        R_sol
    )
    print(f"se tomaron {len(x_historial)} pasos adaptativos")

    #datos y grafica, hacemos las convcersiones adecuadas
    r_normalizado = x_historial / R_sol
    
    #calculamos la probabilidad de cada sabor en cada punto del viaje
    prob_e = np.abs(psi_historial[:, 0])**2
    prob_mu = np.abs(psi_historial[:, 1])**2
    prob_tau = np.abs(psi_historial[:, 2])**2

    #verificamos que la probabilidad total sea 1
    suma_prob = prob_e + prob_mu + prob_tau
    print(f"comprobacion de conservacion de probabilidad (min, max): ({np.min(suma_prob):.4f}, {np.max(suma_prob):.4f})")

    #graficamos
    plt.figure(figsize=(10, 6))
    
    plt.plot(r_normalizado, prob_e, label=r'$P(\nu_e \rightarrow \nu_e)$ (Electronico)', color='blue')
    plt.plot(r_normalizado, prob_mu, label=r'$P(\nu_e \rightarrow \nu_\mu)$ (Muonico)', color='red')
    plt.plot(r_normalizado, prob_tau, label=r'$P(\nu_e \rightarrow \nu_\tau)$ (Tauonico)', color='green')
    
    plt.title(f"evolucion del vector de sabor en el sol (energia = {energia_MeV} MeV)")
    plt.xlabel(r"posicion en el sol ($R/R_{\odot}$)")
    plt.ylabel("probabilidad")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    función_viaje()