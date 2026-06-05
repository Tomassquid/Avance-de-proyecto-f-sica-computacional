import numpy as np
import matplotlib.pyplot as plt

#llamamos a los cripts que ya hicismos
from parametros import delta_m21, theta_12
from evolucion import potencial_materia
from densidad_solar_inciso1 import funcion_intepolacion

#definimos funcion que calcula la densidad de masa necesaria para llegar a la resonancia MSW
def funcion_densidad(E_MeV, V_unitario):
    #paso 1: cambiamos la energia de MeV a eV
    E_eV = E_MeV * 1e6
    
    #paso 2: aplicamos condicion de resonancia, que viene dada por V_e = (dm2 * cos(2*theta)) / (2 * E)
    V_resonancia = (delta_m21 * np.cos(2 * theta_12)) / (2 * E_eV)
    
    #paso 3: tomamos V_resonancia, dividimos por V_unitario para obtener la densidad rho
    rho_resonancia = V_resonancia / V_unitario
    
    return rho_resonancia

#definimos funcion para implementar la condicion de resonancia, de modo que calcula la densidad de materia
#necesaria para que el denominador de la mezcla se anule
def resonancia_msw():
    print("codificando! factor de conversion de potencial de materia")
    V_unitario = np.real(potencial_materia(1)[0, 0])
    
    print("codificando! perfil solar para extraer limites fisicos")
    _, r_datos, rho_datos = funcion_intepolacion()
    rho_centro = np.max(rho_datos) # La densidad maxima en el nucleo
    
    #definimos rando de 0.1 a 50 MeV, con 200 puntos para suavidad
    energias_MeV = np.linspace(0.1, 50, 200)
    
    #calculamos la densidad de resonancia para cada energia
    densidades = [funcion_densidad(E, V_unitario) for E in energias_MeV]
    
    #graficamos
    print("generando grafico de condiciones de resonancia")
    plt.figure(figsize=(10, 6))
    
    #curva teorica de resonancia
    plt.plot(energias_MeV, densidades, '-', color='darkred', lw=2.5, 
             label=r'condicion de resonancia MSW ($\rho_{\text{res}}$)')
    
    #dibujamos una linea horizontal, representa el centro del sol
    plt.axhline(y=rho_centro, color='black', ls='--', lw=1.5, 
                label=rf'densidad en el centro del sol ($\approx {rho_centro:.1f}$ g/cm³)')
    
    ##marcamos la zona de resonancia sobreandola
    #usamos 1e-2 en vez de 0 para que no pete el logaritmo
    plt.fill_between(energias_MeV, 1e-2, rho_centro, color='green', alpha=0.12, 
                     label='Región de Resonancia Permitida (Dentro del Sol)')
    
    plt.yscale('log') #usamos escala logaritmica
    plt.ylim(bottom=1e-1)
    
    plt.title("densidad de Resonancia MSW vs. energia del neutrino", fontsize=13, pad=15)
    plt.xlabel("energia del neutrino $E$ (MeV)", fontsize=11)
    plt.ylabel(r"densidad de masa $\rho$ (g/cm³)", fontsize=11)
    
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='upper right', fontsize=10)
    plt.xlim(0.1, 50)
    plt.tight_layout()
    
    print("graficando grafico......")
    plt.show()

if __name__ == "__main__":
    resonancia_msw()