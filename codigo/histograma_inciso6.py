import numpy as np
import matplotlib.pyplot as plt

#llamamos a los cripts que ya hicismos
from densidad_solar_inciso1 import funcion_intepolacion
from parametros import hamiltoniano_vacio
from evolucion import rk4_adaptativo

#definimos una aproximacion del espectro de emision beta del Boro-8 en el sol
#esta genera entre 0 a 15 neutrinos
def flujo_boro8(E_MeV):
    E_max = 15
    if E_MeV <= 0 or E_MeV >= E_max:
        return 0
    #usamos la forma analitica de un decaimiento beta estandar
    return (E_MeV**2) * ((E_max - E_MeV)**2)

#definimos una funcion de probabilidad de interaccion por corriente cargada CC en agua pesada
#la reaccion es: nu_e + d -> p + p + e-
def seccion_eficaz(E_MeV):
    Q_umbral = 1.442 #MeV requeridos para romper el deuterio
    if E_MeV <= Q_umbral:
        return 0
    #entregamos la seccion eficaz, que crezca al cuadrado de la energia sobre el umbral
    return (E_MeV - Q_umbral)**2

#en un detector de agua pesada D2O, un neutrino choca con un nucleo de deuterio con la reaccion: nu_e + d -> p + p + e-
def detector_aguapesada():
    print("codificando simulacion del detector de Agua Pesada (CC)")
    
    #condiciones iniciales para el RK4 
    R_sol = 6.96e8 * 5.06e6 
    x_i = 0
    x_f = R_sol
    psi_i = np.array([1 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    h_i = R_sol / 5000
    tolerancia = 1e-3 
    
    print("extrayendo datos de densidad solar...")
    #extraemos los datos para que Numba pueda leerlos
    _, r_datos, rho_datos = funcion_intepolacion()

    #definimos los bins del histograma, en un rango de 2 MeV a 15 MeV en 30 saltos
    n_bins = 30
    energias = np.linspace(2, 15, n_bins)
    ancho_bin = energias[1] - energias[0]
    
    #guardamos la info
    eventos = []
    flujo_escalado = []
    
    print(f"\ncalculando espectro esperado en la tierra para {n_bins} rangos de energia")
    
    #hacemos un ciclo que para cada caja de energia, disparamos un neutrino desde el centro del sol usando el RK4 que hicimos

    for i, E in enumerate(energias):
        #paso 1: probabilidad de supervivencia
        H0 = hamiltoniano_vacio(E)
        x_hist, psi_historial = rk4_adaptativo(
            x_i, x_f, psi_i, h_i, tolerancia, 
            r_datos, rho_datos, H0, R_sol
        )
        prob_supervivencia = np.abs(psi_historial[-1][0])**2
        
        #paso 2: calculamos flujo de Boro-8 y seccion eficaz de deuterio
        flujo = flujo_boro8(E)
        sigma = seccion_eficaz(E)
        
        #paso 3: eventos totales = Flujo * Supervivencia * Choque
        evento = flujo * prob_supervivencia * sigma
        eventos.append(evento)
        
        #guardamos el flujo original
        flujo_escalado.append(flujo * sigma)
        
        print(f"codificando bin {i+1}/{n_bins} ({E:.1f} MeV)")

    print("\ncreando histograma de eventos CC")
    plt.figure(figsize=(10, 6))
    
    #dibujamos  barras del histograma detectado con el Efecto MSW
    plt.bar(energias, eventos, width=ancho_bin*0.8, color='royalblue', 
            edgecolor='black', alpha=0.8, label='Eventos Detectados (Con Efecto MSW)')
    
    #dibujamos curva de lo que se esperaria teoricamente sin oscilacion
    plt.plot(energias, flujo_escalado, '--', color='red', lw=2.5, 
             label='expectativa teorica Sin oscilacion')
    
    plt.title("histograma de eventos en detector de agua pesada", fontsize=13)
    plt.xlabel("energia del neutrino (MeV)", fontsize=11)
    plt.ylabel("numero de eventos relativos", fontsize=11)
    
    plt.xlim(1.5, 15.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    print("histograma:")
    plt.show()

if __name__ == "__main__":
    detector_aguapesada()