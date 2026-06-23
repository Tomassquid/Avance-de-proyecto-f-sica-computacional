import numpy as np
import matplotlib.pyplot as plt

#llamamos a los cripts que ya hicismos
from densidad_solar_inciso1 import funcion_intepolacion
from parametros import hamiltoniano_vacio, matriz_pmns, theta_12, theta_13, theta_23, delta_cp
from evolucion import rk4_adaptativo

def calcular_supervivencia_vs_energia():
    #definimos condiciones iniciales
    R_sol = 6.96e8 * 5.06e6 
    x_i = 0
    x_f = R_sol
    
    #definimos el neutrino que nace como un neutrino electronico
    psi_i = np.array([1 + 0j, 0 + 0j, 0 + 0j], dtype=np.complex128)
    
    #configuramos el RK4 
    h_i = R_sol / 5000
    #usamos una tolerancia minima de 1e-3 para que el bucle no tarde tanto
    tolerancia = 1e-3 
    
    print("extrayendo datos de densidad solar")
    #extraemos los datos numericos crudos para pasarselos a Numba
    _, r_datos, rho_datos = funcion_intepolacion()
    
    #definimos el rango de energia
    puntos_energia = 15  
    energias_MeV = np.linspace(0.1, 15, puntos_energia)
    prob_supervivencia = []
    
    print(f"\niniciando simulacion para {puntos_energia} energias.....")
    print("esto va a tardar, asi que a la calma nomas!\n")
    
    #bucle para barrer todas las energias
    for i, E_MeV in enumerate(energias_MeV):
        print(f"simulando energia {i+1}/{puntos_energia} ({E_MeV:.2f} MeV)...", end=" ", flush=True)
        
        #construimos el hamiltoniano para la energia actual
        H0 = hamiltoniano_vacio(E_MeV)
        
        #ejecutamos la simulacion, que nos entrega el ultimo elemento de la lista, que seria el estado cuantico
        #del neutrino justo al salir a la superficie del sol
        x_hist, psi_hist = rk4_adaptativo(
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
        
        #extraemis estado cuantico final
        psi_f = psi_hist[-1] 

        #probabilidad que el neutrino siga siendo electronico
        #calculamos la matriz PMNS actual y matriz PMNS daga
        U = matriz_pmns(theta_12, theta_13, theta_23, delta_cp)
        U_daga = np.conjugate(U).T
        
        #proyectamos el estado final en la base de masas
        psi_masa = np.dot(U_daga, psi_f)
        
        #suma P_ee = sum( |<v_i | psi_superficie>|^2 * |U_{ei}|^2 )
        prob_e_f = (np.abs(psi_masa[0])**2 * np.abs(U[0,0])**2 + 
                    np.abs(psi_masa[1])**2 * np.abs(U[0,1])**2 + 
                    np.abs(psi_masa[2])**2 * np.abs(U[0,2])**2)
      



        print(f"probabilidad de salida: {prob_e_f:.4f}")
        prob_supervivencia.append(prob_e_f)

    #graficamos
    print("\nsimulacion completada, graficando.....")
    plt.figure(figsize=(10, 6))
     
    #se muestran puntos conectados por 'o-'
    plt.plot(energias_MeV, prob_supervivencia, 'o-', color='blue', label=r'$P(\nu_e \rightarrow \nu_e)$')
    
    plt.title("probabilidad de supervivencia de neutrinos solares vs energia")
    plt.xlabel("energia del neutrino (MeV)")
    plt.ylabel("probabilidad de supervivencia en la superficie")
    plt.grid(True, alpha=0.4)
    plt.legend()

    #limitamos los ejes
    plt.xlim(0, 16)
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    calcular_supervivencia_vs_energia()
