import numpy as np
import matplotlib.pyplot as plt
from codigo_inciso_1 import ModeloSolar #importamos el trabajo del codigo 1

class EvolucionNeutrino:
    def __init__(self, modelo_solar):
        self.sol = modelo_solar
        
        #parametros fisicos del PDG, en eV y radianes
        self.theta12 = np.radians(33.4)
        self.theta13 = np.radians(8.57)
        self.theta23 = np.radians(49.2)
        
        self.dm2_21 = 7.42e-5 # eV^2
        self.dm2_31 = 2.51e-3 # eV^2
        
        #energia del neutrino
        self.E = 10e6 # eV
        
        #factor de conversion espacial
        self.factor_espacial = 3.527e15 # eV^-1
        
        #pre-calculamos el hamiltoniano de vacio
        self.H_vac = self._construir_H_vac()

    def _construir_H_vac(self):
        """Construye la matriz de vacío 3x3 usando los ángulos de mezcla."""
        c12, s12 = np.cos(self.theta12), np.sin(self.theta12)
        c13, s13 = np.cos(self.theta13), np.sin(self.theta13)
        c23, s23 = np.cos(self.theta23), np.sin(self.theta23)
        
        #matrices de rotacion
        R12 = np.array([[c12, s12, 0], [-s12, c12, 0], [0, 0, 1]])
        R13 = np.array([[c13, 0, s13], [0, 1, 0], [-s13, 0, c13]])
        R23 = np.array([[1, 0, 0], [0, c23, s23], [0, -s23, c23]])
        
        #matriz PMNS (U)
        U = R23 @ R13 @ R12
        
        #matriz de masas
        M2 = np.diag([0, self.dm2_21, self.dm2_31])
        
        H = (1.0 / (2 * self.E)) * (U @ M2 @ U.conj().T)
        return H

    def H_total(self, r):
        """Calcula el Hamiltoniano total sumando Vacío + Materia en el radio r."""
        rho = self.sol.obtener_densidad(r)

        #asumimos fraccion de electrones en el Sol Ye ≈ 0.5
        V_msw = 7.56e-14 * rho * 0.5 
        
        H_mat = np.diag([V_msw, 0, 0]) 
        
        return self.H_vac + H_mat

    def derivada(self, r, psi):
        """Ecuación de Schrödinger: dPsi/dr = -i * H * factor * Psi"""
        H = self.H_total(r)
        #multiplicacion de matriz 3x3 por vector 3x1
        return -1j * self.factor_espacial * (H @ psi)

    def paso_rk4(self, r, psi, h):
        """Calcula un paso individual del método Runge-Kutta 4 Clásico."""
        k1 = h * self.derivada(r, psi)
        k2 = h * self.derivada(r + h/2.0, psi + k1/2.0)
        k3 = h * self.derivada(r + h/2.0, psi + k2/2.0)
        k4 = h * self.derivada(r + h, psi + k3)
        return psi + (k1 + 2*k2 + 2*k3 + k4) / 6.0

    def evolucionar_rk4_adaptativo(self, tol=1e-3):
        """
        Evoluciona el vector de sabor desde r=0 hasta r=1 adaptando el tamaño del paso.
        """
        r = 0.0
        psi = np.array([1+0j, 0+0j, 0+0j]) 
        h = 1e-4 
        
        radios_guardados = [r]
        prob_electronico = [np.abs(psi[0])**2] 
        
        print("Iniciando simulación RK4 Adaptativo...")
        progreso_impreso = 0.1 #para rastrear el 10%, 20%
        
        while r < 1.0:
            #se imprime el progreso
            if r >= progreso_impreso:
                print(f"Progreso: {int(progreso_impreso*100)}% del Sol recorrido...")
                progreso_impreso += 0.1

            if r + h > 1.0:
                h = 1.0 - r
                
            psi_paso_entero = self.paso_rk4(r, psi, h)
            psi_medio_1 = self.paso_rk4(r, psi, h/2.0)
            psi_medio_2 = self.paso_rk4(r + h/2.0, psi_medio_1, h/2.0)
            
            error = np.linalg.norm(psi_paso_entero - psi_medio_2)
            
            #agregamos un limite inferior de h para evitar que se quede congelado
            if error < tol or h < 1e-7: 
                r += h
                psi = psi_medio_2
                psi = psi / np.linalg.norm(psi)
                
                radios_guardados.append(r)
                prob_electronico.append(np.abs(psi[0])**2)
                
                h = h * min(2.0, 0.9 * (tol / (error + 1e-15))**0.2)
            else:
                h = h * max(0.1, 0.9 * (tol / error)**0.25)
                
        print("¡Simulación completada! Generando gráfico...")
        return np.array(radios_guardados), np.array(prob_electronico)


if __name__ == "__main__":
    import os
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_completa = os.path.join(carpeta_actual, "bs05_agsop.dat")
    
    #cargamos el modelo solar del inciso 1
    modelo = ModeloSolar(ruta_completa)
    
    #nuestra nueva clase de oscilacion
    simulacion = EvolucionNeutrino(modelo)
    
    #se evoluciona el neutrino con RK4 adaptativo
    radios, probabilidades = simulacion.evolucionar_rk4_adaptativo()
    
    #graficamos el resultado con suavizado estadistico
    plt.figure(figsize=(10, 5))
    
    #graficamos los datos crudos, el bloque azul, pero con transparencia para que quede como un fondo tenue.
    plt.plot(radios, probabilidades, 'b-', alpha=0.3, label=r'Oscilación Cruda (Fase Cuántica)')
    
    #aplicamos el filtro de Media Movil
    ventana = 500 
    
    #nos aseguramos de tener suficientes datos para la ventana
    if len(probabilidades) > ventana:
        #creamos un array de pesos uniformes
        pesos = np.ones(ventana) / ventana
        
        #convolucion: desliza la ventana promediando los datos
        prob_suavizada = np.convolve(probabilidades, pesos, mode='valid')
        
        #al usar mode='valid', el array resultante es mas pequeño. 
        #recortamos el array de radios para que coincidan los tamaños.
        margen = (ventana - 1) // 2
        radios_suavizados = radios[margen : -margen - (1 if ventana % 2 == 0 else 0)]
        
        # Graficamos la linea de tendencia en rojo
        plt.plot(radios_suavizados, prob_suavizada, 'r-', linewidth=2.5, label=r'Tendencia Promedio (Efecto MSW)')

    plt.title('Oscilación de Neutrinos Solares: Suavizado del Efecto MSW')
    plt.xlabel(r'Radio del Sol ($R/R_\odot$)')
    plt.ylabel(r'Probabilidad ($P_{\nu_e \rightarrow \nu_e}$)')
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()