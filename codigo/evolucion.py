import numpy as np
from numba import njit 

#potencial MSW inducido por los electrones del sol
@njit  #usamos en todas las funciones que definimos para que numba no se confunda con la funcion densidad
def potencial_materia(rho):
    Y_e = 0.5 
    fact_conversion = 7.56e-14 
    
    V_e = fact_conversion * rho * Y_e
    #debemos tener en cuenta que el potencial solo afecta al neutrino
    V = np.zeros((3, 3), dtype=np.complex128)
    V[0, 0] = V_e
    return V
#con la funcion potencial_materia convertimos densidad en energia y creamos una matriz 3x3
#la cual coloca el potencial en la posicion (0,0) ya que solo el neutrino electronico siente
#el potencial como tal


@njit  #usamos en todas las funciones que definimos para que numba no se confunda con la funcion densidad
def ecuacion_schrodinger(x, psi, r_datos, rho_datos, H0, R_sol):
    #convertimos la posición (en eV^-1) a radio normalizado [0, 1]
    r_norma = x / R_sol
    if r_norma > 1: 
        r_norma = 1
        
    #usamos interpolacion nativa de numpy, que Numba procesa rapidamente
    rho = np.interp(r_norma, r_datos, rho_datos)  #llamamos esta funcion del script densidad
    V_materia = potencial_materia(rho)
    
    #hamiltoniano efec = hamiltoniano de vacio + potencial
    H_eff = H0 + V_materia
    #multiplicacion matricial del hamiltoniano por el vector de estado (psi)
    derivada = -1j * (H_eff @ psi)
    
    return derivada
#creamos un hamiltoniano efectivo real, usando el H0 que definimos en el script "parametros"
#y le sumamos el potencial de materia

#algoritmo RK4 clasico, calcula un solo paso del metodo RK4
@njit  #usamos en todas las funciones que definimos para que numba no se confunda con la funcion densidad
def rk4_paso(x, y, h, r_datos, rho_datos, H0, R_sol):
    k1 = h * ecuacion_schrodinger(x, y, r_datos, rho_datos, H0, R_sol)
    k2 = h * ecuacion_schrodinger(x + h/2, y + k1/2, r_datos, rho_datos, H0, R_sol)
    k3 = h * ecuacion_schrodinger(x + h/2, y + k2/2, r_datos, rho_datos, H0, R_sol)
    k4 = h * ecuacion_schrodinger(x + h, y + k3, r_datos, rho_datos, H0, R_sol)
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

#motor adaptativo, este integra la ecuacion a maxima velocidad preasignando la memoria
@njit  #usamos en todas las funciones que definimos para que numba no se confunda con la funcion densidad
def rk4_adaptativo(x_i, x_f, y_i, h_i, tol, r_datos, rho_datos, H0, R_sol):
    x = x_i
    y = y_i.copy()
    h = h_i
    
    #asignamos memoria para soportar los de pasos sin que se congele
    max_steps = 5000000
    #creamos listas para guardar el historial de la evolucion
    x_hist = np.zeros(max_steps, dtype=np.float64)
    y_hist = np.zeros((max_steps, 3), dtype=np.complex128)
    
    x_hist[0] = x
    y_hist[0] = y
    count = 1
    
    while x < x_f:
        #para no pasarnos de la superficie solar definimos
        if x + h > x_f:
            h = x_f - x

        #paso 1: calculamos con un paso 'h'    
        y1 = rk4_paso(x, y, h, r_datos, rho_datos, H0, R_sol)

        #paso 2: calculamos con dos medios pasos 'h/2'
        y_medio = rk4_paso(x, y, h/2, r_datos, rho_datos, H0, R_sol)
        y2 = rk4_paso(x + h/2, y_medio, h/2, r_datos, rho_datos, H0, R_sol)
        
        diff = np.abs(y2 - y1)
        #paso 3: estimamos error
        error = np.max(diff)
        if error == 0:
            error = 1e-20 #para no dividir por 0

        #paso 4: verificamos tolerancia    
        if error <= tol:
            x = x + h
            y = y2  #guardamos resultado
            if count < max_steps:
                x_hist[count] = x
                y_hist[count] = y
                count += 1

        #ajustamos la velocidad, osea el tamaño de cada paso        
        factor = 0.9 * (tol / error)**0.2
        factor = min(2, max(0.1, factor)) 
        h = h * factor
        
    return x_hist[:count], y_hist[:count]