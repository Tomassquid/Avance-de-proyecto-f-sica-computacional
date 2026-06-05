import numpy as np

#definimos constantes fisicas que seran de utilidad
#diferencias de masas al cuadrado, en eV^2
delta_m21 = 7.53e-5
delta_m31 = 2.453e-3

#angulos de mezcla en radianes
theta_12 = np.radians(33.44)
theta_13 = np.radians(8.57)
theta_23 = np.radians(49.20)

#fase de violacion de CP en radianes
delta_cp = 0

#definimos la matriz de mezcla PMNS
def matriz_pmns(t12, t13, t23, dcp):
    # Senos y cosenos de los ángulos
    s12, c12 = np.sin(t12), np.cos(t12)
    s13, c13 = np.sin(t13), np.cos(t13)
    s23, c23 = np.sin(t23), np.cos(t23)
    
    #definimos fase compleja para la violacion de CP
    edcp = np.exp(-1j * dcp)
    
    #creamos una matriz U molde 3x3 vacia
    U = np.zeros((3, 3), dtype=complex)
    
    #llenamos la matriz
    U[0,0] = c12 * c13
    U[0,1] = s12 * c13
    U[0,2] = s13 * np.conjugate(edcp)
    
    U[1,0] = -s12 * c23 - c12 * s23 * s13 * edcp
    U[1,1] = c12 * c23 - s12 * s23 * s13 * edcp
    U[1,2] = s23 * c13
    
    U[2,0] = s12 * s23 - c12 * c23 * s13 * edcp
    U[2,1] = -c12 * s23 - s12 * c23 * s13 * edcp
    U[2,2] = c23 * c13
    
    return U

#definimos el hamiltoniano de vacio H0
def hamiltoniano_vacio(E_MeV):
    #conversion de energia del neutrino de MeV a eV
    E_eV = E_MeV * 1e6
    
    #creamos una matriz 3x3 donde solo tiene elementods la diagonal,
    #estos representan las masas de los neutrinos en su estado de propagacion.  
    M2 = np.diag([0, delta_m21, delta_m31])
    
    #la matriz unitaria PMNS
    U = matriz_pmns(theta_12, theta_13, theta_23, delta_cp)
    
    #definimos el factor 1 / (2E) del hamiltoniano de vacio
    fact_e = 1 / (2 * E_eV)
    
    #calculamos H0, para ello usamos la funcion np.conjugate(U.T) transpone 
    #y conjuga la matriz (U daga), necesario para obtener H0
    H0 = fact_e * (U @ M2 @ np.conjugate(U.T))
    
    return H0

if __name__ == "__main__":
    print("probando parametros.py")
    U_test = matriz_pmns(theta_12, theta_13, theta_23, delta_cp)
    print("magnitud de los elementos de la matriz PMNS:")
    #mostramos en pantalla el valor absoluto
    print(np.round(np.abs(U_test), 3))