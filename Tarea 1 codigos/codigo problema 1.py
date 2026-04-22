import sys
import subprocess

def asegurar_librerias():
    """Verifica e instala las librerías necesarias automáticamente."""
    librerias = ["numpy", "scipy", "matplotlib"]
    for lib in librerias:
        try:
            __import__(lib)
        except ImportError:
            print(f"Librería '{lib}' no encontrada. Instalando automáticamente, por favor espera...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--break-system-packages"])
                print(f"'{lib}' instalada con éxito.")
            except Exception as e:
                print(f"Error al intentar instalar {lib}: {e}")
asegurar_librerias()

#ya instalamos la librerias
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def build_hamiltonian(N, J, B):
    """Construye la matriz del Hamiltoniano de Ising transversal."""
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    
    def op_sitio(op, j, N):
        res = op if j == 0 else I
        for i in range(1, N):
            res = np.kron(res, op if i == j else I)
        return res
        
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N - 1):
        H += J * (op_sitio(sigma_x, i, N) @ op_sitio(sigma_x, i + 1, N))
    for i in range(N):
        H += B * op_sitio(sigma_z, i, N)
    return H

#codigo que ejecuta los incisos b) y c)
if __name__ == "__main__":
    print("Iniciando cálculos del Hamiltoniano...")
    #Parametros
    N_espines = 4
    J_int = 1.0
    B_campo = 0.5
    
    #inciso b)
    H = build_hamiltonian(N_espines, J_int, B_campo)
    print(f"\n--- Inciso (b) ---")
    print(f"Dimensión de la matriz para N={N_espines}: {H.shape}")
    
    #Configuramos NumPy para que muestre todos los elementos de la matris sin mostrar . . .
    np.set_printoptions(threshold=np.inf, linewidth=200)
    
    #mostramos la parte real
    print("Matriz Hamiltoniana explícita de 16x16:")
    print(np.real(H))

#inciso c)
    print(f"\n--- inciso c) ---")
    print("Calculando evolución temporal para 3 regímenes y graficando...")
    
    N_c = 4
    J_c = 1.0
    #casos que nos piden: B/J << 1, B/J = 1, B/J >> 1
    casos_B = [0.1, 1.0, 10.0] 
    etiquetas = ["B/J << 1 (B=0.1)", "B/J = 1 (B=1.0)", "B/J >> 1 (B=10.0)"]
    
    tiempos = np.linspace(0, 10, 150)
    plt.figure(figsize=(9, 6))
    
    for B_val, etiqueta in zip(casos_B, etiquetas):
        H_c = build_hamiltonian(N_c, J_c, B_val)
        
        #estado inicial |psi(0)>: todos los spines up
        psi_0 = np.zeros(2**N_c, dtype=complex)
        psi_0[0] = 1.0 
        
        probabilidades = []
        for t in tiempos:
            U_t = expm(-1j * H_c * t)
            psi_t = U_t @ psi_0
            survival_prob = np.abs(np.vdot(psi_0, psi_t))**2
            probabilidades.append(survival_prob)
            
        plt.plot(tiempos, probabilidades, label=etiqueta)

    plt.xlabel("Tiempo (t)")
    plt.ylabel("Probabilidad de Supervivencia P(t)")
    plt.title("inciso c) - Evolución Cuántica para distintos B/J")
    plt.grid(True)
    plt.legend()
    plt.show(block=False)
    plt.pause(2)

    # ==========================================
    #inciso d): medicion de tiempos de calculo
    import time
    
    print(f"\n--- inciso d) ---")
    print("calculando tiempos de construcción y diagonalización...")
    
    N_valores = [4, 5, 6, 7, 8]
    tiempos_ejecucion = []
    
    for n in N_valores:
        #iniciamos cronometro
        t_inicio = time.perf_counter()
        
        #construimos hamiltoniano
        H_n = build_hamiltonian(n, J=1.0, B=1.0)
        
        #diagonalizanmos hamiltoniano (np.linalg.eigh es para las matrices hermiticas)
        valores_propios, vectores_propios = np.linalg.eigh(H_n)
        
        #stop cronometro
        t_fin = time.perf_counter()
        
        tiempo_total = t_fin - t_inicio
        tiempos_ejecucion.append(tiempo_total)
        
        print(f"Tamaño N = {n} (Matriz {2**n}x{2**n}): {tiempo_total:.6f} segundos")
    
    # ==========================================
    #inciso e): Gráfico
    print(f"\n--- inciso e) ---")
    print("Generando gráfico (Tiempo vs N)...")
    
    plt.figure(figsize=(8, 5))
    #graficamos los datos reales obtenidos en el inciso d)
    plt.plot(N_valores, tiempos_ejecucion, 'o-', color='red', markersize=8, label="Tiempos medidos")
    
    #cambiamos el eje Y a una escala logarítmica
    plt.yscale('log')
    plt.xlabel("Número de espines (N)")
    plt.ylabel("Tiempo de ejecución (segundos)")
    plt.title("Inciso (e) - Escalamiento del tiempo de simulación")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # ==========================================
    #inciso f): Estimación para N = 50
    print(f"\n--- inciso f) ---")
    print("Ajustando curva exponencial y estimando tiempo para N=50...")
    
    #para ajustar una curva exponencial t = A * exp(b*N), hacemos un ajuste lineal con el logaritmo: ln(t) = ln(A) + b*N
    log_tiempos = np.log(tiempos_ejecucion)
    coeficientes = np.polyfit(N_valores, log_tiempos, 1)
    
    b = coeficientes[0] #la pendiente
    ln_A = coeficientes[1] #la interseccion
    
    #estimamos para N = 50 usando la fórmula: t = exp(ln(A) + b*50)
    N_objetivo = 50
    t_50_segundos = np.exp(ln_A + b * N_objetivo)
    
    #convertimos a años para visualizar mejor
    segundosenunano = 365.25 * 24 * 3600
    t_50_anos = t_50_segundos / segundosenunano
    
    print(f"Ecuación de ajuste: t(N) = exp({b:.4f} * N + {ln_A:.4f})")
    print(f"Tiempo estimado para N=50: {t_50_segundos:.2e} segundos")
    print(f"Esto equivale a {t_50_anos:.2e} años")
    
    #la línea de tendencia en el gráfico
    N_tendencia = np.linspace(min(N_valores), max(N_valores)+1, 50)
    tiempo_tendencia = np.exp(ln_A + b * N_tendencia)
    plt.plot(N_tendencia, tiempo_tendencia, '--', color='black', label=f"Ajuste exponencial: $\\propto e^{{{b:.2f}N}}$")
    
    #agregamos una caja de texto con el resultado de N=50 en el gráfico
    resultado = f"Estimación N=50:\n{t_50_anos:.1e} años"
    plt.text(min(N_valores), max(tiempos_ejecucion), resultado, 
             fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))
    
    plt.legend()
    
    print("\nProblema 1 resuelto")
    plt.show() #mostramos grafico final y pausamos el programa