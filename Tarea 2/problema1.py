import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import time

#b) Generacion de parametros
np.random.seed(42)

#discretizacion temporal: 1000 puntos en [0, 10]
N = 3000
Nt = 1000
t = np.linspace(0, 10, Nt)
sigma = 0.02

#generamos 3000 pares (γ, k) al azar dentro de los rangos del enunciado, desde distribuciones uniformes
gammas = np.random.uniform(0.05, 1, N)
ks     = np.random.uniform(1,  5, N)

#las etiquetas son los pares (gamma, k) que el modelo debe predecir
y = np.column_stack((gammas, ks))

#la ecuacion x'' + gamma*x' + k*x = 0 se reescribe como sistema de primer orden:
#   dx/dt = v
#   dv/dt = -gamma*v - k*x
def oscilador(estado, t, gamma, k):
    x, v = estado
    return [v, -gamma * v - k * x]

#resolvemos numericamente las N EDOs con condiciones iniciales x(0)=1, v(0)=0
X_clean = np.zeros((N, Nt))
ic = [1, 0]

#para cada par (γ, k) se resuelve la EDO y se guarda solo la posicion x(t)
print(f"generando {N} señales... puede tomar unos segundos.")
for i in range(N):
    sol = odeint(oscilador, ic, t, args=(gammas[i], ks[i]))
    X_clean[i] = sol[:, 0]  # solo guardo la posicion x(t), no la velocidad

#a cada punto de cada señal se le suma ruido con distribucion normal N(0, sigma^2)
X_obs = X_clean + np.random.normal(0, sigma, X_clean.shape)


#para ver el efecto de cada parametro por separado, fijamos uno y variamos el otro.
#k fijo, gamma variando nos muestra como cambia el decaimiento
#gamma fijo, k variando nos muestra como cambia la frecuencia
gamma_vals = [0.1, 0.4, 0.9]   
k_vals     = [1, 2.5, 5]  
k_fijo     = 2.5
gamma_fijo = 0.3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

#subplot izquierdo: variamos gamma manteniendo k cte
for g in gamma_vals:
    sol = odeint(oscilador, ic, t, args=(g, k_fijo))
    ax1.plot(t, sol[:, 0], label=f'γ={g}')
ax1.set_title(f"Efecto de γ (k={k_fijo} fijo)")
ax1.set_xlabel("Tiempo (t)")
ax1.set_ylabel("Posición x(t)")
ax1.legend()
ax1.grid(True)

#subplot derecho: variamos k manteniendo gamma cte
for k in k_vals:
    sol = odeint(oscilador, ic, t, args=(gamma_fijo, k))
    ax2.plot(t, sol[:, 0], label=f'k={k}')
ax2.set_title(f"Efecto de k (γ={gamma_fijo} fijo)")
ax2.set_xlabel("Tiempo (t)")
ax2.set_ylabel("Posición x(t)")
ax2.legend()
ax2.grid(True)

plt.suptitle("oscilador amortiguado: variacion de parametros (sin ruido)")
plt.tight_layout()
plt.show(block=False)
plt.pause(0.1)

# c) Entrenamiento de modelos de regresion

# dividimos en 80% entrenamiento y 20% validacion, para evaluar que tan bien
# predicen en datos que nunca vieron
X_train, X_test, y_train, y_test = train_test_split( X_obs, y, test_size=0.20, random_state=42)

# Random Forest: ensamble de 100 arboles de decision. cada arbol aprende a predecir
# (γ, k) desde la señal completa, y la prediccion final es el promedio de todos.
# n_jobs=-1 usa todos los nucleos del procesador para entrenar en paralelo.
print("\nentrenamiento de Random Forest...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
t0 = time.time()
rf.fit(X_train, y_train)
print(f"  listo en {time.time() - t0:.1f} s")

# MLP (red neuronal de dos capas ocultas de 100 y 50 neuronas): aprende una funcion
# no lineal que mapea las 1000 muestras de la señal a los dos parametros fisicos.
# max_iter=500 es el numero maximo de epocas (iteraciones del optimizador).
print("entrenando MLP (100-50)...")
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
t0 = time.time()
mlp.fit(X_train, y_train)
print(f"  listo en {time.time() - t0:.1f} s")

# toma el modelo ya entrenado, pide predicciones sobre el conjunto de evaluacion,
# y calcula el RMSE por separado para γ y para k.
def calcular_rmse(modelo, X_eval, y_eval):
    """devuelve el RMSE por separado para gamma y para k."""
    y_pred = modelo.predict(X_eval)
    rmse_g = np.sqrt(mean_squared_error(y_eval[:, 0], y_pred[:, 0]))
    rmse_k = np.sqrt(mean_squared_error(y_eval[:, 1], y_pred[:, 1]))
    return rmse_g, rmse_k

# evaluamos ambos modelos en el conjunto de test
rmse_g_rf,  rmse_k_rf  = calcular_rmse(rf,  X_test, y_test)
rmse_g_mlp, rmse_k_mlp = calcular_rmse(mlp, X_test, y_test)


# d) Efecto del nivel de ruido sobre el error de prediccion
# para cada nivel de ruido σ se genera una version nueva de los datos,
# se entrena un Random Forest con 50 arboles, y se guarda el RMSE resultante.
sigmas = [0, 0.01, 0.02, 0.05, 0.1]
rmse_g_list = []
rmse_k_list = []

print("\nevaluando distintos niveles de ruido...")
for sig in sigmas:
    # cuando sig=0 usamos directamente X_clean para evitar llamadas innecesarias al RNG
    if sig == 0:
        X_noisy = X_clean.copy()
    else:
        X_noisy = X_clean + np.random.normal(0, sig, X_clean.shape)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_noisy, y, test_size=0.20, random_state=42
    )

    # usamos 50 arboles para que el loop sea mas rapido
    rf_sig = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf_sig.fit(X_tr, y_tr)

    rg, rk = calcular_rmse(rf_sig, X_te, y_te)
    rmse_g_list.append(rg)
    rmse_k_list.append(rk)
    print(f"  σ={sig:.2f}  →  RMSE γ: {rg:.4f}   RMSE k: {rk:.4f}")

# graficamos como crece el error a medida que aumenta el ruido
plt.figure(figsize=(8, 5))
plt.plot(sigmas, rmse_g_list, 'o-', color='red',  label='RMSE γ (amortiguacion)')
plt.plot(sigmas, rmse_k_list, 's-', color='blue', label='RMSE k (constante elastica)')
plt.title("RMSE vs nivel de ruido en los datos de entrenamiento")
plt.xlabel("σ (desviacion estandar del ruido)")
plt.ylabel("RMSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()