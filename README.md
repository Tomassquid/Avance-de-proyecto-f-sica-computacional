# Simulador de Propagación de Neutrinos Solares y Efecto MSW

**Universidad Técnica Federico Santa María** **Departamento de Física** \
**Profesores:** Dr. Ariel Norambuena & Dr. Nicolas Viaux \
**Ayudante:** Cristóbal Benavides \
**Estudiante:** Tomas Silva  

---

## Descripción del Proyecto

Este repositorio contiene el desarrollo de un entorno de simulación computacional para el estudio de la dinámica cuántica de los neutrinos solares. El objetivo es modelar la propagación de estas partículas desde el núcleo del Sol hasta la Tierra, evaluando la conversión resonante y adiabática dictada por el Efecto Mikheyev-Smirnov-Wolfenstein (MSW) en un formalismo estricto de tres sabores.

El simulador resuelve la ecuación de Schrödinger espacial acoplada al perfil de densidad del Modelo Solar Estándar (BS2005) utilizando un método iterativo de Runge-Kutta de cuarto orden (RK4) con control de paso adaptativo. Además, cuenta con optimización *Just-In-Time* (JIT) para generar mapas de contorno en el espacio de parámetros (validando la solución LMA) y convoluciona los resultados para predecir el espectro de eventos por corriente cargada en detectores de agua pesada.

---

## Estructura del Repositorio

El material está organizado en los siguientes scripts y archivos:

**Datos y Parámetros Base:**
  * `datos/bs05op.dat` *(Si tienes esta carpeta)*: Archivo con los datos crudos del Modelo Solar Estándar BS2005.
  * `parametros.py`: Repositorio de constantes físicas universales y construcción de la matriz de mezcla PMNS.
  * `densidad_solar_inciso1.py`: Procesamiento e interpolación del perfil de densidad solar mediante Splines Cúbicos Naturales.

**Motor Numérico:**
  * `evolucion.py`: Núcleo matemático del proyecto. Contiene la definición del potencial de materia y el integrador RK4 adaptativo fuertemente acelerado a código de máquina.

**Scripts de Ejecución (Incisos de la Investigación):**
  * `dinamica_inciso2.py`: Simula y grafica la evolución espacial del vector de sabor cuántico desde el núcleo hasta la superficie solar.
  * `supervivencia_inciso3.py`: Calcula y grafica la probabilidad de supervivencia asintótica vs. energía (perfil de "bañera").
  * `contorno_inciso4.py`: Genera un mapa de contornos 2D en el espacio de parámetros cuánticos ($\theta_{12}$ vs $\Delta m_{21}^2$).
  * `resonancia_inciso5.py`: Analiza de forma analítica y visual la condición matemática de la resonancia MSW.
  * `histograma_inciso6.py`: Simula la respuesta fenomenológica y la tasa de eventos en un detector terrestre de agua pesada análogo a SNO.

**Documentación:**
  * `Informe_Final_Fisica_Computacional.pdf`: informe del 2do avance estilo *paper* con el marco teórico, metodología computacional, análisis físico detallado de resultados y conclusiones.
  * `Informe_Avance_Fisica_Computacional_Tomas_Silva.pdf`: Es el informe final, con *paper* con el marco teórico, metodología computacional, análisis físico detallado de resultados y conclusiones.
---

## Requisitos y Dependencias

El simulador está desarrollado enteramente en **Python**. Para ejecutar el código de manera óptima y compilar el motor matemático, es estrictamente necesario tener instaladas las siguientes librerías científicas:

* `numpy` (Manejo de arreglos de alta dimensionalidad y tipos de datos complejos)
* `scipy` (Rutinas de interpolación continua)
* `matplotlib` (Generación de la fenomenología gráfica)
* `numba` (Compilación *Just-In-Time* requerida en `evolucion.py` para evitar cuellos de botella)

Puedes instalarlas todas rápidamente desde la terminal usando `pip`:

```bash
pip install numpy scipy matplotlib numba
