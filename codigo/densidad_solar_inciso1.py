import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def funcion_intepolacion():

    directorio = os.path.dirname(os.path.abspath(__file__))
    ruta = os.path.join(directorio, "..", "datos", "bs05op.dat")

    #creamos listas vacias para poder guardar los datos
    radios = []
    densidades = []
                    
#leemos el archivo ya limpio, que solamente tiene los datos solares que nos interesan
    with open(ruta, 'r') as archivo:
        for linea in archivo:
            #cortamos la linea en una lista de numeros usando espacios
            partes = linea.split() 
            
            #para evitar errores si el archivo de datos solares tiene lineas en blanco
            if len(partes) < 4:
                continue

            #extraemos las columnas de interes 
            r = float(partes[1])    #columna 2: radio
            rho = float(partes[3])    #columna 4: densidad
            
            #guardamos los numeros en las listas
            radios.append(r)
            densidades.append(rho)

#en este bloque de codigo lo que hacemos es limpiar el archivo de datos para quedarnos solamente con una matriz numerica, mas especificamente
#extraemos la columna 2 para el radio y la columna 4 para la densidad y guardamos los datos en 2 listas para interpolarlos

#transformamos las listas de python a matrices de numpy, para trabajar de manera mas rapida
    radios = np.array(radios)
    densidades = np.array(densidades)
    #para que el spline cubico funcione bien debemos ordenar los radios de menor a mayor, osean del nucleo solar a la superficie. para 
    #asegurarnos de que este todo ordenado usamos la funcion np.argsort(radios)
    orden = np.argsort(radios)
    radios = radios[orden]
    densidades = densidades[orden]

    #ahora creamos la funcion de interpolacion splinen. usamos bc_type='natural' como condicion de borde, al ser 'natural' forzamos a que la 2da
    #derivada de la curva en los extremos R=0 Y R=1 sea 0. esto para evitar que la curva haga saltos imposibles en los bordes de la grafica.
    d_spline = CubicSpline(radios, densidades, bc_type='natural')
    
    return d_spline, radios, densidades

if __name__ == "__main__":
    print("calculando spline de densidad solar")
    f_densidad, r_datos, rho_datos = funcion_intepolacion()
    
    #creamos 1000 puntos equiespaciados entre 0 y 1, luego usamos esos 1000 puntos en la funcion f_densidad
    r_continuo = np.linspace(0, 1, 1000)
    rho_continuo = f_densidad(r_continuo)
    
    #graficamos todo junto, de modo que la linea roja es el spline cubico y los puntos negros son los puntos de
    #interpolacion, podemos ver que la linea coincide con los puntos como tal
    plt.figure(figsize=(8, 5))
    plt.plot(r_datos, rho_datos, 'o', label="datos originales del modelo solar BS2005", markersize=3, color='black')
    plt.plot(r_continuo, rho_continuo, '-', label="interpolacion Spline Cubico", color='red')
    plt.title("perfil de densidad solar Modelo BS2005")
    plt.xlabel(r"radio solar ($R/R_{\odot}$)")
    plt.ylabel(r"densidad ($\text{g/cm}^3$)")
    plt.yscale('log')      #usamos una escala logaritmica, ya que es mejor para ver la caidade densidad
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    #guardamos la imagen en la carpeta de resultados
    directorio = os.path.dirname(os.path.abspath(__file__))
    ruta_resultados = os.path.join(directorio, "..", "resultados")
    ruta_imagen = os.path.join(ruta_resultados, "densidad_solar.png")
    
    if not os.path.exists(ruta_resultados):
        os.makedirs(ruta_resultados)

    plt.savefig(ruta_imagen, dpi=300)
    print(f"grafico guardada exitosamente en {ruta_imagen}")
    
    plt.show()