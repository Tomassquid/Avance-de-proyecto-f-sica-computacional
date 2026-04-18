import numpy as np                          #calculadora cientifica que nos permite manejar numeros grandes
import matplotlib.pyplot as plt             #abre la ventana de los graficos
from scipy.interpolate import interp1d      #libreria de scipy que nos conecta los datos con una curva, nos permite interpolar en 1D
import os                                   #busca los archivos y rutas necesarias (que todo este dentro de la misma carpeta)

class ModeloSolar:
    """
    Clase que representa el Modelo Solar Estándar (BS2005).
    Se encarga de cargar los datos de densidad desde un archivo local y generar una interpolación continua.
    """
    
    def __init__(self, ruta_archivo="bs05_agsop.dat"):
        """
        Constructor de la clase. Carga los datos locales y prepara la interpolación.
        """
        self.fuente = ruta_archivo
        self.radio = None
        self.densidad = None
        self.funcion_densidad = None
        
        # Al instanciar la clase, cargamos los datos inmediatamente
        self._cargar_datos()
        self._crear_interpolacion()
#creamos la plantilla modelosolar, donde __init__ es el constructor, se ejecuta al instante. se define los atributos vacios (none)
#para guardar los datos, para despues llamar a los dos metodos internios (cargar_datos y crear_interpolacion) para que apenas nazca
#el objeto pueda usarse

    def _cargar_datos(self):
        """
        Método privado (encapsulación) para leer el archivo BS2005 local.
        Extrae el Radio (Columna 1) y la Densidad (Columna 3). 
        Nota: En Python el índice empieza en 0.
        """
        print(f"Cargando datos desde archivo local: {self.fuente} ...")
        
        # Leemos el archivo limpio. Extraemos directamente las columnas de radio (1) y densidad (3).
        datos = np.loadtxt(self.fuente, usecols=(1, 3))
        
        self.radio = datos[:, 0]     #R/R_sol
        self.densidad = datos[:, 1]  #g/cm^3
#la variable (datos) se encarga de abrir el archivo bs05_agsop.dat y lee todos los numeros.
#usecols=(1, 3) le dice a numpy que solo lea y extraiga la columna 1 y la columna 3 (radio y densidad, respectivamente)
#separamos los datos de las dos columnas en dos listas distintas (en las variables self.radio y self.densidad)

    def _crear_interpolacion(self):
        """
        Método privado que crea la función matemática continua a partir de los datos discretos.
        Se usa un spline cúbico ('cubic') para que la curva sea suave y físicamente realista.
        """
        # interp1d crea una función llamable que evalúa la interpolación
        self.funcion_densidad = interp1d(self.radio, self.densidad, kind='cubic', fill_value="extrapolate")

    def obtener_densidad(self, r):
        """
        Método público para obtener la densidad en cualquier radio r (en unidades de R_sol).
        """
        # Evitar valores negativos por extrapolación numérica en el borde
        dens = self.funcion_densidad(r)
        return np.maximum(dens, 0.0)
#interp1d toma las listas de dos puntos y las une usando polinomios de 3er grado (de ahi viene el kind='cubic'). esto asegura 
#que la curva sea suave.
#self.funcion_densidad(r) es una funcion que si le entregas un radio, te entrega una densidad.
#obtener_densidad(self, r) usa self.funcion_densidad(r), ademas le forzamos con np.maximum(dens, 0.0), ya que al tratar de 
#curvar la linea al final del radio del sol, podria darnos una densidad negatica por error. np.maximum(dens, 0.0) nos
#asegura que el valor minimo posible sea 0

    def validar_y_graficar(self):
        """
        Método para validar que la interpolación tiene sentido físico,
        comparando los datos originales (puntos) con la función continua (línea).
        """
        r_continuo = np.linspace(0, 1, 500)
        dens_continua = self.obtener_densidad(r_continuo)

        plt.figure(figsize=(8, 5))
        plt.plot(r_continuo, dens_continua, 'b-', label='Interpolación (Spline Cúbico)')
        # Graficamos solo 1 de cada 20 puntos originales para que se vea claro
        plt.plot(self.radio[::20], self.densidad[::20], 'ro', label='Datos Discretos BS2005') 
        
        plt.title('Perfil de Densidad Solar (Modelo BS2005)')
        plt.xlabel(r'Radio Solar ($R/R_\odot$)')
        plt.ylabel(r'Densidad ($\text{g/cm}^3$)')
        plt.yscale('log') # Escala logarítmica es mejor para visualizar la caída abrupta
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.show()
#np.linspace(0, 1, 500) crea una lista de 500 numeros espaciados entre 0 y 1. con esos 500 numeros calculamos las 500
#densidades continuas (linea azul del grafico). el resto de lineas de plot son el formato de un grafico estetico.

if __name__ == "__main__":
    #Obtenemos la ruta de la carpeta donde está guardado ESTE script de Python
    carpeta_actual = os.path.dirname(os.path.abspath(__file__))
    
    #Unimos la ruta de la carpeta con el nombre del archivo
    ruta_completa = os.path.join(carpeta_actual, "bs05_agsop.dat")
    
    #Instanciamos el objeto usando la ruta completa
    sol = ModeloSolar(ruta_completa)
    
    #Validamos visualmente
    sol.validar_y_graficar()
    
    #Probamos que podemos pedirle valores intermedios
    r_test = 0.153
    print(f"\nLa densidad en r = {r_test} R_sol es: {sol.obtener_densidad(r_test):.2f} g/cm^3")

