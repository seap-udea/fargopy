# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.patches import Ellipse
from matplotlib.patches import Ellipse
from matplotlib.colors  import ListedColormap
from matplotlib.patches import Arc
from matplotlib.patches import Circle
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

import astropy.constants as aconst
import astropy.units as u

from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import scipy.spatial
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
import warnings
from numpy.linalg import eig
from numpy.linalg import eigh
import os
import matplotlib.ticker as tck
from scipy.ndimage import gaussian_filter1d

smooth_factor = 15
def smooth(x,window_len=smooth_factor,window='hanning'):

        if x.ndim != 1:
                raise "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise  "Input vector needs to be bigger than window size."
        if window_len<3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise  "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s=np.r_[2*x[0]-x[window_len-1::-1],x,2*x[-1]-x[-1:-window_len:-1]]
        if window == 'flat': #moving average
                w=np.ones(window_len,'d')
        else:  
                w=eval('np.'+window+'(window_len)')
        y=np.convolve(w/w.sum(),s,mode='same')
        return y[window_len:-window_len+1]



path = "/home/jzuluaga/public/outputs/p3diso_f0/"

path0 = "/home/jzuluaga/public/outputs/p3diso_f0/"
path1 = "/home/jzuluaga/public/outputs/p3diso_f0/"

output = 2
# Define the grid dimensions and Phi cut index
variables_par = np.genfromtxt(path+"variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
parametros_par, valores_par = [],[]

for posicion in variables_par:
    parametros_par.append(posicion[0].decode("utf-8"))
    valores_par.append(posicion[1].decode("utf-8"))

def P(parametro):
    return valores_par[parametros_par.index(parametro)]

def get_Parameter(path, parametro):
    """
    Lee un archivo .par y devuelve el valor asociado a un parámetro específico.
    
    Parámetros:
        path (str): Ruta del directorio donde se encuentra el archivo .par.
        parametro (str): Nombre del parámetro para el cual se quiere obtener el valor.
    
    Retorna:
        valor_parametro (str): Valor del parámetro especificado.
    """

    variables_par = np.genfromtxt(path+"variables.par", dtype={'names': ("parametros", "valores"), 'formats': ("|S30", "|S300")}).tolist()
    parametros_par, valores_par = [], []

    for posicion in variables_par:
        parametros_par.append(posicion[0].decode("utf-8"))
        valores_par.append(posicion[1].decode("utf-8"))

    try:
        valor_parametro = valores_par[parametros_par.index(parametro)]
        return valor_parametro
    except ValueError:
        return "Parámetro no encontrado"


#Unidades
G             = 6.67259e-8                                                                                              #cm3/gr/s2
mu            = 2.4                                                                                                             #gr/mol
Rgas          = 8.314472e7                                                                                              #erg/K/mol = gr*cm2/s2/K/mol
sigma_SB      = 5.67051e-5                                                                                              #erg/cm2/K4/s
mproton       = 1.6726231e-24                                                                                   #gr
h_planck      = 6.6260755e-27                                                                                   #erg*s
kb            = 1.380658e-16                                                                                    #erg/K
unit_eV       = 1.60218e-12                                                                                     #erg
unit_length   = (1.0)*1.49598e13                                                                                #       cm
unit_mass     = 1.9891e33                                                                                               #gr


#unit_disktime = float(get_Parameter(path,("NINTERM"))*float(get_Parameter(path,("DT"))*unit_time                    #yr/snapshot      [Las unidades de tiempo del disco se toman en R=1]

R_scale = 1.0
unit_density  = unit_mass/(R_scale*unit_length)**3                                                              #gr / cm3
unit_time     = np.sqrt( pow((R_scale*unit_length),3.) / G / unit_mass)/3.154e7 #yr
unit_disktime = float(get_Parameter(path,("NINTERM")))*float(get_Parameter(path,("DT")))*unit_time                    #yr/snapshot      [Las unidades de tiempo del disco se toman en R=1]
unit_velocity = 1e-5*(unit_length*R_scale)/float(unit_time*3.154e7)                             #km/s
unit_energy   = unit_mass/(unit_time*3.154e7)**2/(R_scale*unit_length)                  #erg/cm3 = gr/s2/cm


#Planet
print("Leyendo Planeta ...")
snapshot = 0
planet  = np.genfromtxt(path+"bigplanet0.dat") 
xp      = planet[snapshot][1] ; yp      = planet[snapshot][2]*R_scale
zp      = planet[snapshot][3] ; mp      = planet[snapshot][7]
rp      = np.sqrt((xp**2)+(yp**2)+(zp**2))

Rhill   = rp*(mp/3)**(1./3)
Rb = 0.2
print(Rb/Rhill)

# Load the data
p = np.genfromtxt(path+'domain_x.dat')# +np.pi/2 # azimuthal coordinate phi
r = np.genfromtxt(path+'domain_y.dat')[3:-3]*R_scale  # radial coordinate r
t = np.genfromtxt(path+'domain_z.dat')[3:-3]  # colatitude theta

r = 0.5*(r[:-1] + r[1:]) # r (radial)
p = 0.5*(p[:-1] + p[1:]) # p (azimuth, phi)
t = 0.5*(t[:-1] + t[1:]) # t (co-latitud, theta)

R_centro = r.reshape(int(float(P("NY"))),1)


# Define the grids with the correct dimensions
T, R, Phi = np.meshgrid(t, r, p, indexing='ij')
x, y, z = R*np.sin(T)*np.cos(Phi) ,  R*np.sin(T)*np.sin(Phi) , R*np.cos(T)

RP    = np.sqrt(((x)**2)+(y-xp)**2+(z)**2)

NX = int(get_Parameter(path, "NX"))
NY = int(get_Parameter(path, "NY"))
NZ = int(get_Parameter(path, "NZ"))

print("NX: ",NX, "NY: ",NY, "NZ: ",NZ)
# Load Fields
density = np.fromfile(path+'gasdens'+str(output)+'.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_density       #gr/cm3
energy = np.fromfile(path+'gasenergy'+str(output)+'.dat').reshape(NZ,NY,NX)*unit_energy      #erg/cm3
temp   = energy/density/(Rgas/mu)*(float(P("GAMMA"))-1) 

##### Compute Ionization degree

#k_rhill = 1.0/4.0
k_rhill = 1.0/3.0
cs      = np.sqrt(temp*float(P("GAMMA"))*Rgas/mu)                                      #cm/s
Mp      = mp*unit_mass                                                                                          #gr
Rbondi  = (G*Mp/(cs**2))/unit_length  
#   1/Reff = 1/Rbondi+1/(k*Rhill)   |----->|   1/Reff = Rbondi+(k*Rhill)/(Rbondi*k*Rhill)   |--->|   Reff = (Rbondi*k*Rhill)/(Rbondi+(k*Rhill))
Reff    = (Rbondi*k_rhill*Rhill)/(Rbondi+(k_rhill*Rhill))
#print('Rbondi: ', Rbondi)

def calculate_Gamma_X(temp):
    print("Calculando Gamma y X ...")
    
    # Convert the input temperature to float128
    temp1 = np.float128(temp)

    # Perform calculations
    exponente     = np.float128(-(13.6 * unit_eV) / (kb * temp1))
    exponencial   = np.float128(np.exp(exponente))
    termino_gamma = np.float128((2 * np.pi * mproton * kb * temp1 / h_planck ** 2) ** (3. / 2))
    Gamma         = np.float128(termino_gamma * exponencial)  # 1/cm**3

    n_numero  = np.float128(1.0)
    negative  = np.float128(-1.0)
    X_termino  = np.float128(negative * Gamma / n_numero)
    X_square_1 = np.float128(Gamma ** 2 / n_numero ** 2)
    X_square_2 = np.float128(4.0 * Gamma / n_numero)
    X_square   = np.float128(np.sqrt(X_square_1 + X_square_2))
    X_final    = (X_termino + X_square) / 2.0
    
    return X_final


X_final_on = calculate_Gamma_X(temp)
X_final_off = calculate_Gamma_X(temp)

#----------------------------------------------------------------------#
#              Calculo R_eff_X y R_eff con Eq.
#----------------------------------------------------------------------#

print("\n")
print("Calculando R eff  with k = ",k_rhill)

def get_r_eff(mapa, condition, output, path):
    # Lectura y procesamiento de coordenadas
    p = np.genfromtxt(path+'domain_x.dat')
    r = np.genfromtxt(path+'domain_y.dat')[3:-3] * R_scale
    t = np.genfromtxt(path+'domain_z.dat')[3:-3]

    r = 0.5 * (r[:-1] + r[1:])
    p = 0.5 * (p[:-1] + p[1:])
    t = 0.5 * (t[:-1] + t[1:])

    p2 = p + np.pi/2
    p2 = 0.5 * (p2[:-1] + p2[1:])

    # Dimensiones de la grilla
    NX = int(get_Parameter(path, "NX"))
    NY = int(get_Parameter(path, "NY"))
    NZ = int(get_Parameter(path, "NZ"))

    # Cálculo de RP2
    T2, R2, Phi2 = np.meshgrid(t, r, p2, indexing='ij')
    x2, y2, z2 = R2 * np.sin(T2) * np.cos(Phi2), R2 * np.sin(T2) * np.sin(Phi2), R2 * np.cos(T2)
    RP2 = np.sqrt(x2**2 + y2**2 + z2**2)

    # Cálculo de r_effs
    r_effs = []
    pos_eff = []

    for i in range(0, NZ):
        for j in range(0, NY):
            for k in range(0, NX):
                if mapa[i][j][k] >= condition:
                    r_effs.append(RP2[i][j][k])
                    pos_eff.append([i, j, k])

    if len(r_effs) > 0:
        return np.max(r_effs)
    else:
        print(f"Array r_effs es vacío en el output {output}.")
        return None


X_condition = 0.5 #1e-6


def calculate_Reff_for_output(output, path, X_condition):
    NX = int(get_Parameter(path, "NX"))
    NY = int(get_Parameter(path, "NY"))
    NZ = int(get_Parameter(path, "NZ"))
    density = np.fromfile(path+'gasdens'+str(output)+'.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_density
    energy = np.fromfile(path+'gasenergy'+str(output)+'.dat').reshape(NZ,NY,NX)*unit_energy      #erg/cm3
    temp_current   = energy/density/(Rgas/mu)*(float(P("GAMMA"))-1)
    X_final_current = calculate_Gamma_X(temp_current)
    Reff_current = get_r_eff(X_final_current, X_condition, output, path)
    return Reff_current


import numpy as np

def calculate_Reff_for_all_outputs(initial, final, path, X_condition):
    Reff_list = []
    output_times = []
    for output in range(initial, final + 1):
        Reff_current = calculate_Reff_for_output(output, path, X_condition)
        if Reff_current is not None:
            Reff_list.append(Reff_current)
            output_times.append(output)

    # Guardar en archivo .txt
    txt_path = os.path.join(path, 'Reff_vs_output_times.txt')
    data_to_save = np.column_stack((output_times, Reff_list))
    np.savetxt(txt_path, data_to_save)

    return output_times, Reff_list

# Uso de la función
#initial = 0  # Establecer el output inicial
#final = 100  # Establecer el output final
#path = "tu/ruta/aqui"  # Establecer la ruta al directorio de datos
#X_condition = 0.5  # Establecer la condición para X

#output_times, ReffON = calculate_Reff_for_all_outputs(initial, final, path1, X_condition)
#output_times, ReffOFF = calculate_Reff_for_all_outputs(initial, final, path0, X_condition)

# Si quieres guardar los resultados en un archivo puedes hacerlo así:
#import numpy as np
#result_data = np.column_stack((output_times, Reff_list))
#np.savetxt("Reff_vs_time.txt", result_data)



def calculate_Reff_and_enclosed_mass(path, initial, final, X_condition, Reff_current=None):
    NX = int(get_Parameter(path, "NX"))
    NY = int(get_Parameter(path, "NY"))
    NZ = int(get_Parameter(path, "NZ"))
    txt_path = os.path.join(path, 'Reff_values.txt')
    enclosed_mass_list = []
    enclosed_gas_energy_list = []
    enclosed_rad_energy_list = []
    
    # Si se proporciona un valor para Reff_current, usarlo directamente y saltar el cálculo
    if Reff_current is not None:
        Reff_array = np.full(final - initial, Reff_current)
    else:
        # Verificar si el archivo ya existe, si es así, simplemente leerlo
        if os.path.exists(txt_path):
            Reff_array = np.loadtxt(txt_path)
        else:
            Reff_list = []
            for output in range(initial, final):
                Reff_current_calculated= calculate_Reff_for_output(output, path, X_condition)
                if Reff_current_calculated is not None:
                    Reff_list.append(Reff_current_calculated)
            Reff_array = np.array(Reff_list)
            np.savetxt(txt_path, Reff_array)
    
    # Utilizar Reff_array para calcular enclosed_mass_list
    for output, Reff_current in enumerate(Reff_array):
        density = np.fromfile(path + 'gasdens' + str(output) + '.dat', dtype=np.float64).reshape(NZ, NY, NX) * unit_density
        gasenergy = np.fromfile(path + 'gasenergy' + str(output) + '.dat', dtype=np.float64).reshape(NZ, NY, NX) * unit_energy
        #radenergy = np.fromfile(path + 'energyrad' + str(output) + '.dat', dtype=np.float64).reshape(NZ, NY, NX) * unit_energy 
        radenergy = np.fromfile(path + 'gasenergy' + str(output) + '.dat', dtype=np.float64).reshape(NZ, NY, NX) * unit_energy

        # Define the grids with the correct dimensions
        p = np.genfromtxt(path+'domain_x.dat')
        r = np.genfromtxt(path+'domain_y.dat')[3:-3] * R_scale
        t = np.genfromtxt(path+'domain_z.dat')[3:-3]
     # Cálculo de dTheta, dPhi y dr
        r = 0.5 * (r[:-1] + r[1:])
        p = 0.5 * (p[:-1] + p[1:])
        t = 0.5 * (t[:-1] + t[1:])

        dTheta = t[1] - t[0]
        dPhi = p[1] - p[0]
        dr = r[1] - r[0]

    # Cálculo de dV

        T, R, Phi = np.meshgrid(t, r, p, indexing='ij')
        x, y, z = R*np.sin(T)*np.cos(Phi) ,  R*np.sin(T)*np.sin(Phi) , R*np.cos(T)

        dV = (R**2 * np.sin(T) * dTheta * dPhi * dr)
        RP    = np.sqrt(((x)**2)+(y-xp)**2+(z)**2)

        mask = RP <= Reff_current
        ### Calculo de la masas encerrada:
        enclosed_mass = np.sum((density[mask] / unit_density) * dV[mask]) #para que quede en solar units
        enclosed_mass_list.append(enclosed_mass)

        enclosed_gas_energy = np.sum((gasenergy[mask]) * dV[mask])  
        enclosed_gas_energy_list.append(enclosed_gas_energy)

        enclosed_rad_energy = np.sum((radenergy[mask]) * dV[mask])
        enclosed_rad_energy_list.append(enclosed_rad_energy)


        #data_to_save = np.column_stack((enclosed_mass_list, enclosed_gas_energy_list))
        #np.savetxt(txt_path, data_to_save)
    
    return np.array(enclosed_mass_list), np.array(enclosed_gas_energy_list), np.array(enclosed_rad_energy_list) , Reff_array

### clean data func


def clean_data(time_array, enclosed_mass_list, n_std_dev=1.5, window_size=3, fallback_to_global_stats=False):
    n = len(enclosed_mass_list)
    filter_mask = np.ones(n, dtype=bool)
    
    global_mean = np.mean(enclosed_mass_list) if fallback_to_global_stats else None
    global_std = np.std(enclosed_mass_list) if fallback_to_global_stats else None

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        
        local_data = enclosed_mass_list[start:end]
        
        mean_value = np.mean(local_data) if len(local_data) > 0 else global_mean
        std_dev_value = np.std(local_data) if len(local_data) > 0 else global_std

        if mean_value is not None and std_dev_value is not None:
            filter_mask[i] = np.abs(enclosed_mass_list[i] - mean_value) <= n_std_dev * std_dev_value
    
    time_array_filtered = time_array[filter_mask]
    enclosed_mass_list_filtered = enclosed_mass_list[filter_mask]
    
    return time_array_filtered, enclosed_mass_list_filtered

## Funcion para las derivadas


def moving_average(data, window_size=5):
    """Aplica una media móvil con una ventana dada."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def calculate_derivative(data_array, time_interval, window_size=5):
    """
    Calcula la derivada de un conjunto de datos utilizando diferencias finitas centradas
    y un filtro suavizante previo.

    Parámetros:
        data_array (numpy.ndarray): Array de datos.
        time_interval (float): Intervalo de tiempo entre cada punto de datos.
        window_size (int): Tamaño de la ventana para el filtro suavizante.

    Retorna:
        derivative_array (numpy.ndarray): Array de derivadas.
        time_array_diff (numpy.ndarray): Array de tiempo ajustado para la derivada.
    """
    
    # Aplicar media móvil para suavizar los datos
    smoothed_data = moving_average(data_array, window_size)
    
    # Calcular la derivada utilizando diferencias finitas centradas
    derivative_array = np.diff(smoothed_data) / time_interval
    
    # Ajustar el array de tiempo
    half_window = window_size // 2
    time_array_diff = np.arange(half_window, len(derivative_array) * time_interval + half_window, time_interval) + 0.5 * time_interval
    
    return derivative_array, time_array_diff


def calculate_derivative_centered(data_array, time_interval):
    """
    Calcula la derivada de un conjunto de datos utilizando diferencias finitas centradas.

    Parámetros:
        data_array (numpy.ndarray): Array de datos.
        time_interval (float): Intervalo de tiempo entre cada punto de datos.

    Retorna:
        derivative_array (numpy.ndarray): Array de derivadas.
        time_array_diff (numpy.ndarray): Array de tiempo ajustado para la derivada.
    """
    
    # Calcular la derivada utilizando diferencias finitas centradas
    derivative_array = (data_array[2:] - data_array[:-2]) / (2 * time_interval)
    
    # Ajustar el array de tiempo para excluir los extremos
    time_array_diff = np.arange(1, len(data_array) - 1) * time_interval
    
    return derivative_array, time_array_diff



def calculate_smoothed_derivative_centered(data_array, time_interval, sigma=1):
    """
    Calcula una derivada suavizada de un conjunto de datos utilizando diferencias finitas centradas.
    
    Parámetros:
        data_array (numpy.ndarray): Array de datos.
        time_interval (float): Intervalo de tiempo entre cada punto de datos.
        sigma (float): Desviación estándar del filtro Gaussiano.
        
    Retorna:
        derivative_array (numpy.ndarray): Array de derivadas suavizadas.
        time_array_diff (numpy.ndarray): Array de tiempo ajustado para la derivada.
    """
    
    # Aplicar suavizado Gaussiano a los datos
    smoothed_data = gaussian_filter1d(data_array, sigma)
    
    # Calcular la derivada utilizando diferencias finitas centradas
    derivative_array = (smoothed_data[2:] - smoothed_data[:-2]) / (2 * time_interval)
    
    # Ajustar el array de tiempo para excluir los extremos
    time_array_diff = np.arange(1, len(data_array) - 1) * time_interval
    
    return derivative_array, time_array_diff


def calculate_higher_order_derivative_centered(data_array, time_interval):
    """
    Calcula la derivada de un conjunto de datos utilizando una aproximación de diferencias finitas de segundo orden.

    Parámetros:
        data_array (numpy.ndarray): Array de datos.
        time_interval (float): Intervalo de tiempo entre cada punto de datos.

    Retorna:
        derivative_array (numpy.ndarray): Array de derivadas.
        time_array_diff (numpy.ndarray): Array de tiempo ajustado para la derivada.
    """

    # Inicializar array de derivadas
    derivative_array = np.zeros(len(data_array) - 2)

    # Calcular la derivada utilizando diferencias finitas de segundo orden
    for i in range(1, len(data_array) - 1):
        derivative_array[i-1] = (-1.5 * data_array[i-1] + 2 * data_array[i] - 0.5 * data_array[i+1]) / time_interval

    # Ajustar el array de tiempo para excluir los extremos
    time_array_diff = np.arange(1, len(data_array) - 1) * time_interval

    return derivative_array, time_array_diff

#######################################

initial = 1
final = 100

# Calcular masa encerrada

mass_off, gas_energy_off, rad_energy_off , Reff_off = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition, Reff_current = 0.3*Rhill)
mass_on, gas_energy_on, rad_energy_on, Reff_on = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition, Reff_current = 0.4*Rhill)


#caso off
mass_off_Ri, gas_energy_off_Ri, rad_energy_off_Ri , Reff_off_Ri = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition, Reff_current = 0.3*Rhill)
mass_off_Rb, gas_energy_off_Rb, rad_energy_off_Rb , Reff_off_Rb = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition, Reff_current = Rb)
mass_off_Rh, gas_energy_off_Rh, rad_energy_off_Rh , Reff_off_Rh = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition, Reff_current = Rhill)

#caso on

mass_on_Ri, gas_energy_on_Ri, rad_energy_on_Ri , Reff_on_Ri = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition, Reff_current = 0.3*Rhill)
mass_on_Rb, gas_energy_on_Rb, rad_energy_on_Rb , Reff_on_Rb = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition, Reff_current = Rb)
mass_on_Rh, gas_energy_on_Rh, rad_energy_on_Rh , Reff_on_Rh = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition, Reff_current = Rhill)



#mass_off, gas_energy_off, rad_energy_off , Reff_off = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition)
#mass_on, gas_energy_on, rad_energy_on, Reff_on = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition)

total_energy_off  = gas_energy_off + rad_energy_off 
total_energy_on  = gas_energy_on + rad_energy_on 

total_energy_on_Ri  = gas_energy_on_Ri + rad_energy_on_Ri 
total_energy_on_Rb  = gas_energy_on_Rb + rad_energy_on_Rb 
total_energy_on_Rh  = gas_energy_on_Rh + rad_energy_on_Rh 


#mass_off, Reff_off = calculate_Reff_and_enclosed_mass(path0, initial, final, X_condition)
#mass_on, Reff_on = calculate_Reff_and_enclosed_mass(path1, initial, final, X_condition)

# Array de tiempo
time_off = unit_disktime * np.linspace(0, final - initial, len(mass_off))  # Ahora Reff_array1 está definido 
time_on = unit_disktime * np.linspace(0, final - initial, len(mass_on))  # Ahora Reff_array1 está definido

time_on_Ri = unit_disktime * np.linspace(0, final - initial, len(mass_on_Ri))
time_on_Rb = unit_disktime * np.linspace(0, final - initial, len(mass_on_Rb))
time_on_Rh = unit_disktime * np.linspace(0, final - initial, len(mass_on_Rh))

### Limpiar los datos
time_cleaned_off, mass_cleaned_off = clean_data(time_off, mass_off, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)
time_cleaned_on, mass_cleaned_on = clean_data(time_on, mass_on, n_std_dev=2.0,window_size=3, fallback_to_global_stats=True )

## test con Rionization, Rbondi, Rhill
time_cleaned_off_Ri, mass_cleaned_off_Ri = clean_data(time_off, mass_off_Ri, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)
time_cleaned_off_Rb, mass_cleaned_off_Rb = clean_data(time_off, mass_off_Rb, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)
time_cleaned_off_Rh, mass_cleaned_off_Rh = clean_data(time_off, mass_off_Rh, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)

time_cleaned_on_Ri, mass_cleaned_on_Ri = clean_data(time_on_Ri, mass_on_Ri, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)
time_cleaned_on_Rh, mass_cleaned_on_Rh = clean_data(time_on_Rh, mass_on_Rh, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)
time_cleaned_on_Rb, mass_cleaned_on_Rb = clean_data(time_on_Rb, mass_on_Rb, n_std_dev=2.0, window_size=9, fallback_to_global_stats=True)


# Calcular las derivadas usando la masa limpia

time_interval = unit_disktime  #
rate_off, time_rate_off = calculate_derivative_centered((mass_cleaned_off), time_interval)
rate_on, time_rate_on = calculate_derivative_centered((mass_cleaned_on), time_interval)

## rate for tests Rion, Rb, Rhill
rate_off_Ri, time_rate_off_Ri = calculate_derivative_centered((mass_cleaned_off_Ri), time_interval)
rate_off_Rb, time_rate_off_Rb = calculate_derivative_centered((mass_cleaned_off_Rb), time_interval)
rate_off_Rh, time_rate_off_Rh = calculate_derivative_centered((mass_cleaned_off_Rh), time_interval)


rate_on_Ri, time_rate_on_Ri = calculate_derivative_centered((mass_cleaned_on_Ri), time_interval)
rate_on_Rh, time_rate_on_Rh = calculate_derivative_centered((mass_cleaned_on_Rh), time_interval)
rate_on_Rb, time_rate_on_Rb = calculate_derivative_centered((mass_cleaned_on_Rb), time_interval)


gas_energy_rate_off, time_energy_off = calculate_derivative_centered(gas_energy_off, time_interval)
rad_energy_rate_off, time_energy_off = calculate_derivative_centered(rad_energy_off, time_interval)
total_energy_rate_off, time_energy_off = calculate_derivative_centered(total_energy_off, time_interval)
total_energy_rate_on, time_energy_on = calculate_derivative_centered(total_energy_on, time_interval)
total_energy_rate_on_Rh, time_energy_on_Rh = calculate_derivative_centered(total_energy_on_Rh, time_interval)

## esta derivada queda ruidosa, pero da el orden de magnitud correcto. 
# limpiemos rate usando la funcion de masa (tienen el mismo trend)
# Paso 1: Interpolación
# Paso 3: Escalar los datos interpolados


def generate_new_rate(time_cleaned, mass_cleaned, time_rate, rate):
    # Paso 1: Interpolación
    interp_mass = interp1d(time_cleaned, mass_cleaned, kind='linear', bounds_error=False, fill_value="extrapolate")
    mass_interpolated = interp_mass(time_rate)
    
    # Paso 2: Calcular el factor alpha
    mu_mass = np.mean(mass_interpolated)
    mu_rate = np.mean(np.abs(rate))
    alpha = mu_rate / mu_mass
    
    # Paso 3: Escalar los datos interpolados
    new_rate = alpha * mass_interpolated
    
    return new_rate



new_rate_on = generate_new_rate(time_cleaned_on, mass_cleaned_on, time_rate_on, rate_on)
new_rate_off = generate_new_rate(time_cleaned_off, mass_cleaned_off, time_rate_off, rate_off)

new_rate_off_Ri = generate_new_rate(time_cleaned_off_Ri, mass_cleaned_off_Ri, time_rate_off_Ri, rate_off_Ri)
new_rate_off_Rb = generate_new_rate(time_cleaned_off_Rb, mass_cleaned_off_Rb, time_rate_off_Rb, rate_off_Rb)
new_rate_off_Rh = generate_new_rate(time_cleaned_off_Rh, mass_cleaned_off_Rh, time_rate_off_Rh, rate_off_Rh)


new_rate_on_Ri = generate_new_rate(time_cleaned_on_Ri, mass_cleaned_on_Ri, time_rate_on_Ri, rate_on_Ri)
new_rate_on_Rh = generate_new_rate(time_cleaned_on_Rh, mass_cleaned_on_Rh, time_rate_on_Rh, rate_on_Rh)
new_rate_on_Rb = generate_new_rate(time_cleaned_on_Rb, mass_cleaned_on_Rb, time_rate_on_Rb, rate_on_Rb)

total_energy_rate_off = generate_new_rate(time_off, total_energy_off, time_energy_off, total_energy_rate_off)
total_energy_rate_on = generate_new_rate(time_on, total_energy_on, time_energy_on, total_energy_rate_on)
total_energy_rate_on_Rh = generate_new_rate(time_on, total_energy_on_Rh, time_energy_on_Rh, total_energy_rate_on_Rh)



# Identificando índices donde el tiempo es mayor a 300 órbitas
def calculate_gain_factor(time_on, new_on, time_off, new_off, t):
    # Identify indices where time exceeds threshold for both cases
    indices_on = [i for i, time in enumerate(time_on) if time > t]
    indices_off = [i for i, time in enumerate(time_off) if time > t]
    
    # Compute mean accretion rates
    mean_on = np.mean([new_on[i] for i in indices_on])
    mean_off = np.mean([new_off[i] for i in indices_off])
    
    # Calculate the factor
    factor = (np.abs(mean_on - mean_off)) / mean_off
    
    return factor, mean_on, mean_off


# Calculando los incrementos:

gain_on_MdotRh_to_MdotRi, mean_Mdot_on_Rh, mean_Mdot_on_Ri = calculate_gain_factor(time_rate_on_Rh, new_rate_on_Rh, time_rate_on, new_rate_on, 400)
gain_off_MdotRh_to_MdotRi, mean_Mdot_off_Rh, mean_Mdot_off_Ri = calculate_gain_factor(time_rate_off_Rh, new_rate_off_Rh, time_rate_off, new_rate_off, 400)

accretion_gain, mean_acc_on, mean_acc_off = calculate_gain_factor(time_rate_on, new_rate_on, time_rate_off, new_rate_off, 400)
lum_gain, mean_L_on, mean_L_off = calculate_gain_factor(time_energy_on, total_energy_rate_on, time_energy_off, total_energy_rate_off, 400)

print(f"mean_Mdot_Rh_on: {mean_Mdot_on_Rh:.2e}")
print(f"mean_Mdot_Rh_off: {mean_Mdot_off_Rh:.2e}")


print(f"mean_Mdot_ri_on: {mean_acc_on:.2e}")
print(f"mean_Mdot_ri_off: {mean_acc_off:.2e}")

print(f"mean_L_on: {mean_L_on:.2e}")
print(f"mean_L_off: {mean_L_off:.2e}")

print(f"Accretion gain: {accretion_gain:.2e}")
print(f"Luminosity gain: {lum_gain:.2e}")


###############
## Fig Reff
# Crear el gráfico
plt.figure()

# Añadir las series al gráfico
plt.plot(Reff_off, label='Reff_off')
plt.plot(Reff_on, label='Reff_on')

# Añadir etiquetas y título
plt.xlabel('Índice o Tiempo')
plt.ylabel('Reff')
plt.title('Comparación de Reff_on y Reff_off')

# Añadir leyenda
plt.legend()


### calcular correlacion entre Mdot y L

# Radio de Júpiter en metros
r_jupiter_m = aconst.R_jup
r_jupiter_au = r_jupiter_m.to(u.au) #0.3*Rhill*u.au #r_jupiter_m.to(u.au)  #0.3*Rhill*u.au
rate_unit = u.M_sun/u.yr
Rout = Rhill*u.au
Rin = (R_centro[1]-R_centro[0])*u.au#  (Rhill/9)*u.au
print(Rin*u.au, (Rhill/9)*u.au)
Lum_factor = (aconst.G*mp*aconst.M_sun)*rate_unit*(1./Rin - 1./Rout)

Lacc_on = (new_rate_on * Lum_factor).to(u.L_sun)
Lacc_off = (new_rate_off * Lum_factor).to(u.L_sun)

### calculos de Lacc para otros radios
Lacc_off_Ri = (new_rate_off * Lum_factor).to(u.L_sun)
Lacc_off_Rb = (new_rate_off_Rb * Lum_factor).to(u.L_sun)
Lacc_off_Rh = (new_rate_off_Rh * Lum_factor).to(u.L_sun)

Lacc_on_Ri = (new_rate_on_Ri * Lum_factor).to(u.L_sun)
Lacc_on_Rh = (new_rate_on_Rh * Lum_factor).to(u.L_sun)
Lacc_on_Rb = (new_rate_on_Rb * Lum_factor).to(u.L_sun)
print("")


lum_Ri_Rh_gain, mean_L_Rh, mean_L_Ri = calculate_gain_factor(time_rate_off_Rh, Lacc_off_Rh.value, time_rate_off_Ri, Lacc_off_Ri.value, 400)

lum_Rh_On2off_gain, mean_L_Rh_on, mean_L_Rh_on = calculate_gain_factor(time_rate_off_Rh, Lacc_on_Rh.value, time_rate_on_Rh, Lacc_on_Rh.value, 400)




print(f"mean Lacc Ri off: {mean_L_Ri:.2e}")
print(f"mean Lacc Rh off: {mean_L_Rh:.2e}")
print(f"Luminosity gain Ri to Rh (caso off): {lum_Ri_Rh_gain:.2e}")
print(f"Luminosity gain Rh (off to on): {lum_Rh_On2off_gain:.2e}")



print("Accretion Luminosity on (classic)", mean_acc_on*(Lum_factor).to(u.L_sun))
print("Accretion Luminosity off (classic)", mean_acc_off*(Lum_factor).to(u.L_sun))

print("Luminosity on in erg", mean_acc_on*(Lum_factor).to(u.erg/u.s))
print("0.5e27 erg/s = ", (0.5e27*(u.erg/u.s)).to(u.L_sun))


#################################
### Plot de la energia / luminosidad
#Figura
bottom = 0.12
left   = 0.17
top    = 1.-0.08
right  = 1.-0.04
figwidth = 7.20
figheight = 5.7



fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje x
ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje y
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax.tick_params(axis="both",bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False,labelleft=True, labelright=False)      #Habilita ticks en los 4 lados
ax.tick_params(axis="both",which="minor",direction="in",length=2,width=1.5,color="black") #Ajuste de ticks menores
ax.tick_params(axis="both",which="major",direction="in",length=3,width=1.5,color="black",labelsize=15) #Ajuste de ticks mayores
#ax.set_yscale('log')


# ON
ax.semilogy(time_rate_on_Rh, new_rate_on_Rh,"lightcoral",lw=3.0, label=r"$\dot{M}(R_{\rm Hill})$ (feedback $on$)", alpha = 0.6)
#ax.semilogy(time_rate_on_Rh, smooth(new_rate_on_Rh),"lightcoral",lw=3.0, alpha = 0.6)
## OFF
ax.semilogy(time_rate_off_Rh, new_rate_off_Rh, "lightsteelblue", label=r"$\dot{M}(R_{\rm Hill})$ (feedback $off$)" ,lw=3.0, alpha=0.6)
#ax.semilogy(time_rate_off_Rh, smooth(new_rate_off_Rh), "lightsteelblue",lw=3.0, alpha=0.6)


plt.xlabel("Time [orbits]",fontsize=15)
plt.ylabel(r"Planet accretion rate [$M_{\odot}  yr^{-1}$]",fontsize=15)
plt.legend(loc=1,fontsize=12,frameon=True)

plt.legend()




#################################
### Plot de la energia / luminosidad
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje x
ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje y
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax.tick_params(axis="both",bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False,labelleft=True, labelright=False)      #Habilita ticks en los 4 lados
ax.tick_params(axis="both",which="minor",direction="in",length=2,width=1.5,color="black") #Ajuste de ticks menores
ax.tick_params(axis="both",which="major",direction="in",length=3,width=1.5,color="black",labelsize=15) #Ajuste de ticks mayores
#ax.set_yscale('log')


ax.semilogy(time_rate_on, Lacc_on, "lightsteelblue", label=r"$L_{\rm acc}$" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on, Lacc_on_Rh, "blue", label=r"$L_{\rm acc} en Rhill$" ,lw=3.0, alpha=0.6)
## OFF

ax.semilogy(time_energy_on, total_energy_rate_on,"lightcoral",lw=3.0, label=r"$L_{\rm env}$", alpha = 0.6)
ax.semilogy(time_energy_on_Rh, total_energy_rate_on_Rh,"red",lw=3.0, label=r"$L_{\rm env}$ en Rhill", alpha = 0.6)

#plt.semilogy(time_off, rad_energy_rate_off,"lightcoral",lw=3.0,label=r"Rad Energy envelope $\it{off}$", alpha = 0.6)
#plt.semilogy(time_off, rad_energy_rate_off,"lightcoral",lw=3.0)

# ON
#ax.semilogy(time_rate_on, Lacc_on, "lightsteelblue", label=r"Lacc $\it{on}$" ,lw=3.0, alpha=0.6)
#ax.semilogy(time_rate_off, smooth(new_rate_off), "lightsteelblue" ,lw=3.0)

plt.xlabel("Time [orbits]",fontsize=15)
plt.ylabel(r"Luminosity [$L_{sun}$]",fontsize=15)
plt.legend(loc=1,fontsize=12,frameon=True)
plt.title("Feedback on case")
plt.legend()
#plt.show()
############

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(figwidth, figheight))
plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje x
ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje y
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax.tick_params(axis="both",bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False,labelleft=True, labelright=False)      #Habilita ticks en los 4 lados
ax.tick_params(axis="both",which="minor",direction="in",length=2,width=1.5,color="black") #Ajuste de ticks menores
ax.tick_params(axis="both",which="major",direction="in",length=3,width=1.5,color="black",labelsize=15) #Ajuste de ticks mayores
#ax.set_yscale('log')

ax.semilogy(time_rate_off_Ri, new_rate_off_Ri, lw=3.0, label=r"Ri", alpha=0.6)
ax.semilogy(time_rate_off_Ri, smooth(new_rate_off_Ri), lw=3.0)

ax.semilogy(time_rate_off_Rb, new_rate_off_Rb, lw=3.0, label=r"Rb", alpha=0.6)
ax.semilogy(time_rate_off_Rb, smooth(new_rate_off_Rb), lw=3.0)

ax.semilogy(time_rate_off_Rh, new_rate_off_Rh, lw=3.0, label=r"Rh", alpha=0.6)
ax.semilogy(time_rate_off_Rh, smooth(new_rate_off_Rh), lw=3.0)

plt.xlabel("Time [orbits]",fontsize=15)
plt.ylabel(r"Planet accretion rate [$M_{\odot}  yr^{-1}$]",fontsize=15)
plt.legend(loc=1,fontsize=12,frameon=True)

plt.legend()
#plt.show()


#################### Plot Mdot y Lacc para distintos radios

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Configuración común para ambos subgráficos
for ax in axs:
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
    ax.tick_params(axis="both", bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=1.5, color="black")
    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1.5, color="black", labelsize=15)

### Primer subgráfico
ax = axs[0]

#OFF
ax.semilogy(time_rate_off_Ri, new_rate_off_Ri, linestyle = '-',color="steelblue", label=r"$R_{\rm ion}$ (fdbk off)" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Ri, new_rate_on_Ri, linestyle = '-',color="Firebrick", label=r"$R_{\rm ion}$ (fdbk on)" ,lw=3.0, alpha=0.6)

ax.semilogy(time_rate_off_Rh, new_rate_off_Rh, linestyle='-.', color="dodgerblue", label=r"$R_{\rm Hill}$ (fdbk off)", lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Rh, new_rate_on_Rh, linestyle='-.', color="tomato", label=r"$R_{\rm Hill}$ (fdbk on)", lw=3.0, alpha=0.6)

ax.semilogy(time_rate_off_Rb, new_rate_off_Rb,linestyle = ':' ,color="deepskyblue", label=r"$R_{\rm bound}$ (fdbk off)" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Rb, new_rate_on_Rb,linestyle = ':' ,color="indianred", label=r"$R_{\rm bound}$ (fdbk on)" ,lw=3.0, alpha=0.6)


ax.set_xlabel("Time [orbits]", fontsize=15)
ax.set_ylabel(r'Planet accretion rate [$M_{\odot}  yr^{-1}$]', fontsize=15)


axs[0].legend(loc=1, fontsize=9, frameon=True)
### Segundo subgráfico
ax = axs[1]

#OFF
ax.semilogy(time_rate_off_Ri, Lacc_off_Ri, color="steelblue",linestyle = '-' ,label=r"$R_{\rm ion}$ (fdbk off)" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Ri, Lacc_on_Ri, color="Firebrick",linestyle = '-' ,label=r"$R_{\rm ion}$ (fdbk on)" ,lw=3.0, alpha=0.6)

ax.semilogy(time_rate_off_Rh, Lacc_off_Rh, color="dodgerblue",linestyle = '-.' ,label=r"$R_{\rm Hill}$ (fdbk off)" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Rh, Lacc_on_Rh, color="tomato",linestyle = '-.' ,label=r"$R_{\rm Hill}$ (fdbk on)" ,lw=3.0, alpha=0.6)

ax.semilogy(time_rate_off_Rb, Lacc_off_Rb, color="deepskyblue",linestyle = ':' ,label=r"$R_{\rm bound}$ (fdbk off)" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_on_Rb, Lacc_on_Rb, color="indianred",linestyle = ':' ,label=r"$R_{\rm bound}$ (fdbk on)" ,lw=3.0, alpha=0.6)


ax.set_xlabel("Time [orbits]", fontsize=15)
ax.set_ylabel(r"$\rm L_{acc}$ [$L_{\odot}$]", fontsize=15)


# Leyendas
axs[1].legend(loc=1, fontsize=9, frameon=True)

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc=1, fontsize=12, frameon=True)


#plt.show()


################ 



#############################
# Grafica los resultados

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

# Configuración común para ambos subgráficos
for ax in axs:
    ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
    ax.tick_params(axis="both", bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax.tick_params(axis="both", which="minor", direction="in", length=2, width=1.5, color="black")
    ax.tick_params(axis="both", which="major", direction="in", length=3, width=1.5, color="black", labelsize=15)

### Primer subgráfico
ax = axs[0]

#ON
ax.semilogy(time_rate_on, new_rate_on, "lightcoral", lw=3.0, label=r"feedback $\it{on}$", alpha=0.6)
ax.semilogy(time_rate_on, smooth(new_rate_on), "lightcoral", lw=3.0)

#OFF
ax.semilogy(time_rate_off, new_rate_off, "lightsteelblue", label=r"feedback $\it{off}$" ,lw=3.0, alpha=0.6)
ax.semilogy(time_rate_off, smooth(new_rate_off), "lightsteelblue" ,lw=3.0)

ax.set_xlabel("Time [orbits]", fontsize=15)
ax.set_ylabel(r'Planet accretion rate [$M_{\odot}  yr^{-1}$]', fontsize=15)


axs[0].legend(loc=1, fontsize=12, frameon=True)
### Segundo subgráfico
ax = axs[1]

#ON
ax.semilogy(time_energy_on, total_energy_rate_on,"lightcoral",lw=3.0, label=r"feedback $\it{on}$", alpha = 0.6)
ax.semilogy(time_energy_on, smooth(total_energy_rate_on),"lightcoral",lw=3.0)
#OFF
ax.semilogy(time_energy_off, total_energy_rate_off,"lightsteelblue",lw=3.0, label=r"feedback $\it{off}$", alpha = 0.6)
ax.semilogy(time_energy_off, smooth(total_energy_rate_off),"lightsteelblue",lw=3.0)

ax.set_xlabel("Time [orbits]", fontsize=15)
ax.set_ylabel(r"$\rm L_{envelope}$ [$L_{sun}$]", fontsize=15)


# Leyendas
axs[1].legend(loc=1, fontsize=12, frameon=True)

#handles, labels = ax.get_legend_handles_labels()
#ax.legend(handles, labels, loc=1, fontsize=12, frameon=True)


#plt.show()


################ 
# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(9,4))

ax.xaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje x
ax.yaxis.set_minor_locator(tck.AutoMinorLocator(2))             #Agrega 2 ticks pequenos entre los ticks eje y
ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax.tick_params(axis="both",bottom=True, top=True, left=True, right=True,labelbottom=True, labeltop=False,labelleft=True, labelright=False)      #Habilita ticks en los 4 lados
ax.tick_params(axis="both",which="minor",direction="in",length=2,width=1.5,color="black") #Ajuste de ticks menores
ax.tick_params(axis="both",which="major",direction="in",length=3,width=1.5,color="black",labelsize=15) #Ajuste de ticks mayores
# Graficar datos originales

#ax.semilogy(time_rate_off, np.abs(rate_off), "lightsteelblue", lw=3.0, label=r"feedback $\it{off}$ grid: 256 × 128 × 64", alpha=0.6)
#ax.semilogy(time_rate_off, smooth(np.abs(rate_off)), "lightsteelblue", lw=3.0)
#ax.semilogy(time_rate_on, np.abs(rate_on), "lightcoral", lw=3.0, label=r"feedback $\it{off}$ grid: 256 x 128 x 32", alpha=0.6)
#ax.semilogy(time_rate_on, smooth(np.abs(rate_on)), "lightcoral", lw=3.0)

#ax.semilogy(time_rate_off, new_rate_off, "dodgerblue", lw=3.0, label=r"feedback $\it{on}$ lowres grid: 256 x 64 x 64", alpha=0.6)
#ax.semilogy(time_cleaned_off, smooth(mass_cleaned_off), "red", lw=3.0)
#ax.semilogy(time_rate_on, new_rate_on, "darkorange", lw=3.0, label=r"feedback $\it{on}$ highres grid: 256 x 128 x 128", alpha=0.6)
#ax.semilogy(time_cleaned_on, smooth(mass_cleaned_on), "blue", lw=3.0)


ax.semilogy(time_rate_off, new_rate_off, "lightsteelblue", lw=3.0, label=r"feedback $\it{off}$", alpha=0.6)
ax.semilogy(time_rate_off, smooth(new_rate_off), "lightsteelblue", lw=3.0)
ax.semilogy(time_rate_on, new_rate_on, "lightcoral", lw=3.0, label=r"feedback $\it{on}$", alpha=0.6)
ax.semilogy(time_rate_on, smooth(new_rate_on), "lightcoral", lw=3.0)

ax.set_xlabel('Time [orbits]')
ax.set_ylabel(r'Planet accretion rate [$M_{\odot}  yr^{-1}$]')

# Añadir etiquetas y leyenda
#ax.set_xlabel('Time [yr]')
#ax.set_ylabel(r'Planet accretion rate [$M_{\odot} ~ yr^{-1}$]')
ax.legend()
############


####################
# Crear una figura y un conjunto de subtramas
fig, axs = plt.subplots(2, 1,figsize=(8, 10))

# Configuración de la primera subtrama (masa encerrada)
ax1 = axs[0]
ax1.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
ax1.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
ax1.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax1.tick_params(axis="both", bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax1.tick_params(axis="both", which="minor", direction="in", length=2, width=1.5, color="black")
ax1.tick_params(axis="both", which="major", direction="in", length=3, width=1.5, color="black", labelsize=15)

ax1.semilogy(time_cleaned_off, mass_cleaned_off, "lightsteelblue", lw=3.0, label=r"feedback $\it{off}$", alpha=0.6)
ax1.semilogy(time_cleaned_off, smooth(mass_cleaned_off), "lightsteelblue", lw=3.0)
ax1.semilogy(time_cleaned_on, mass_cleaned_on, "lightcoral", lw=3.0, label=r"feedback $\it{on}$", alpha=0.6)
ax1.semilogy(time_cleaned_on, smooth(mass_cleaned_on), "lightcoral", lw=3.0)

#ax1.set_title('Limpieza de Datos')
ax1.set_xlabel('Time [orbits]')
ax1.set_ylabel(r'Envelope mass [$M_{\odot}$]')
ax1.legend()

# Configuración de la segunda subtrama (tasa de cambio de la masa)
ax2 = axs[1]
ax2.xaxis.set_minor_locator(tck.AutoMinorLocator(2))
ax2.yaxis.set_minor_locator(tck.AutoMinorLocator(2))
ax2.yaxis.set_major_formatter(tck.FormatStrFormatter('%.1e'))
ax2.tick_params(axis="both", bottom=True, top=True, left=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax2.tick_params(axis="both", which="minor", direction="in", length=2, width=1.5, color="black")
ax2.tick_params(axis="both", which="major", direction="in", length=3, width=1.5, color="black", labelsize=15)

ax2.semilogy(time_rate_off, np.abs(rate_off), "lightsteelblue", lw=3.0, label=r"feedback $\it{off}$", alpha=0.6)
ax2.semilogy(time_rate_off, smooth(np.abs(rate_off)), "lightsteelblue", lw=3.0)
ax2.semilogy(time_rate_on, np.abs(rate_on), "lightcoral", lw=3.0, label=r"feedback $\it{on}$", alpha=0.6)
ax2.semilogy(time_rate_on, smooth(np.abs(rate_on)), "lightcoral", lw=3.0)

# Añadir etiquetas y leyenda
ax2.set_xlabel('Time [orbits]')
ax2.set_ylabel(r'Planet accretion rate [$M_{\odot} ~ yr^{-1}$]')
ax2.legend()

# Ajuste del diseño
#plt.tight_layout()
####################
plt.show()


