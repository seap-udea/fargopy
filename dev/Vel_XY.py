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

from scipy.optimize import curve_fit
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import griddata
from scipy.interpolate import CubicSpline
from scipy.ndimage import zoom

def smooth(x, window_len=9, window='hanning'):
    """
    Suaviza la señal 1D x usando una ventana de tipo y tamaño especificados.
    Parámetros:
    x : array-like
        La señal a suavizar.
    window_len : int
        El tamaño de la ventana de suavizado.
    window : str
        El tipo de ventana. Debe ser uno de 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
    Devuelve:
    y : array-like
        La señal suavizada.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if x.ndim != 1:
        raise ValueError("Input array must be 1D")
    if x.size < window_len:
        raise ValueError("Input array needs to be larger than window size")
    if window_len < 3:
        return x
    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Invalid window type")
    s = np.concatenate([2*x[0] - x[window_len-1::-1], x, 2*x[-1] - x[-1:-window_len:-1]])
    if window == 'flat':
        w = np.ones(window_len)
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len:-window_len+1]

path = "/home/jzuluaga/public/outputs/p3diso_f0/"
# Define the grid dimensions and Phi cut index
variables_par = np.genfromtxt(path+"variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
parametros_par, valores_par = [],[]

for posicion in variables_par:
    parametros_par.append(posicion[0].decode("utf-8"))
    valores_par.append(posicion[1].decode("utf-8"))

def P(parametro):
    return valores_par[parametros_par.index(parametro)]


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

R_scale = 1.0
unit_density  = unit_mass/(R_scale*unit_length)**3                                                              #gr / cm3
unit_time     = np.sqrt( pow((R_scale*unit_length),3.) / G / unit_mass)/3.154e7 #yr
unit_disktime = float(P("NINTERM"))*float(P("DT"))*unit_time                    #yr/snapshot      [Las unidades de tiempo del disco se toman en R=1]
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
#print(Rhill, rp, xp, yp, zp)

# Domains
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))
print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)

# Load the data
p = np.genfromtxt(path+'domain_x.dat')# +np.pi/2 # azimuthal coordinate phi
r = np.genfromtxt(path+'domain_y.dat')[3:-3]*R_scale  # radial coordinate r
t = np.genfromtxt(path+'domain_z.dat')[3:-3]  # colatitude theta

r = 0.5*(r[:-1] + r[1:]) # r (radial)
p = 0.5*(p[:-1] + p[1:]) # p (azimuth, phi)
t = 0.5*(t[:-1] + t[1:]) # t (co-latitud, theta)

# Define the grids with the correct dimensions
T, R, Phi = np.meshgrid(t, r, p, indexing='ij')
x, y, z = R*np.sin(T)*np.cos(Phi) ,  R*np.sin(T)*np.sin(Phi) , R*np.cos(T)

RP    = np.sqrt(((x)**2)+(y-xp)**2+(z)**2)


# Load Fields
density = np.fromfile(path+'gasdens2.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_density       #gr/cm3
energy = np.fromfile(path+"gasenergy2.dat").reshape(NZ,NY,NX)*unit_energy      #erg/cm3
temp   = energy/density/(Rgas/mu)*(float(P("GAMMA"))-1) 
temp2 = np.copy(temp)
temp3 = np.copy(temp)
temp4 = np.copy(temp)

##### Compute Ionization degree

#k_rhill = 1.0/4.0
k_rhill = 1.0/3.0
cs      = np.sqrt(temp4*float(P("GAMMA"))*Rgas/mu)                                      #cm/s
Mp      = mp*unit_mass                                                                                          #gr
Rbondi  = (G*Mp/(cs**2))/unit_length  
#   1/Reff = 1/Rbondi+1/(k*Rhill)   |----->|   1/Reff = Rbondi+(k*Rhill)/(Rbondi*k*Rhill)   |--->|   Reff = (Rbondi*k*Rhill)/(Rbondi+(k*Rhill))
Reff    = (Rbondi*k_rhill*Rhill)/(Rbondi+(k_rhill*Rhill))
#print('Reff: ', Reff)

print("Calculando Gamma y X ...")
exponente     = np.float128(-(13.6*unit_eV)/(kb*temp))
exponencial   = np.float128(np.exp(exponente))
termino_gamma = np.float128((2*np.pi*mproton*kb*temp3/h_planck**2)**(3/2))
Gamma         = np.float128(termino_gamma*exponencial)                                  #1/cm**3

Gamma2 = np.float128(np.copy(Gamma))
Gamma3 = np.float128(np.copy(Gamma))
Gamma4 = np.float128(np.copy(Gamma))

n_numero  = np.float128(1.0)
negative  = np.float128(-1.0)

X_temrino  = np.float128(negative*Gamma2/n_numero)
X_square_1 = np.float128(Gamma3**2/n_numero**2) 
X_square_2 = np.float128(4.0*Gamma4/n_numero)

X_square   = np.float128(np.sqrt(X_square_1+X_square_2))
X_final    = (X_temrino+X_square)/2.0

print(X_final.max())
print('Max Temp: ', temp.max())
print('Min Tem: ', temp.min())
#----------------------------------------------------------------------#
#              Calculo R_eff_X y R_eff con Eq.
#----------------------------------------------------------------------#

print("\n")
print("Calculando R eff  with k = ",k_rhill)
def get_r_eff(mapa,mapa_radios,condition):
        r_effs = []
        pos_eff = []
            
        for i in range(0,NZ):
                for j in range(0,NY):
                        for k in range(0,NX):
                                if mapa[i][j][k] >= condition:
                                        r_effs.append(mapa_radios[i][j][k])
                                        pos_eff.append([i,j,k])
                                        #print("r: ",mapa_radios[i][j][k]," au","X: ",mapa[i][j][k],"indices" ,i, j, k)
        
        idx = r_effs.index(np.max(r_effs))
        print(idx, pos_eff[idx], r_effs[idx])
        #Toma el maximo valor de r effs
        return np.max(r_effs)

X_condition = 0.5 #1e-6
#Phi2 = Phi - np.pi/2  # Restar pi/2 para cancelar el cambio
# Define the grids with the correct dimensions
p2 = np.genfromtxt(path+'domain_x.dat') +np.pi/2 # azimuthal coordinate phi
p2 = 0.5*(p2[:-1] + p2[1:]) # p (azimuth, phi)
T2, R2, Phi2 = np.meshgrid(t, r, p2 , indexing='ij')
x2, y2, z2 = R*np.sin(T2)*np.cos(Phi2) ,  R*np.sin(T2)*np.sin(Phi2) , R*np.cos(T2)
RP2    = np.sqrt(((x2)**2)+(y2-xp)**2+(z2)**2)
#print(RP/RP2)

R_eff_X     =  get_r_eff(X_final,RP2,X_condition)

print('Ionization radius: ', R_eff_X)
print("R_eff_X:",round(R_eff_X,5)," au","for X>=",X_condition)
print("R_eff_X / Rhill:", round(R_eff_X,5)/Rhill)
print('')

#####

# Load the velocity data from the files
vx = np.fromfile(path+'gasvx2.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_velocity
vy = np.fromfile(path+'gasvy2.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_velocity
vz = np.fromfile(path+'gasvz2.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_velocity

# Calculate the spherical-to-Cartesian transformation for the velocity components
v_x = (vx * np.sin(T) * np.cos(Phi) + vy * np.sin(T) * np.sin(Phi) + vz * np.cos(T)) #km/s
v_y = (vx * np.cos(T) * np.cos(Phi) + vy * np.cos(T) * np.sin(Phi) - vz * np.sin(T)) #km/s
v_z = (-vx * np.sin(Phi) + vy * np.cos(Phi)) #km/s  ## REVISAR vz=((np.cos(T)*vr)-(np.sin(T)*vt))#v_z=((np.cos(T)*vy)-(np.sin(T)*vz))


######
z_slice_index = NZ // 2  # Middle of the phi-axis

top = int( NZ // 1)- 1  # top of the phi-axis
mid = NZ // 2           # Middle of the phi-axis
zslice_index = mid

# data slice
density_cut = (density[zslice_index, :, :])
temperature_cut = temp[zslice_index, :, :]
X_final_cut = X_final[top, :, :]

# Cut the Y, Z, and v_z arrays along the phi axis at phi = 0
X_cut = x[zslice_index, :, :]  #NZ,NY,NX
Y_cut = y[zslice_index, :, :]
Z_cut = z[zslice_index, :, :]

v_x_cut = v_y[zslice_index, :, :]
v_y_cut = v_y[zslice_index, :, :]
v_z_cut = v_z[zslice_index, :, :]

field1 = X_final_cut
print("Temperature_cut MAX:"  , field1.max())
# Plot the data in Cartesian coordinates
plt.figure(figsize=(10, 10))
plt.pcolormesh(X_cut, Y_cut, np.log10(field1), cmap='hot')
plt.colorbar(label='Temperature')
plt.title(f'Gas Density Visualization (Phi-slice at index {zslice_index})')
plt.xlabel(r'Radial distance [au]')
plt.ylabel('X')
plt.axis('equal')
plt.show()

plt.figure(figsize=(10, 10))
plt.pcolormesh(X_cut, Y_cut, np.log10(field1), cmap='hot',shading='gouraud')
plt.colorbar(label='Temperature')
plt.scatter([xp], [yp], c='red', marker='x')  # Dibuja el punto (xp, yp)
plt.title(f'Gas Density Visualization')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()



plt.figure(figsize=(10, 10))
#plt.pcolormesh(Y, Z, v_z_cut, cmap='hot')
# Plot the grid lines
#for i in range(X_cut.shape[0]):
#    plt.plot(X_cut[i, :], Z_cut[i, :], color='k', linewidth=0.5)

#for j in range(X_cut.shape[1]):
#    plt.plot(X_cut[:, j], Z_cut[:, j], color='k', linewidth=0.5)


# Calculate the magnitude of the velocity
v_magnitude = np.sqrt(v_x_cut**2 + v_y_cut**2 + v_y_cut**2)

# Parameters for quiver
scale = 30  # Adjust as needed. This scales the arrows. Larger scale means smaller arrows.
subsample = 2  # Only plot one arrow every 'subsample' points
alpha = 1.0  # Adjust as needed (0 is fully transparent, 1 is fully opaque)
clim = [v_magnitude.min(), np.percentile(v_magnitude, 95)]  # Adjust as needed. This controls the color limits for the arrow colors.
linewidths = 1  # Adjust as needed. This controls the thickness of the arrows.
headwidth = 6  # Adjust as needed. This controls the head width of the arrows.
headlength = 10  # Adjust as needed. This controls the head length of the arrows.
minshaft = 0.01  # Adjust as needed. This controls the minimum length that an arrow must have to get a triangle at the end.
minlength = 0.01  # Adjust as needed. This controls the minimum length that an arrow must have to be drawn.


show_quiver = True  # Cambia esto a False para desactivar quiver
# Add velocity arrows with color representing the magnitude
if show_quiver:
    quiver_plot = plt.quiver(X_cut[::subsample, ::subsample], Y_cut[::subsample, ::subsample], 
               v_x_cut[::subsample, ::subsample], v_y_cut[::subsample, ::subsample], 
               v_magnitude[::subsample, ::subsample], angles='xy', scale_units='xy', scale=scale, 
               cmap='viridis', alpha=alpha, clim=clim, linewidths=linewidths,
               headwidth=headwidth, headlength=headlength, minshaft=minshaft, minlength=minlength)
    # Crear la segunda barra de colores
    cbar2 = plt.colorbar(quiver_plot)
    cbar2.ax.set_ylabel('Velocity Magnitude', rotation=270)

plt.title(f'Grid Visualization (z-slice at index {zslice_index})')
plt.xlabel('Y')
plt.ylabel('Z')
plt.show()

print('Rhill ', Rhill)

# Reescala los ejes Y y Z a unidades de Rh
X_scaled = (X_cut - rp) / Rhill 
Y_scaled = (Y_cut ) / Rhill
Z_scaled = Z_cut / Rhill
Y_min = 1.0
Y_max = 1.1
delta = Y_max - Y_min

# Crea una nueva figura
fig, ax = plt.subplots(figsize=(10, 10))

# Variable para controlar la visualización de la grilla
show_grid = False  # Cambia esto a False para ocultar la grilla

# Condición para añadir grilla
if show_grid:
    for i in range(X_scaled.shape[0]):
        plt.plot(X_scaled[i, :], Y_scaled[i, :], color='black', linewidth=1, linestyle='-', antialiased=True, alpha=0.5)
        
    for j in range(X_scaled.shape[1]):
        plt.plot(X_scaled[:, j], Y_scaled[:, j], color='black', linewidth=1, linestyle='-', antialiased=True, alpha=0.5)


# Parameters for quiver
scale = 10  # Adjust as needed. This scales the arrows. Larger scale means smaller arrows.
subsample = 1  # Only plot one arrow every 'subsample' points
alpha = 1.0  # Adjust as needed (0 is fully transparent, 1 is fully opaque)
clim = [v_magnitude.min(), np.percentile(v_magnitude, 90)]  # Adjust as needed. This controls the color limits for the arrow colors.
linewidths = 1  # Adjust as needed. This controls the thickness of the arrows.
headwidth = 5  # Adjust as needed. This controls the head width of the arrows.
headlength = 10  # Adjust as needed. This controls the head length of the arrows.
minshaft = 1e-10  # Adjust as needed. This controls the minimum length that an arrow must have to get a triangle at the end.
minlength = 1e-10 # Adjust as needed. This controls the minimum length that an arrow must have to be drawn.



# data slice
#density_cut = (density[zslice_index, :, :])
#temperature_cut = temp[zslice_index, :, :]
#X_final_cut = X_final[top, :, :]

field2 = X_final[top, :, :]

# Variable para controlar la escala
use_log = False  # Cambia esto a False para escala normal
# Límites para la escala de colores
vmin_value = 1e-5  # Reemplaza con el valor mínimo que desees
vmax_value = field2.max()   # Reemplaza con el valor máximo que desees
print("Max Ionization ", vmax_value)

# Definición de la norma en función de use_log y límites
if use_log:
    norm_instance = colors.LogNorm(vmin=vmin_value, vmax=vmax_value)
else:
    norm_instance = colors.Normalize(vmin=vmin_value, vmax=vmax_value)

# Para field2 usando pcolormesh
mappable1 = plt.pcolormesh(X_scaled, Y_scaled, field2, shading='gouraud', cmap='viridis', norm=norm_instance, alpha = 1.0)
cbar1 = plt.colorbar(mappable1, label='Ionization')

# Para v_magnitude usando contourf
#contour_levels = np.linspace(v_magnitude.min(), v_magnitude.max(), num=50)  # Ajusta según sea necesario
#mappable2 = plt.contourf(X_scaled, Y_scaled, v_magnitude, levels=contour_levels, cmap='viridis', alpha=0.5)
#cbar2 = plt.colorbar(mappable2, label='Velocity Magnitude')


# Interpolación para suavizar v_magnitude
zoom_factor = 2  # Ajusta este factor según tus necesidades
v_magnitude_smooth = zoom(v_magnitude, zoom_factor, order=3)

# Ajusta la escala de los ejes si has suavizado el campo
X_scaled_zoom = zoom(X_scaled, zoom_factor, order=1)
Y_scaled_zoom = zoom(Y_scaled, zoom_factor, order=1)

print("Vmin =", v_magnitude.min(), "Vmax =", v_magnitude.max())
Vmin = 0.5
Vmax = 3

# Niveles de contorno y trazado
contour_levels = np.linspace(Vmin, Vmax, num=30)  # Ajusta según sea necesario
#mappable2 = plt.contour(X_scaled_zoom, Y_scaled_zoom, v_magnitude_smooth, levels=contour_levels, cmap='viridis',linewidths=2,alpha=0.5)
#cbar2 = plt.colorbar(mappable2, label='Velocity Magnitude')

contour_object = plt.contour(X_scaled_zoom, Y_scaled_zoom, v_magnitude_smooth, levels=contour_levels, cmap='jet', linewidths=2, alpha=0.5)
plt.clabel(contour_object, inline=True, fontsize=10, fmt='%1.2f')
cbar2 = plt.colorbar(contour_object, label='Velocity Magnitude')

# Variable para controlar la visualización de quiver
show_quiver = False  # Cambia esto a False para desactivar quiver

# Condición para añadir quiver
if show_quiver:
    plt.quiver(X_scaled[::subsample, ::subsample], Y_scaled[::subsample, ::subsample],
               v_x_cut[::subsample, ::subsample], v_y_cut[::subsample, ::subsample],
               v_magnitude[::subsample, ::subsample], angles='xy', scale_units='xy', scale=scale,
               cmap='viridis', alpha=alpha, clim=clim, linewidths=linewidths,
               headwidth=headwidth, headlength=headlength, minshaft=minshaft, minlength=minlength)
    
    # Añadir la barra de colores para quiver
    velocity_cbar = plt.colorbar(orientation='horizontal', label='Velocity magnitude', pad=0.1)

# Establece el título y las etiquetas con las unidades de Rh
#plt.title('Velocity Vectors')
plt.xlabel(r'Radial distance [$R_{Hill}$]')
plt.ylabel(r'Azimuthal distance [$R_{Hill}$]')

# Set equal scaling (i.e., make circles circular) by changing dimensions of the plot box.
ax.axis('equal')
# Establece los límites en el eje Y con los valores reescalados
plt.xlim((1 - rp - delta) / Rhill, (1 - rp + delta) / Rhill)
plt.ylim((1 - rp - delta) / Rhill, (1 - rp + delta) / Rhill)


#plor the planet
plt.plot(xp-xp, yp, marker='o') 
# Creating a circle patch
circle1 = patches.Circle((0, yp), radius=Rhill/Rhill, fill=False, edgecolor='k')
circle2 = patches.Circle((0, yp), radius=R_eff_X/Rhill, fill=False, edgecolor='red')

# Adding the circle to the plot
ax.add_patch(circle1)
ax.add_patch(circle2)



##### TEST obtener field1d(r)

# data slices
#density_cut = (density[zslice_index, :, :])
#temperature_cut = temp[zslice_index, :, :]
#X_final_cut = X_final[top, :, :]

dx = X_cut[0, 1] - X_cut[0, 0]
dy = Y_cut[1, 0] - Y_cut[0, 0]

# Define el radio máximo (Rhill)
r_max = Rhill

# Calcula la distancia radial desde cada punto de la grilla hasta (xp, yp)
r_grid = np.sqrt((X_cut - xp)**2 + (Y_cut - yp)**2)

# Valores radiales para el gráfico
r_values = np.arange(0, r_max, dx)

# Inicializa una lista para almacenar los valores medios

def get_1D_values(Field):
    mean_values = []
    # Loop sobre anillos radiales con el mismo espaciado que la grilla
    for r in np.arange(0, r_max, dx):
        mask = (r_grid >= r) & (r_grid < r + dx)
        mean_value = np.mean(Field[mask])
        mean_values.append(mean_value)
    # Convertir la lista en un array de NumPy
    mean_values = np.array(mean_values)
    return mean_values

vel1D = get_1D_values(v_magnitude)
vel1D_smooth = smooth(vel1D)

# Graficar
plt.figure()
plt.plot(r_values / Rhill, smooth(vel1D), 'o-', label='CPD rotational velocity')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('CPD rotational velocity')
plt.title('Perfil radial')


# Constantes
G = 6.674e-11  # m^3 kg^-1 s^-2
M_jupiter = 1.898e27  # kg
v_keplerian = np.sqrt(G * M_jupiter / (r_values*1.496e8)) * 1e-5 #cm/s

# Calcular el quociente entre las dos velocidades
quociente = vel1D_smooth / v_keplerian

# Graficar las velocidades y el quociente
plt.figure()

plt.subplot(1, 2, 1)
plt.plot(r_values/Rhill, vel1D_smooth, label='CPD rotational vel')
plt.plot(r_values/Rhill, v_keplerian, label=f'Keplerian profile')
plt.axvline(x=R_eff_X/Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel(r'Azimuthal velocity [$cm ~ s^{-1}$]')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r_values/Rhill, quociente, label=r'$V_{\theta}/V_{\rm Keplerian}$')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel(r'$V_{\theta}/V_{\rm Keplerian}$')
plt.legend()



### Mas cantidades 1D 

#density_cut = (density[mid, :, :])
#temperature_cut = temp[mid, :, :]
Density1D = get_1D_values(density[top, :, :])   #X_final[top, :, :]  #v_magnitude.astype(np.float64)
Density_smooth = smooth(Density1D)

X1D = get_1D_values(X_final[top, :, :])   #X_final[top, :, :]  #v_magnitude.astype(np.float64)
X1D_smooth = smooth(X1D)

temp1D = get_1D_values(temp[top, :, :])   #X_final[top, :, :]  #v_magnitude.astype(np.float64)
temp1D_smooth = smooth(temp1D)

#compute H/r
Cs_1D = (np.sqrt(get_1D_values(temp[mid, :, :])*float(P("GAMMA"))*Rgas/mu))* 1e-5 #cm/s
Cs_1D_smooth = smooth(Cs_1D)
aspect_ratio1 = (np.delete(Cs_1D,0)/np.delete(v_keplerian,0))**2.0 #eliminar el primer valor, ya que es cero r_value
aspect_ratio2 = (Cs_1D/vel1D)**2.0 #eliminar el primer valor, ya que es cero r_value

# Graficar
plt.figure()
plt.plot(np.delete(r_values,0) / Rhill, aspect_ratio1, 'o-', label='Aspect ratio')
#plt.plot(r_values / Rhill, aspect_ratio2, 'o-', label='Aspect ratio 2')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('CPD Aspect ratio')
plt.title('Aspect ratio')
plt.legend()


# Graficar
plt.figure()
plt.plot(r_values / Rhill, Cs_1D_smooth, 'o-', label='Sound speed smooth')
plt.plot(r_values / Rhill, Cs_1D, 'o-', label='CPD Temperarure')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Sound speed')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('CPD mean sound speed')
plt.title('Perfil radial')
plt.legend()

# Graficar
plt.figure()
plt.plot(r_values / Rhill, temp1D_smooth, 'o-', label='CPD Temperarure smooth')
plt.plot(r_values / Rhill, temp1D, 'o-', label='CPD Temperarure')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('CPD mean temperarure')
plt.title('Perfil radial')
plt.legend()

# Graficar
plt.figure()
plt.plot(r_values / Rhill, X1D_smooth, 'o-', label='Ionization smooth')
plt.plot(r_values / Rhill, X1D, 'o-', label='Ionization')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('Mean ionization level')
plt.title('Ionization')
plt.legend()

# Graficar
plt.figure()
plt.plot(r_values / Rhill, Density_smooth, 'o-', label='Density smooth')
plt.plot(r_values / Rhill, Density1D, 'o-', label='Density')
plt.axvline(x=R_eff_X / Rhill, color='r', linestyle='--', label='Ionization radius')
plt.xlabel(r'Planetocentric distance [$R_{\rm Hill}$]')
plt.ylabel('CPD mean density')
plt.title('Density')
plt.legend()


### grafico de un Field en la direccion vertical:

# Encuentra los índices más cercanos
r_idx = np.argmin(np.abs(r - 1))
theta_idx = np.argmin(np.abs(t - 0))
phi_cut_idx = NX // 2
# Extrae la columna de temperatura vertical
temperature_1D = temp[:, r_idx, phi_cut_idx]
# Extrae los valores correspondientes de z
z_values = z[:, r_idx, phi_cut_idx]/Rhill


# Grafica
plt.figure()
plt.plot(z_values, temperature_1D)
plt.axvline(x=R_eff_X/Rhill, color='r', linestyle='--')  # Añade línea vertical en x = Reff
plt.xlabel('Z')
plt.ylabel('Temperature')
plt.title('Temperature profile at planet position')
plt.xlim(0, Rhill/Rhill)
plt.show()
