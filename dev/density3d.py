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
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
import scipy.spatial
from scipy.interpolate import Rbf
from scipy.spatial import Delaunay
import warnings
from numpy.linalg import eig
from numpy.linalg import eigh


# Ignorar todos los warnings de deprecación
warnings.filterwarnings("ignore", category=DeprecationWarning)

#path_on = "/home/matias/Dropbox/Simulations/Fargo3D/Visualizations/M1_f1_highres/"
#path_off = "/home/matias/Dropbox/Simulations/Fargo3D/Visualizations/M1_f0_highres/"
path_on = "/home/jzuluaga/public/outputs/p3diso_f0/"
path_off = "/home/jzuluaga/public/outputs/p3diso_f0/"

output = 2

# Define the grid dimensions and Phi cut index
variables_par = np.genfromtxt(path_off+"variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
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
planet  = np.genfromtxt(path_off+"bigplanet0.dat") 
xp      = planet[snapshot][1] ; yp      = planet[snapshot][2]*R_scale
zp      = planet[snapshot][3] ; mp      = planet[snapshot][7]
rp      = np.sqrt((xp**2)+(yp**2)+(zp**2))

Rhill   = rp*(mp/3)**(1./3)
#print(Rhill, rp, xp, yp, zp)

# Domains
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))
print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)

#Dominios
print("Calculando Dominios ...")
dx = np.loadtxt(path_off+"domain_x.dat")
dy = np.loadtxt(path_off+"domain_y.dat")[3:-3]*R_scale
dz = np.loadtxt(path_off+"domain_z.dat")[3:-3]
r = 0.5*(dy[:-1] + dy[1:]) #X-Center
p = 0.5*(dx[:-1] + dx[1:]) #X-Center
t = 0.5*(dz[:-1] + dz[1:]) #X-Center
#Grillas
print("Calcuando Grillas ...")
R,T,Phi = np.meshgrid(r,t,p)
X,Y,Z = R*np.cos(Phi)*np.sin(T) ,  R*np.sin(Phi)*np.sin(T) , R*np.cos(T)
RS    = np.sqrt(X**2+Y**2+Z**2) 


# Load Fields
density_on = np.fromfile(path_on+'gasdens'+str(output)+'.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_density       #gr/cm3
energy_on = np.fromfile(path_on+'gasenergy'+str(output)+'.dat').reshape(NZ,NY,NX)*unit_energy      #erg/cm3
temp_on   = energy_on/density_on/(Rgas/mu)*(float(P("GAMMA"))-1) 

density_off = np.fromfile(path_off+'gasdens'+str(output)+'.dat', dtype=np.float64).reshape(NZ, NY, NX)*unit_density       #gr/cm3
energy_off = np.fromfile(path_off+'gasenergy'+str(output)+'.dat').reshape(NZ,NY,NX)*unit_energy      #erg/cm3
temp_off   = energy_off/density_off/(Rgas/mu)*(float(P("GAMMA"))-1)


##### Compute Ionization degree

#k_rhill = 1.0/4.0
#k_rhill = 1.0/3.0
#cs      = np.sqrt(temp*float(P("GAMMA"))*Rgas/mu)                                      #cm/s
#Mp      = mp*unit_mass                                                                                          #gr
#Rbondi  = (G*Mp/(cs**2))/unit_length  
#   1/Reff = 1/Rbondi+1/(k*Rhill)   |----->|   1/Reff = Rbondi+(k*Rhill)/(Rbondi*k*Rhill)   |--->|   Reff = (Rbondi*k*Rhill)/(Rbondi+(k*Rhill))
#Reff    = (Rbondi*k_rhill*Rhill)/(Rbondi+(k_rhill*Rhill))
#print('Reff: ', Reff)

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

X_final_on = calculate_Gamma_X(temp_on)
X_final_off = calculate_Gamma_X(temp_off)

print(X_final_on)

# Flatten the arrays for 3D scatter plot
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()

Density_flat = X_final_on.flatten()
#Density_flat = density_on.flatten()

# Filter the data to include only the points that satisfy Phi = 0
#mask = (Phi >= -1.570796) & (Phi <= 1.570796)
#mask = (Phi >= 0.0) & (Phi <= np.pi/6) #Phi = 0, posicion del planeta
mask = (Phi >= -0.3) & (Phi <= 0.0) #Phi = 0, posicion del planeta

# Apply the mask to the flattened arrays
X_cut = X_flat[mask.flatten()]
Y_cut = Y_flat[mask.flatten()]
Z_cut = Z_flat[mask.flatten()]
Density_cut = np.log10(Density_flat[mask.flatten()])

# Prepare the color map
cmap = plt.get_cmap('viridis')

# Adjust color normalization: ignore the top and bottom 1% of the density values
vmin = -20# np.percentile(Density_cut, 1)
vmax = 0 #np.percentile(Density_cut, 99)
norm = plt.Normalize(vmin, vmax)
colors = cmap(norm(Density_cut))

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 8), facecolor='black')  # Set the background color of the figure to black
ax = fig.add_subplot(111, projection='3d', facecolor='black')  # Set the background of the axes to black

scatter = ax.scatter(X_cut, Y_cut, Z_cut, c=colors, alpha=0.5)

# Rotate the view of the plot
ax.view_init(azim=-110)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Ionization degree', color='white')  # Set the color of the label
cbar.ax.yaxis.set_tick_params(color='white')  # Set the color of the colorbar ticks
cbar.outline.set_edgecolor('white')  # Set the color of the colorbar outline

# Change the color of the ticks
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')
ax.tick_params(axis='z', colors='white')

# Set labels with a specific color
ax.set_xlabel('X', color='white')
ax.set_ylabel('Y', color='white')
ax.set_zlabel('Z', color='white')

# Set the color of the axis labels
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.zaxis.label.set_color('white')

# Set grid and pane color
ax.xaxis.pane.fill = False  # Remove the grid background
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)  # Disable the grid
"""
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the x axis line color to transparent (invisible)
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the y axis line color to transparent
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the z axis line color to transparent
"""
plt.show()
