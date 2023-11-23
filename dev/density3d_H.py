import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
from matplotlib.colors  import ListedColormap
from matplotlib.ticker import FuncFormatter, MultipleLocator


variables_par = np.genfromtxt("/home/jzuluaga/public/outputs/p3diso_f0/variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
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
c_speed       = 2.99e10                                                                                                 #cm/s
unit_length   = (1.0)*1.49598e13                                                                                #cm
unit_mass     = 1.9891e33                                                                                               #gr
unit_density  = unit_mass/unit_length**3                                                                #gr / cm3
unit_time     = np.sqrt( pow(unit_length,3.) / G / unit_mass)/3.154e7   #yr 
unit_disktime = float(P("NINTERM"))*float(P("DT"))*unit_time                    #yr/snapshot      [Las unidades de tiempo del disco se toman en R=1]
unit_velocity = 1e-5*unit_length/float(unit_time*3.154e7)                               #km/s
unit_energy   = unit_mass/(unit_time*3.154e7)**2/unit_length                    #erg/cm3 = gr/s2/cm
unit_eV       = 1.60218e-12                                                                                     #erg
mproton       = 1.6726231e-24                                                                                   #gr
h_planck      = 6.6260755e-27                                                                                   #erg*s
kb            = 1.380658e-16                                                                                    #erg/K

 #Dominios 
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))

print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)

# Load the gas density data from the file
Density = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasdens2.dat', dtype=np.float64).reshape(NZ, NY, NX)
Energy = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasenergy2.dat', dtype=np.float64).reshape(NZ, NY, NX)
gasvx = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvx2.dat').reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) 
gasvy = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvy2.dat').reshape(int(P("NZ")),int(P("NY")),int(P("NX"))) 
#gasvt = gasvx+gasvp

Temperature   = Energy/Density/(Rgas/mu)*(float(P("GAMMA"))-1)                                                                                           #K
cs     = np.sqrt(Temperature*float(P("GAMMA"))*Rgas/mu)/unit_length*3.154e7


# Load the domain_x.dat, domain_y.dat, and domain_z.dat files
domain_x = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0//domain_x.dat')        # coordenada azimuthal phi
domain_y = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_y.dat')[3:-3]  # coordenada radial r
domain_z = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_z.dat')[3:-3]  # coordenada co-latitud theta

r = 0.5*(domain_y[:-1] + domain_y[1:]) #X-Center
p = 0.5*(domain_x[:-1] + domain_x[1:]) #X-Center
t = 0.5*(domain_z[:-1] + domain_z[1:]) #X-Center

R_centro = r.reshape(int(float(P("NY"))),1)
G = 1.0
M = 1.0
Omega = (G*M/R_centro)**0.5
H = 1e5* cs / Omega

# Locate the index closest to r=1, theta=0, phi=0
r_idx = np.argmin(np.abs(r - 1))
theta_idx = np.argmin(np.abs(t - 0))
phi_idx = np.argmin(np.abs(p - 0))

index_closest_to_one = np.argmin(np.abs(r - 1))
H_at_planet = H[index_closest_to_one]

print("Value of H just above the planet:", H_at_planet)



#Grillas
print("Calcuando Grillas ...")
R,T,Phi    = np.meshgrid(r,t,p)
X,Y,Z      = R*np.cos(Phi)*np.sin(T) ,  R*np.sin(Phi)*np.sin(T) , R*np.cos(T)
# Calculate the actual z coordinate of each cell
Z_actual = R * np.cos(T)

# Create a mask for the cells where the actual z coordinate is greater than or equal to H
H = H.reshape(Z_actual.shape)
mask_H = Z_actual >= H

# Apply the mask for height H to the density data
Density_alpha = Density.copy()  # Create a copy of Density
alpha = 1.0
Density_alpha[mask_H] *= alpha  # Set the masked values to their original density but with transparency alpha

# Create a mask for the cells where 0 <= Phi <= pi/4
mask_Phi = (Phi >= 0.0) & (Phi <= np.pi/18)

# Flatten the arrays for 3D scatter plot
X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
Density_alpha_flat = Density_alpha.flatten()

# Apply the mask to the flattened arrays
X_cut_Phi = X_flat[mask_Phi.flatten()]
Y_cut_Phi = Y_flat[mask_Phi.flatten()]
Z_cut_Phi = Z_flat[mask_Phi.flatten()]
Density_alpha_cut_Phi = Density_alpha_flat[mask_Phi.flatten()]

# Prepare the color map
cmap = plt.get_cmap('viridis')

# Adjust color normalization: ignore the top and bottom 1% of the density values
vmin = np.percentile(Density_alpha_cut_Phi, 1)
vmax = np.percentile(Density_alpha_cut_Phi, 99)
norm = plt.Normalize(vmin, vmax)
colors = cmap(norm(Density_alpha_cut_Phi))

# Create 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_cut_Phi, Y_cut_Phi, Z_cut_Phi, c=colors, alpha=0.5)

# Rotate the view of the plot
ax.view_init(azim=-120)

# Add a color bar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
cbar.set_label('Gas Density')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
