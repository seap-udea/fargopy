import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

variables_par = np.genfromtxt("/home/jzuluaga/public/outputs/p3diso_f0/variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
parametros_par, valores_par = [],[]


for posicion in variables_par:
    parametros_par.append(posicion[0].decode("utf-8"))
    valores_par.append(posicion[1].decode("utf-8"))


def P(parametro):
    return valores_par[parametros_par.index(parametro)]


 #Dominios 
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))

print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)



# Load data
domain_x = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/domain_x.dat", dtype=np.float64)
domain_y = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/domain_y.dat", dtype=np.float64)
domain_z = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/domain_z.dat", dtype=np.float64)
vX_data = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/gasvx2.dat", dtype=np.float64).reshape((NZ, NY, NX), order='F')
vY_data = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/gasvy2.dat", dtype=np.float64).reshape((NZ, NY, NX), order='F')
vZ_data = np.fromfile("/home/jzuluaga/public/outputs/p3diso_f0/gasvz2.dat", dtype=np.float64).reshape((NZ, NY, NX), order='F')

# Load the domain_x.dat, domain_y.dat, and domain_z.dat files
#domain_x = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_x.dat')        # coordenada azimuthal phi
#domain_y = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_y.dat')[3:-3]  # coordenada radial r
#domain_z = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_z.dat')[3:-3]  # coordenada co-latitud theta

# Load the velocity data files
#vX_data = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvx2.dat', dtype=np.float64).reshape(NZ, NY, NX)
#vY_data = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvy2.dat', dtype=np.float64).reshape(NZ, NY, NX)
#vZ_data = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvz2.dat', dtype=np.float64).reshape(NZ, NY, NX)




# Convert spherical polar coordinates to cartesian
R, T = np.meshgrid(domain_y, domain_z)
Y, Z = R * np.sin(T), R * np.cos(T)

# Define the position in the phi direction where the cut is made
cut_phi = np.argmin(np.abs(domain_x - np.pi/2))

# Interpolate the velocity data to the cell centers
x = np.linspace(domain_y[0], domain_y[-1], NY)
y = np.linspace(domain_z[0], domain_z[-1], NZ)

# Define the interpolation function
x = np.sort(x)
interp_func_vy = RectBivariateSpline(y, x, vY_data[:,:,cut_phi])
interp_func_vz = RectBivariateSpline(y, x, vZ_data[:,:,cut_phi])

# Perform the interpolation
vy_cut = interp_func_vy.ev(Z, Y)
vz_cut = interp_func_vz.ev(Z, Y)

# Calculate the magnitude of the velocity vector
v_magnitude_cut = np.sqrt(vy_cut**2 + vz_cut**2)

# Normalize the colors based on the magnitude of the velocity vector
colors_cut = v_magnitude_cut / np.max(v_magnitude_cut)

# Define plot parameters
step_arrow = 5
scale = 1.0

# Plot the velocity vectors in the YZ plane
fig, ax = plt.subplots(figsize=(10,10))

ax.quiver(Y[::step_arrow, ::step_arrow], Z[::step_arrow, ::step_arrow], 
          vy_cut[::step_arrow, ::step_arrow], vz_cut[::step_arrow, ::step_arrow], 
          colors_cut[::step_arrow, ::step_arrow], pivot="middle", units='xy', scale=scale)

ax.set_xlabel('Y')
ax.set_ylabel('Z')
ax.set_title('2D vector plot of velocity in the YZ plane at Phi = {}'.format(domain_x[cut_phi]))
#ax.set_yscale('log')
#ax.set_xscale('log')

# Add a colorbar to the plot
cbar = plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, np.max(v_magnitude_cut)), cmap='viridis'))
cbar.set_label('Velocity magnitude')

plt.show()
