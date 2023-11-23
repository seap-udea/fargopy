# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

path = "/home/jzuluaga/public/outputs/p3diso_f0/"
# Define the grid dimensions and Phi cut index
variables_par = np.genfromtxt(path+"variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
parametros_par, valores_par = [],[]

for posicion in variables_par:
    parametros_par.append(posicion[0].decode("utf-8"))
    valores_par.append(posicion[1].decode("utf-8"))

def P(parametro):
    return valores_par[parametros_par.index(parametro)]

# Domains
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))
print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)
phi_cut_idx = NX // 2

# Load the data
p = np.genfromtxt(path+'domain_x.dat')  # azimuthal coordinate phi
r = np.genfromtxt(path+'domain_y.dat')[3:-3]  # radial coordinate r
t = np.genfromtxt(path+'domain_z.dat')[3:-3]  # colatitude theta

r = 0.5*(r[:-1] + r[1:]) # r (radial)
p = 0.5*(p[:-1] + p[1:]) # p (azimuth, phi)
t = 0.5*(t[:-1] + t[1:]) # t (co-latitud, theta)

# Define the grids with the correct dimensions
T, R, Phi = np.meshgrid(t, r, p, indexing='ij')
x, y, z = R*np.sin(T)*np.cos(Phi) ,  R*np.sin(T)*np.sin(Phi) , R*np.cos(T)

# Load the velocity data from the files
vx = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvx2.dat', dtype=np.float64).reshape(NZ, NY, NX)
vy = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvy2.dat', dtype=np.float64).reshape(NZ, NY, NX)
vz = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasvz2.dat', dtype=np.float64).reshape(NZ, NY, NX)

# Calculate the spherical-to-Cartesian transformation for the velocity components
v_x = vx * np.sin(T) * np.cos(Phi) + vy * np.sin(T) * np.sin(Phi) + vz * np.cos(T)
v_y = vx * np.cos(T) * np.cos(Phi) + vy * np.cos(T) * np.sin(Phi) - vz * np.sin(T)
v_z = -vx * np.sin(Phi) + vy * np.cos(Phi)



# Calculate the phi cut index
phi_cut_idx = NX // 2

# Cut the Y, Z, and v_z arrays along the phi axis at phi = 0
Y_cut = y[:, :, phi_cut_idx]
Z_cut = z[:, :, phi_cut_idx]
v_z_cut = v_z[:, :, phi_cut_idx]


X, Y, Z = R*np.sin(T)*np.cos(Phi) ,  R*np.sin(T)*np.sin(Phi) , R*np.cos(T)
# Cut the R, Z, and v_z arrays along the phi axis at phi = 0
R_cut = R[:, :, phi_cut_idx]

# Use imshow to plot the grid
plt.figure(figsize=(10, 6))

# Use imshow to plot the grid and save it in the variable im
im = plt.imshow(v_z_cut, origin='lower', aspect='auto', cmap='jet', extent=[r.min(), r.max(), t.min(), t.max()])

# Draw grid lines for each unique value of R and Z
for r_val in r:
    plt.axvline(x=r_val, color='white', linewidth=0.5)
for t_val in t:
    plt.axhline(y=t_val, color='white', linewidth=0.5)

# Draw velocity vectors
step = 5  # Change this value to adjust the number of vectors
#Q = plt.quiver(R_cut[::step, ::step], Z_cut[::step, ::step], np.zeros_like(v_z_cut[::step, ::step]), v_z_cut[::step, ::step], color='white', scale=1e-3)

# Add a key to the quiver plot
#plt.quiverkey(Q, X=0.3, Y=1.1, U=10, label='Quiver key, length = 10', labelpos='E')

# Pass the imshow image to colorbar
plt.colorbar(im, label='Vertical Velocity (v_z)')

plt.xlabel('R (Radial Coordinate)')
plt.ylabel('Z (Vertical Coordinate)')
plt.title('v_z on RZ Plane Cut at Phi = 0')

plt.show()



R, T = np.meshgrid(r, t)  # This gives us the radius and azimuth
Y, Z = R * np.sin(T), R * np.cos(T)  # Convert to Cartesian coordinates

slice_index_x = NX // 2
data_slice_x = v_z[:, :, slice_index_x]

# Find the minimum and maximum values in the Y and Z grids
Y_min, Y_max = Y.min(), Y.max()
Z_min, Z_max = Z.min(), Z.max()

# Plot the data in Cartesian coordinates
plt.figure(figsize=(10, 10))
plt.pcolormesh(Y, Z, data_slice_x, cmap='hot')
plt.colorbar(label='Density')
plt.title(f'Gas Density Visualization (X-slice at index {slice_index_x})')
plt.xlabel('Y')
plt.ylabel('Z')
plt.axis([Y_min, Y_max, Z_min, Z_max])
plt.show()






