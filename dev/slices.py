import numpy as np
import matplotlib.pyplot as plt

# Load the gas density data from the file
data = np.fromfile('/home/jzuluaga/public/outputs/p3diso_f0/gasdens2.dat', dtype=np.float64)

# Load the domain_x.dat, domain_y.dat, and domain_z.dat files
domain_x = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_x.dat')        # coordenada azimuthal theta
domain_y = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_y.dat')[3:-3]  # coordenada radial r
domain_z = np.genfromtxt('/home/jzuluaga/public/outputs/p3diso_f0/domain_z.dat')[3:-3]  # coordenada co-latitud phi

# Get the dimensions of the data grid
NZ, NY, NX = len(domain_z)-1, len(domain_y)-1, len(domain_x)-1

# Reshape the data array to match the dimensions of the grid
data = data.reshape(NZ, NY, NX)

# Define a function to create the Y, Z grid
def Grilla_YZ():
    # Inverse spherical coordinates: R = r sin(theta), Phi = Phi, z = r cos(theta)
    # In this case theta = z is the colatitude angle, while phi = x is the azimuthal angle
    R, T = np.meshgrid(domain_y, domain_z)  # This gives us the radius and azimuth
    Y, Z = R * np.sin(T), R * np.cos(T)  # Convert to Cartesian coordinates
    return Y, Z

# Define a function to create the X, Y grid
def Grilla_XY():
    # Create 2D grids
    R = 0.5 * (domain_y[1:] + domain_y[:-1])
    Phi = 0.5 * (domain_x[1:] + domain_x[:-1]) #En Fargo theta (azimuth) -> phi (coodenada X)
    P, R = np.meshgrid(Phi, R)
    # Convert the cylindrical coordinates (R, P) to Cartesian coordinates (X, Y)
    X, Y = R*np.cos(P), R*np.sin(P)
    return X, Y


# Use the functions to create the Y, Z and X, Y grids
Y, Z = Grilla_YZ()
#X, Y = Grilla_XY()


# Select a slice along the X-axis at the middle
slice_index_x = NX // 2
data_slice_x = data[:, :, slice_index_x]

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
