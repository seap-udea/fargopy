import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

path = "/home/jzuluaga/public/outputs/p3diso_f0/"
variables_par = np.genfromtxt(path+"variables.par",dtype={'names': ("parametros","valores"),'formats': ("|S30","|S300")}).tolist()
parametros_par, valores_par = [],[]


for posicion in variables_par:
    parametros_par.append(posicion[0].decode("utf-8"))
    valores_par.append(posicion[1].decode("utf-8"))


def P(parametro):
    return valores_par[parametros_par.index(parametro)]


 #Dominios
NX = int(P("NX")); NY = int(P("NY")); NZ = int(P("NZ"))

print('NX= ', NX, 'NY= ', NY, 'NZ= ', NZ)

# Load the gas density data from the file
data = np.fromfile(path+'gasdens2.dat', dtype=np.float64).reshape(NZ, NY, NX)
#data.reshape(NZ, NY, NX)

# Load the domain_x.dat, domain_y.dat, and domain_z.dat files
domain_x = np.genfromtxt(path+'domain_x.dat')        # coordenada azimuthal phi
domain_y = np.genfromtxt(path+'domain_y.dat')[3:-3]  # coordenada radial r
domain_z = np.genfromtxt(path+'domain_z.dat')[3:-3]  # coordenada co-latitud theta

r = 0.5*(domain_y[:-1] + domain_y[1:]) #X-Center
p = 0.5*(domain_x[:-1] + domain_x[1:]) #X-Center
t = 0.5*(domain_z[:-1] + domain_z[1:]) #X-Center

#Grillas
print("Calcuando Grillas ...")
R,T,Phi    = np.meshgrid(r,t,p)
X,Y,Z      = R*np.cos(Phi)*np.sin(T) ,  R*np.sin(Phi)*np.sin(T) , R*np.cos(T)

# Define the slice mask based on the Phi grid
#slice_angle = np.pi/4
slice_angle = 0.0
#slice_mask = (Phi > slice_angle) & (Phi < 3*slice_angle)
slice_mask = (Phi > slice_angle) & (Phi < np.pi/2)
#print(Phi)
# Get the sliced X, Y, Z and gasdens25 arrays
X_slice = X[slice_mask]
Y_slice = Y[slice_mask]
Z_slice = Z[slice_mask]
data_slice = data[slice_mask]




# Define the color normalization
norm = plt.Normalize(np.log10(data_slice).min(), np.log10(data_slice).max())
colors = plt.cm.viridis(norm(np.log10(data_slice)))

# Plot the slice
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_slice, Y_slice, Z_slice, c=colors.reshape(-1,4), s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D slice plot of gas density')

plt.show()


