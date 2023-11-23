###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import os
import numpy as np
import re

###############################################################
# Constants
###############################################################
KB = 1.380650424e-16  # Boltzmann constant: erg/K, erg = g cm^2 / s^2
MP = 1.672623099e-24 # Mass of the proton, g
GCONST = 6.67259e-8 # Gravitational constant, cm^3/g/s^2 
RGAS = 8.314472e7 # Gas constant, erg /K/mol
MSUN = 1.9891e33 # g
AU = 1.49598e13 # cm
YEAR = 31557600.0 # s

# Map of coordinates into FARGO3D coordinates
COORDS_MAP = dict(
    cartesian = dict(x='x',y='y',z='z'),
    cylindrical = dict(phi='x',r='y',z='z'),
    spherical = dict(phi='x',r='y',theta='z'),
)

###############################################################
# Classes
###############################################################
class Fargobj(object):
    def __init__(self):
        self.fobject = True

    def has(self,key):
        if key in self.__dict__.keys():
            return True
        else:
            return False

class Fargo3d(Fargobj):

    def __init__(self):
        super().__init__()
   
    def compile_fargo3d(self,clean=True):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):    
            if clean:
                Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)
            
            error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} PARALLEL={self.parallel} GPU={self.gpu}',verbose=False)
            if error:
                if not Conf._check_fargo_binary(Conf.FARGO3D_FULLDIR,quiet=True):
                    print("An error compiling the code arose. Check dependencies.")
                    print(Util.STDERR)
                return False
            
            return True

class Field(Fargobj):
    """Fields:

    Attributes:
        coordinates: type of coordinates (cartesian, cylindrical, spherical)
        data: numpy arrays with data of the field

    Methods:
        slice: get an slice of a field along a given spatial direction.
            Examples: 
                density.slice(r=0.5) # Take the closest slice to r = 0.5
                density.slice(ir=20) # Take the slice through the 20 shell
                density.slice(phi=30*RAD,interp='nearest') # Take a slice interpolating to the nearest
    """

    def __init__(self,data=None,coordinates='cartesian',domains=None):
        self.data = data
        self.coordinates = coordinates
        self.domains = domains

    def meshslice(self,slice='x,y,z',output='cartesian:x,y,z'):
        """Perform a slice on a field and produce as an output the 
        corresponding field slice and the associated matrices of
        coordinates for plotting.

        Examples:
            # Don't do nothing, but r, phi, z are matrices with the same dimensions of gasdens
            r,phi,z,gasdens = gasdens.meshslice(slice='r,phi,z', output='cylindrical:r,phi,z')

            # Make a z=0 slice and return the r and phi matrices for plotting the field.
            r,phi,gasdens_rp = gasdens.meshslice(slice='r,phi,z=0', output='spherica:r,phi')
            plt.pcolormesh(r,phi,gasdens_rp)
            plt.pcolormesh(phi,r,gasdens_rp.T)

            # Make a z=0, iphi = 0 slice and return r
            r, gasdens_r = gasdens.meshslice(slice='r,iphi=0,z=0', output='cylindrical:r')
            plt.plot(r,gasdens_r)

            # Make a z = 0 slice and express the result in cartesian coordinates
            x,y,gasdens_xy = gasdens.meshslice(slice='r,phi,z=0', output='cartesian:x,y')
            plt.pcolormesh(x,y,gasdens_xy)

            # Make a phi = 0 slice and express the result in cartesian coordinates
            y,z,gasdens_yz = gasdens.meshslice(slice='r,phi=0,z', output='cartesian:y,z')
            plt.pcolormesh(y,z,gasdens_yz)
        """
        # Analysis of the slice 

        # Perform the slice
        slice_cmd = f"self.slice({slice},pattern=True)"
        slice,pattern = eval(slice_cmd)
        
        # Create the mesh
        if self.coordinates == 'cartesian':
            z,y,x = np.meshgrid(self.domains.z,self.domains.y,self.domains.x,indexing='ij')
            x,y,z = r*np.cos(phi),r*np.sin(phi),z

            x = eval(f"x[{pattern}]")
            y = eval(f"y[{pattern}]")
            z = eval(f"z[{pattern}]")
            
            mesh = fargopy.Dictobj(dict=dict(x=x,y=y,z=z))

        if self.coordinates == 'cylindrical':
            z,r,phi = np.meshgrid(self.domains.z,self.domains.r,self.domains.phi,indexing='ij')
            x,y,z = r*np.cos(phi),r*np.sin(phi),z

            x = eval(f"x[{pattern}]")
            y = eval(f"y[{pattern}]")
            z = eval(f"z[{pattern}]")
            r = eval(f"r[{pattern}]")
            phi = eval(f"phi[{pattern}]")

            mesh = fargopy.Dictobj(dict=dict(r=r,phi=phi,x=x,y=y,z=z))

        if self.coordinates == 'spherical':
            theta,r,phi = np.meshgrid(self.domains.theta,self.domains.r,self.domains.phi,indexing='ij')
            x,y,z = r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)

            x = eval(f"x[{pattern}]")
            y = eval(f"y[{pattern}]")
            z = eval(f"z[{pattern}]")
            r = eval(f"r[{pattern}]")
            phi = eval(f"phi[{pattern}]")
            theta = eval(f"theta[{pattern}]")

            mesh = fargopy.Dictobj(dict=dict(r=r,phi=phi,theta=theta,x=x,y=y,z=z))

        return slice,mesh

    def slice(self,quiet=True,pattern=False,**kwargs):
        """Extract an slice of a 3-dimensional FARGO3D field

        Parameters:
            quiet: boolean, default = False:
                If True extract the slice quietly.
                Else, print some control messages.

            ir, iphi, itheta, ix, iy, iz: string or integer:
                Index or range of indexes of the corresponding coordinate.

            r, phi, theta, x, y, z: float:
                Value for slicing. The slicing search for the closest
                value in the domain.

        Returns:
            slice: sliced field.

        Examples:
            # 0D: Get the value of the field in iphi = 0, itheta = -1 and close to r = 0.82
            gasvz.slice(iphi=0,itheta=-1,r=0.82)

            # 1D: Get all values of the field in radial direction at iphi = 0, itheta = -1
            gasvz.slice(iphi=0,itheta=-1)

            # 2D: Get all values of the field for values close to phi = 0
            gasvz.slice(phi=0)
        """
        # By default slice
        ivar = dict(x=':',y=':',z=':')

        if len(kwargs.keys()) == 0:
            pattern_str = f"{ivar['z']},{ivar['y']},{ivar['x']}"
            if pattern:
                return self.data, pattern_str
            return self.data
            
        # Check all conditions
        for key,item in kwargs.items():
            match = re.match('^i(.+)',key)
            if match:
                index = item
                coord = match.group(1)
                if not quiet:
                    print(f"Index condition {index} for coordinate {coord}")
                ivar[COORDS_MAP[self.coordinates][coord]] = index
            else:
                if not quiet:
                    print(f"Numeric condition found for coordinate {key}")
                if key in self.domains.keys():
                    # Check if value provided is in range
                    domain = self.domains.item(key)
                    extrema = self.domains.extrema[key]
                    min, max = extrema[0][1], extrema[1][1]
                    if (item<min) or (item>max):
                        raise ValueError(f"You are attempting to get a slice in {key} = {item}, but the valid range for this variable is [{min},{max}]")
                    find = abs(self.domains.item(key) - item)
                    ivar[COORDS_MAP[self.coordinates][key]] = find.argmin()
                    
        pattern_str = f"{ivar['z']},{ivar['y']},{ivar['x']}"
        slice_cmd = f"self.data[{pattern_str}]"
        if not quiet:
            print(f"Slice: {slice_cmd}")
        slice = eval(slice_cmd)

        if pattern:
            return slice,pattern_str
        return slice

    def get_size(self):
        return self.data.nbytes

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)

class Simulation(Fargo3d):

    def __init__(self):
        super().__init__()
        self.output_dir = None
        self.set_units(UL=AU,UM=MSUN)

    def set_output_dir(self,dir):
        if not os.path.isdir(dir):
            print(f"Directory '{dir}' does not exist.")
        else:
            print(f"Now you are connected with output directory '{dir}'")
        self.output_dir = dir

    def list_outputs(self,quiet=False):
        if self.output_dir is None:
            print(f"You have to set forst the outputs directory with <sim>.set_outputs('<directory>')")
        else:
            error,output = fargopy.Sys.run(f"ls {self.output_dir}")
        if error == 0:
            files = output[:-1]
            print(f"{len(files)} files in output directory")
            if not quiet:
                file_list = ""
                for i,file in enumerate(files):
                    file_list += f"{file}, "
                    if ((i+1)%10) == 0:
                        file_list += "\n"
                print(file_list)
            return files

    def load_properties(self,quiet=False,
                        varfile='variables.par',
                        domain_prefix='domain_',
                        dimsfile='dims.dat'
                        ):
        if self.output_dir is None:
            print(f"You have to set forst the outputs directory with <sim>.set_outputs('<directory>')")

        # Read variables
        vars = self._load_variables(varfile)
        print(f"Simulation in {vars.DIM} dimensions")
        
        # Read domains 
        domains = self._load_domains(vars,domain_prefix)

        # Store the variables in the object
        self.vars = vars
        self.domains = domains

        # Optionally read dims
        dims = self._load_dims(dimsfile)
        if len(dims):
            self.dims = dims 
        
        print("Configuration variables and domains load into the object. See e.g. <sim>.vars")
        return vars, domains
    
    def _load_dims(self,dimsfile):
        """Parse the dim directory
        """
        dimsfile = f"{self.output_dir}/{dimsfile}".replace('//','/')
        if not os.path.isfile(dimsfile):
            #print(f"No file with dimensions '{dimsfile}' found.")
            return []
        dims = np.loadtxt(dimsfile)
        return dims

    def _load_variables(self,varfile):
        """Parse the file with the variables
        """

        varfile = f"{self.output_dir}/{varfile}".replace('//','/')
        if not os.path.isfile(varfile):
            print(f"No file with variables named '{varfile}' found.")
            return

        print(f"Loading variables")
        variables = np.genfromtxt(
            varfile,dtype={'names': ("parameters","values"),
            'formats': ("|S30","|S300")}
        ).tolist()
        
        vars = dict()
        for posicion in variables:
            str_value = posicion[1].decode("utf-8")
            try:
                value = int(str_value)
            except:
                try:
                    value = float(str_value)
                except:
                    value = str_value
            vars[posicion[0].decode("utf-8")] = value
        
        vars = fargopy.Dictobj(dict=vars)
        print(f"{len(vars.__dict__.keys())} variables loaded")

        # Create additional variables
        variables = ['x', 'y', 'z']
        if vars.COORDINATES == 'cylindrical':
            variables = ['phi', 'r', 'z']
        elif vars.COORDINATES == 'spherical':
            variables = ['phi', 'r', 'theta']
        vars.VARIABLES = variables

        vars.__dict__[f'N{variables[0].upper()}'] = vars.NX
        vars.__dict__[f'N{variables[1].upper()}'] = vars.NY
        vars.__dict__[f'N{variables[2].upper()}'] = vars.NZ

        # Dimension of the domain
        vars.DIM = 2 if vars.NZ == 1 else 3
        
        return vars

    def _load_domains(self,vars,domain_prefix,
                      borders=[[],[3,-3],[3,-3]],
                      middle=True):

        # Coordinates
        variable_suffixes = ['x', 'y', 'z']
        print(f"Loading domain in {vars.COORDINATES} coordinates:")

        # Correct dims in case of 2D
        if vars.DIM == 2:
            borders[-1] = []

        # Load domains
        domains = dict()
        domains['extrema'] = dict()

        for i,variable_suffix in enumerate(variable_suffixes):
            domain_file = f"{self.output_dir}/{domain_prefix}{variable_suffix}.dat".replace('//','/')
            if os.path.isfile(domain_file):

                # Load data from file
                domains[vars.VARIABLES[i]] = np.genfromtxt(domain_file)

                if len(borders[i]) > 0:
                    # Drop the border of the domain
                    domains[vars.VARIABLES[i]] = domains[vars.VARIABLES[i]][borders[i][0]:borders[i][1]]

                if middle:
                    # Average between domain cell coordinates
                    domains[vars.VARIABLES[i]] = 0.5*(domains[vars.VARIABLES[i]][:-1]+domains[vars.VARIABLES[i]][1:])
                
                # Show indices and value map
                domains['extrema'][vars.VARIABLES[i]] = [[0,domains[vars.VARIABLES[i]][0]],[-1,domains[vars.VARIABLES[i]][-1]]]
                
                print(f"\tVariable {vars.VARIABLES[i]}: {len(domains[vars.VARIABLES[i]])} {domains['extrema'][vars.VARIABLES[i]]}")
            else:
                print(f"\tDomain file {domain_file} not found.")
        domains = fargopy.Dictobj(dict=domains)

        return domains        

    def load_field(self,field,snapshot=None,type='scalar'):

        if not self.has('vars'):
            # If the simulation has not loaded the variables
            dims, vars, domains = sim.load_properties()
        
        # In case no snapshot has been provided use 0
        snapshot = 0 if snapshot is None else snapshot

        field_data = []
        if type == 'scalar':
            file_name = f"{field}{str(snapshot)}.dat"
            file_field = f"{self.output_dir}/{file_name}".replace('//','/')
            field_data = self._load_field_scalar(file_field)
        elif type == 'vector':
            field_data = []
            variables = ['x','y'] 
            if self.vars.DIM == 3:
                variables += ['z'] 
            for i,variable in enumerate(variables):
                file_name = f"{field}{variable}{str(snapshot)}.dat"
                file_field = f"{self.output_dir}/{file_name}".replace('//','/')
                field_data += [self._load_field_scalar(file_field)]
        else:
            raise ValueError(f"Field type '{type}' not recognized.")

        field = Field(data=np.array(field_data), coordinates=self.vars.COORDINATES, domains=self.domains)
        return field
    
    def _load_field_scalar(self,file):
        """Load scalar field from file a file.
        """
        if os.path.isfile(file):
            field_data = np.fromfile(file).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            """
            if self.vars.NZ > 1:
                # 3D field
                field_data = np.fromfile(file).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            else:
                # 2D field
                field_data = np.fromfile(file).reshape(int(self.vars.NY),int(self.vars.NX))
            """
            return field_data
        else:
            raise AssertionError(f"File with field '{file}' not found")

    def load_allfields(self,fluid,snapshot=None):
        """Load all fields in the output
        """
        qall = False
        if snapshot is None:
            qall = True
            fields = fargopy.Dictobj()
        else:
            fields = fargopy.Dictobj()
        
        # Search for field files
        pattern = f"{self.output_dir}/{fluid}*.dat"
        error,output = fargopy.Sys.run(f"ls {pattern}")

        if not error:
            size = 0
            for file_field in output[:-1]:
                comps = Simulation._parse_file_field(file_field)
                if comps:
                    if qall:
                        # Store all snapshots
                        field_name = comps[0]
                        field_snap = int(comps[1])
                        field_data = self._load_field_scalar(file_field)
                        if str(field_snap) not in fields.keys():
                            fields.__dict__[str(field_snap)] = fargopy.Dictobj()
                        size += field_data.nbytes
                        (fields.__dict__[str(field_snap)]).__dict__[f"{field_name}"] = Field(data=field_data, coordinates=self.vars.COORDINATES, domains=self.domains)
                    else:
                        # Store a specific snapshot
                        if int(comps[1]) == snapshot:
                            field_name = comps[0]
                            field_data = self._load_field_scalar(file_field)
                            size += field_data.nbytes
                            fields.__dict__[f"{field_name}"] = Field(data=field_data, coordinates=self.vars.COORDINATES, domains=self.domains)

        else:
            raise ValueError(f"No field found with pattern '{pattern}'. Change the fluid")
    
        if qall:
            fields.snapshots = sorted([int(s) for s in fields.keys() if s != 'size'])
        fields.size = size/1024**2
        return fields

    @staticmethod
    def _parse_file_field(file_field):
        basename = os.path.basename(file_field)
        comps = None
        match = re.match('([a-zA-Z]+)(\d+).dat',basename)
        if match is not None:
            comps = [match.group(i) for i in range(1,match.lastindex+1)]
        return comps
    
    def set_units(self,UM=MSUN,UL=AU,G=1,mu=2.35):
        """Set units of simulation
        """
        # Basic
        self.UM = UM
        self.UL = UL
        self.G = G
        self.UT = (G*self.UL**3/(GCONST*self.UM))**0.5 # In seconds

        # Thermodynamics
        self.UTEMP = (GCONST*MP*mu/KB)*self.UM/self.UL # In K
        
        # Derivative
        self.USIGMA = self.UM/self.UL**2 # In g/cm^2
        self.URHO = self.UM/self.UL**3 # In kg/m^3
        self.UEPS = self.UM/(self.UL*self.UT**2)  # In J/m^3
        self.UV = self.UL/self.UT

    
        
