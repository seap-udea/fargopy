###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import numpy as np
import re
import pandas as pd

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter
from scipy.interpolate import RBFInterpolator

from joblib import Parallel, delayed


from ipywidgets import interact, FloatSlider, IntSlider
from celluloid import Camera
from IPython.display import HTML, Video

from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from tqdm import tqdm

###############################################################
# Constants
###############################################################
# Map of coordinates into FARGO3D coordinates
"""This dictionary maps the coordinates regular names (r, phi, theta, etc.) of
different coordinate systems into the FARGO3D x, y, z
"""
COORDS_MAP = dict(
    cartesian = dict(x='x',y='y',z='z'),
    cylindrical = dict(phi='x',r='y',z='z'),
    spherical = dict(phi='x',r='y',theta='z'),
)

###############################################################
# Classes
###############################################################
class Field(fargopy.Fargobj):
    """Fields:

    Attributes:
        coordinates: type of coordinates (cartesian, cylindrical, spherical)
        data: numpy arrays with data of the field

    Methods:
        slice: get an slice of a field along a given spatial direction.
            Examples: 
                >>> density.slice(r=0.5) # Take the closest slice to r = 0.5
                >>> density.slice(ir=20) # Take the slice through the 20 shell
                >>> density.slice(phi=30*RAD,interp='nearest') # Take a slice interpolating to the nearest
    """

    def __init__(self,data=None,coordinates='cartesian',domains=None,type='scalar',**kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.coordinates = coordinates
        self.domains = domains
        self.type = type

    def meshslice(self,slice=None,component=0,verbose=False):
        """Perform a slice on a field and produce as an output the 
        corresponding field slice and the associated matrices of
        coordinates for plotting.
        """
        # Analysis of the slice 
        if slice is None:
            raise ValueError("You must provide a slice option.")

        # Degrees specification
        slice = slice.replace('deg','*fargopy.DEG')

        # Perform the slice
        slice_cmd = f"self.slice({slice},pattern=True,verbose={verbose})"
        slice,pattern = eval(slice_cmd)
        
        # Create the mesh
        if self.coordinates == 'cartesian':
            z,y,x = np.meshgrid(self.domains.z,self.domains.y,self.domains.x,indexing='ij')
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

    def slice(self,verbose=False,pattern=False,**kwargs):
        """Extract an slice of a 3-dimensional FARGO3D field

        Parameters:
            quiet: boolean, default = False:
                If True extract the slice quietly.
                Else, print some control messages.

            pattern: boolean, default = False:
                If True return the pattern of the slice, eg. [:,:,:]

            ir, iphi, itheta, ix, iy, iz: string or integer:
                Index or range of indexes of the corresponding coordinate.

            r, phi, theta, x, y, z: float/list/tuple:
                Value for slicing. The slicing search for the closest
                value in the domain.

        Returns:
            slice: sliced field.

        Examples:
            # 0D: Get the value of the field in iphi = 0, itheta = -1 and close to r = 0.82
            >>> gasvz.slice(iphi=0,itheta=-1,r=0.82)

            # 1D: Get all values of the field in radial direction at iphi = 0, itheta = -1
            >>> gasvz.slice(iphi=0,itheta=-1)

            # 2D: Get all values of the field for values close to phi = 0
            >>> gasvz.slice(phi=0)
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
                if verbose:
                    print(f"Index condition {index} for coordinate {coord}")
                ivar[COORDS_MAP[self.coordinates][coord]] = index
            else:
                if verbose:
                    print(f"Numeric condition found for coordinate {key}")
                if key in self.domains.keys():
                    # Check if item is a list
                    if isinstance(item,list) or isinstance(item,tuple):
                        if verbose:
                            print(f"You pass the range '{item}' for coordinate {key}")
                        min = abs(self.domains.item(key)-item[0]).argmin()
                        max = abs(self.domains.item(key)-item[1]).argmin()
                        if (min > max) or (min == max):
                            extrema = self.domains.extrema[key]
                            vmin, vmax = extrema[0][1], extrema[1][1]
                            raise ValueError(f"The range provided for '{key}', ie. '{item}' is not valid. You must provide a valid range for the variable with range: [{vmin},{vmax}]")
                        ivar[COORDS_MAP[self.coordinates][key]] = f"{min}:{max}"
                    else:
                        # Check if value provided is in range
                        domain = self.domains.item(key)
                        extrema = self.domains.extrema[key]
                        min, max = extrema[0][1], extrema[1][1]
                        if (item<min) or (item>max):
                            raise ValueError(f"You are attempting to get a slice in {key} = {item}, but the valid range for this variable is [{min},{max}]")
                        find = abs(self.domains.item(key) - item)
                        ivar[COORDS_MAP[self.coordinates][key]] = find.argmin()
                    if verbose:
                        print(f"Range for {key}: {ivar[COORDS_MAP[self.coordinates][key]]}")
                    
        pattern_str = f"{ivar['z']},{ivar['y']},{ivar['x']}"

        if self.type == 'scalar':
            slice_cmd = f"self.data[{pattern_str}]"
            if verbose:
                print(f"Slice: {slice_cmd}")
            slice = eval(slice_cmd)

        elif self.type == 'vector':
            slice = np.array(
                [eval(f"self.data[0,{pattern_str}]"),
                 eval(f"self.data[1,{pattern_str}]"),
                 eval(f"self.data[2,{pattern_str}]")]
            )

        if pattern:
            return slice,pattern_str
        return slice

    def to_cartesian(self):
        if self.type == 'scalar':
            # Scalar fields are invariant under coordinate transformations
            return self
        elif self.type == 'vector':
            # Vector fields must be transformed according to domain
            if self.coordinates == 'cartesian':
                return self
            
            if self.coordinates == 'cylindrical':
                z,r,phi = np.meshgrid(self.domains.z,self.domains.r,self.domains.phi,indexing='ij')
                vphi = self.data[0]
                vr = self.data[1]
                if self.data.shape[0] == 3:
                    vz = self.data[2]
                else:
                    vz = np.zeros_like(vr)
                vx = vr*np.cos(phi) 
                vy = vr*np.sin(phi)
                
                return (Field(vx,coordinates=self.coordinates,domains=self.domains,type='scalar'),
                        Field(vy,coordinates=self.coordinates,domains=self.domains,type='scalar'),
                        Field(vz,coordinates=self.coordinates,domains=self.domains,type='scalar'))
            
            if self.coordinates == 'spherical':

                theta,r,phi = np.meshgrid(self.domains.theta,self.domains.r,self.domains.phi,indexing='ij')
                vphi = self.data[0]
                vr = self.data[1]
                vtheta = self.data[2]

                vx = vr*np.sin(theta)*np.cos(phi) + vtheta*np.cos(theta)*np.cos(phi) - vphi*np.sin(phi)
                vy = vr*np.sin(theta)*np.sin(phi) + vtheta*np.cos(theta)*np.sin(phi) + vphi*np.cos(phi)
                vz = vr*np.cos(theta) - vtheta*np.sin(theta)

                return (Field(vx,coordinates=self.coordinates,domains=self.domains,type='scalar'),
                        Field(vy,coordinates=self.coordinates,domains=self.domains,type='scalar'),
                        Field(vz,coordinates=self.coordinates,domains=self.domains,type='scalar'))
            
    def get_size(self):
        return self.data.nbytes/1024**2

    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)


class FieldInterpolate:
    def __init__(self, sim):
        self.sim = sim

    def load_data(self, field=None, slice=None, snapshots=None):
        self.field = field
        self.slice_definition = slice
        self.slice=slice
        

        """
        Loads data in 2D or 3D depending on the provided parameters.

        Parameters:
            field (list of str, optional): List of fields to load (e.g., ["gasdens", "gasv"]).
            slice (str, optional): Slice definition, e.g., "phi=0", "theta=45", or "z=0,r=[0.8,1.2],phi=[-10 deg,10 deg]".
            snapshots (list or int, optional): List of snapshot indices or a single snapshot to load. Required for both 2D and 3D.
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        if field is None:
            raise ValueError("You must specify at least one field to load using the 'fields' parameter.")

        # Validate and parse the slice parameter
        slice_type = None
        if slice:
            slice = slice.lower()  # Normalize to lowercase for consistency
            if "theta" in slice:
                slice_type = "theta"
            elif "phi" in slice:
                slice_type = "phi"
            else:
                raise ValueError("The 'slice' parameter must contain 'theta' or 'phi'.")

        if not isinstance(snapshots, (int, list, tuple)):
            raise ValueError("'snapshots' must be an integer, a list, or a tuple.")

        if isinstance(snapshots, (list, tuple)) and len(snapshots) == 2:
            if snapshots[0] > snapshots[1]:
                raise ValueError("The range in 'snapshots' is invalid. The first value must be less than or equal to the second.")

        if not hasattr(self.sim, "domains") or self.sim.domains is None:
            raise ValueError("Simulation domains are not loaded. Ensure the simulation data is properly initialized.")

        # Convert a single snapshot to a list
        if isinstance(snapshots, int):
            snapshots = [snapshots]

        # Handle the case where snapshots is a single value or a list with one value
        if len(snapshots) == 1:
            snaps = snapshots
            time_values = [0]  # Single snapshot corresponds to a single time value
        else:
            snaps = np.arange(snapshots[0], snapshots[1] + 1)
            time_values = np.linspace(0, 1, len(snaps))

        if slice:  # Load 2D data
            # Dynamically create DataFrame columns based on the fields
            columns = ['snapshot', 'time', 'var1_mesh', 'var2_mesh']
            if field == "gasdens":
                print(f'Loading 2D density data for slice: {slice}.')
                columns.append('gasdens_mesh')
            if field == "gasv":
                columns.append('gasv_mesh')
                print(f'Loading 2D gas velocity data for slice: {slice}.')
            if field == 'gasenergy':
                columns.append('gasenergy_mesh')
                print(f'Loading 2D gas energy data for slice {slice}')
            df_snapshots = pd.DataFrame(columns=columns)

            for i, snap in enumerate(snaps):
                row = {'snapshot': snap, 'time': time_values[i]}

                # Assign coordinates for all fields
                if field == 'gasdens':
                    gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')
                    gasd_slice, mesh = gasd.meshslice(slice=slice)
                    if slice_type == "phi":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "z")
                    elif slice_type == "theta":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "y")
                    row['gasdens_mesh'] = gasd_slice

                if field == "gasv":
                    gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                    gasvx, gasvy, gasvz = gasv.to_cartesian()
                    vel1_slice, mesh = getattr(gasvx, f'meshslice')(slice=slice)
                    vel2_slice, mesh = getattr(gasvy, f'meshslice')(slice=slice)

                    if slice_type == "phi":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "z")
                    elif slice_type == "theta":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "y")

                    row['gasv_mesh'] = np.array([vel1_slice, vel2_slice])

                if field == "gasenergy":
                    gasenergy = self.sim.load_field('gasenergy', snapshot=snap, type='scalar')
                    gasenergy_slice, mesh = gasenergy.meshslice(slice=slice)
                    row["gasenergy_mesh"] = gasenergy_slice
                    if slice_type == "phi":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "z")
                    elif slice_type == "theta":
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh, "x"), getattr(mesh, "y")

                # Convert the row to a DataFrame and concatenate it
                row_df = pd.DataFrame([row])
                df_snapshots = pd.concat([df_snapshots, row_df], ignore_index=True)

            self.df = df_snapshots
            return df_snapshots

        elif slice is None:  # Load 3D data
            # Generate 3D mesh
            theta, r, phi = np.meshgrid(self.sim.domains.theta, self.sim.domains.r, self.sim.domains.phi, indexing='ij')
            x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)

            # Dynamically create DataFrame columns based on the fields
            columns = ['snapshot', 'time', 'var1_mesh', 'var2_mesh', 'var3_mesh']
            if field == "gasdens":
                print(f'Loading 3D density data ')
                columns.append('gasdens_mesh')
            if field == "gasv":
                columns.append('gasv_mesh')
                print(f'Loading 3D gas velocity data')
            if field == 'gasenergy':
                columns.append('gasenergy_mesh')
                print(f'Loading 3D gas energy data')

            df_snapshots = pd.DataFrame(columns=columns)

            for i, snap in enumerate(snaps):
                row = {'snapshot': snap, 'time': time_values[i]}

                # Assign coordinates for all fields
                if field == 'gasdens':
                    gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')
                    row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z
                    row['gasdens_mesh'] = gasd.data
               
                if field == "gasv":
                    gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                    gasvx, gasvy, gasvz = gasv.to_cartesian()
                    row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z
                    row['gasv_mesh'] = np.array([gasvx.data, gasvy.data, gasvz.data])

                if field == "gasenergy":
                    gasenergy = self.sim.load_field('gasenergy', snapshot=snap, type='scalar')
                    row["gasenergy_mesh"] = gasenergy.data
                    row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z

                # Convert the row to a DataFrame and concatenate it
                row_df = pd.DataFrame([row])
                df_snapshots = pd.concat([df_snapshots, row_df], ignore_index=True)

            self.df = df_snapshots
            return df_snapshots

    def create_mesh(self, slice_definition, radial_divisions=100, angular_divisions=100):
        """
        Create a mesh grid based on the slice definition provided by the user.

        Parameters:
            slice_definition (str): The slice definition string (e.g., "r=[0.8,1.2],phi=0,theta=[0 deg,90 deg]").
            radial_divisions (int): Number of radial divisions for the mesh.
            angular_divisions (int): Number of angular divisions for the mesh.

        Returns:
            tuple: Mesh grid (x, y, z) based on the slice definition.
        """
        import numpy as np
        import re

        # Initialize default ranges
        r_range = [self.sim.domains.r.min(), self.sim.domains.r.max()]
        phi_range = [0, 2 * np.pi]
        theta_range = None
        z = None

        # Regular expressions to extract parameters
        range_pattern = re.compile(r"(\w+)=\[(.+?)\]")  # Matches ranges like r=[0.8,1.2]
        value_pattern = re.compile(r"(\w+)=([-\d.]+)")  # Matches single values like phi=0 or z=0
        degree_pattern = re.compile(r"([-\d.]+) deg")  # Matches angles in degrees like -25 deg

        # Process ranges
        for match in range_pattern.finditer(slice_definition):
            key, values = match.groups()
            values = [float(degree_pattern.sub(lambda m: str(float(m.group(1)) * np.pi / 180), v.strip())) for v in values.split(',')]
            if key == 'r':
                r_range = values
            elif key == 'phi':
                phi_range = values
            elif key == 'theta':
                theta_range = values

        # Process single values
        for match in value_pattern.finditer(slice_definition):
            key, value = match.groups()
            value = float(degree_pattern.sub(lambda m: str(float(m.group(1)) * np.pi / 180), value))
            if key == 'z':
                z = value
            elif key == 'phi':  # Handle single phi values
                phi_range = [value, value]
            elif key == 'theta':  # Handle single theta values
                theta_range = [value, value]

        # Generate the mesh
        r = np.linspace(r_range[0], r_range[1], radial_divisions)

        if theta_range is not None and phi_range is not None:  # Slice in spherical coordinates
            theta = np.linspace(theta_range[0], theta_range[1], angular_divisions)
            phi = np.linspace(phi_range[0], phi_range[1], angular_divisions)
            theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')
            phi_grid = np.full_like(theta_grid, phi_range[0])  # Fix phi if it's a single value

            x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
            y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
            z = r_grid * np.cos(theta_grid)

        elif z is not None:  # Slice in the X-Y plane (z constant)
            phi = np.linspace(phi_range[0], phi_range[1], angular_divisions)
            r_grid, phi_grid = np.meshgrid(r, phi, indexing='ij')
            x = r_grid * np.cos(phi_grid)
            y = r_grid * np.sin(phi_grid)
            z = np.full_like(x, z)

        elif phi_range is not None:  # Slice in the X-Z plane (phi constant)
            theta = np.linspace(theta_range[0], theta_range[1], angular_divisions)
            r_grid, theta_grid = np.meshgrid(r, theta, indexing='ij')
            x = r_grid * np.sin(theta_grid) * np.cos(phi_range[0])
            z = r_grid * np.cos(theta_grid)
            y = np.full_like(x, 0)

        else:
            raise ValueError("Slice definition must include either 'z', 'phi', or 'theta'.")

        return x, y, z


    def evaluate(self, time, var1, var2=None, var3=None, method="griddata"):
        """
        Interpolates a field in 1D, 2D, or 3D using RBFInterpolator or griddata with parallelization.
        Supports both grids and discrete points.

        Parameters:
            time (float): Time at which to interpolate.
            var1 (numpy.ndarray or float): Spatial coordinate for 1D interpolation or the first coordinate for 2D/3D.
            var2 (numpy.ndarray or float, optional): Second spatial coordinate for 2D/3D interpolation.
            var3 (numpy.ndarray or float, optional): Third spatial coordinate for 3D interpolation. If None, 2D is assumed.
            method (str): Interpolation method, either "rbf" or "griddata". Default is "griddata".

        Returns:
            numpy.ndarray or float: Interpolated field values at the given coordinates.
                                    If velocity fields are present, returns a tuple (vx, vy, vz), (vx, vy), or a scalar.
        """
        if method not in ["rbf", "griddata"]:
            raise ValueError("Invalid method. Choose either 'rbf' or 'griddata'.")

        # Automatically determine the field to interpolate
        if "gasdens_mesh" in self.df.columns:
            field_name = "gasdens_mesh"
        elif "gasenergy_mesh" in self.df.columns:
            field_name = "gasenergy_mesh"
        elif "gasv_mesh" in self.df.columns:  # Velocity field
            field_name = "gasv_mesh"
        else:
            raise ValueError("No valid field found in the DataFrame for interpolation.")

        # Sort the DataFrame by time
        df_sorted = self.df.sort_values("time")
        idx = df_sorted["time"].searchsorted(time) - 1
        if idx == -1:
            idx = 0
        idx_after = min(idx + 1, len(df_sorted) - 1)

        t0, t1 = df_sorted.iloc[idx]["time"], df_sorted.iloc[idx_after]["time"]
        factor = (time - t0) / (t1 - t0) if abs(t1 - t0) > 1e-10 else 0
        factor = max(0, min(factor, 1))  # Ensure factor is within [0, 1]

        # Check if the input is a single point or a mesh
        is_scalar = np.isscalar(var1) and (var2 is None or np.isscalar(var2)) and (var3 is None or np.isscalar(var3))
        result_shape = () if is_scalar else var1.shape

        def rbf_interp(coords, values, xi):
            """
            Perform RBF interpolation for 1D, 2D, or 3D data.
            """
            interpolator = RBFInterpolator(coords, values.ravel(), smoothing=1e-6)
            interpolated = interpolator(xi)
            return interpolated

        def griddata_interp(coords, values, xi):
            """
            Perform griddata interpolation for 1D, 2D, or 3D data.
            """
            interpolated = griddata(coords, values.ravel(), xi, method="linear", fill_value=np.nan)
            return interpolated

        def interp(idx, field, component=None):
            if var2 is None and var3 is None:  # 1D interpolation
                coord_x = np.array(df_sorted.iloc[idx]["var1_mesh"])
                if field == "gasv_mesh" and component is not None:
                    data = np.array(df_sorted.iloc[idx][field])[component]
                else:
                    data = np.array(df_sorted.iloc[idx][field])
                coords = coord_x.reshape(-1, 1)
                xi = var1.reshape(-1, 1) if not is_scalar else np.array([[var1]])
                if method == "rbf":
                    return rbf_interp(coords, data, xi)
                else:
                    return griddata_interp(coords, data, xi)
            elif var3 is not None:  # 3D interpolation
                coord_x = np.array(df_sorted.iloc[idx]["var1_mesh"])
                coord_y = np.array(df_sorted.iloc[idx]["var2_mesh"])
                coord_z = np.array(df_sorted.iloc[idx]["var3_mesh"])
                if field == "gasv_mesh" and component is not None:
                    data = np.array(df_sorted.iloc[idx][field])[component]
                else:
                    data = np.array(df_sorted.iloc[idx][field])
                coords = np.column_stack((coord_x.ravel(), coord_y.ravel(), coord_z.ravel()))
                xi = np.column_stack((var1.ravel(), var2.ravel(), var3.ravel()))
                if method == "rbf":
                    return rbf_interp(coords, data, xi)
                else:
                    return griddata_interp(coords, data, xi)
            else:  # 2D interpolation
                coord1 = np.array(df_sorted.iloc[idx]["var1_mesh"])
                coord2 = np.array(df_sorted.iloc[idx]["var2_mesh"])
                if field == "gasv_mesh" and component is not None:
                    data = np.array(df_sorted.iloc[idx][field])[component]
                else:
                    data = np.array(df_sorted.iloc[idx][field])
                coords = np.column_stack((coord1.ravel(), coord2.ravel()))
                xi = np.column_stack((var1.ravel(), var2.ravel()))
                if method == "rbf":
                    return rbf_interp(coords, data, xi)
                else:
                    return griddata_interp(coords, data, xi)

        # Parallelize the interpolation for velocity fields
        if field_name == "gasv_mesh":
            components = 3 if var3 is not None else 2 if var2 is not None else 1
            results = Parallel(n_jobs=-1)(delayed(lambda i: (
                (1 - factor) * interp(idx, field_name, component=i) +
                factor * interp(idx_after, field_name, component=i)
            ))(i) for i in range(components))
            return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])

        # Parallelize the interpolation for scalar fields
        if field_name in ["gasdens_mesh", "gasenergy_mesh"]:
            interpolated = Parallel(n_jobs=-1)(delayed(lambda idx: (
                (1 - factor) * interp(idx, field_name) +
                factor * interp(idx_after, field_name)
            ))(idx) for idx in [idx, idx_after])
            result = interpolated[0] + factor * (interpolated[1] - interpolated[0])
            return result.item() if is_scalar else result.reshape(result_shape)

        # Handle other cases (fallback)
        interpolated = (1 - factor) * interp(idx, field_name) + factor * interp(idx_after, field_name)
        return interpolated.item() if is_scalar else interpolated.reshape(result_shape)