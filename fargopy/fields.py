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
from scipy.interpolate import interp1d
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


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


class FieldInterpolator:
    def __init__(self, sim):
        self.sim = sim
        self.snapshot_time_table = None

    def load_data(self, field=None, slice=None, snapshots=None):
        self.field = field
        self.slice=slice
        
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

        # Guarda la tabla como DataFrame
        self.snapshot_time_table = pd.DataFrame({
            "Snapshot": snaps,
            "Normalized_time": time_values
        })

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
                    if slice_type == "phi":
                        # Plane XZ: use vx and vz
                        vel1_slice, mesh1 = getattr(gasvx, f'meshslice')(slice=slice)
                        vel2_slice, mesh2 = getattr(gasvz, f'meshslice')(slice=slice)
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh1, "x"), getattr(mesh1, "z")
                        row['gasv_mesh'] = np.array([vel1_slice, vel2_slice])
                    elif slice_type == "theta":
                        # Plane XY: use vx and vy
                        vel1_slice, mesh1 = getattr(gasvx, f'meshslice')(slice=slice)
                        vel2_slice, mesh2 = getattr(gasvy, f'meshslice')(slice=slice)
                        row["var1_mesh"], row["var2_mesh"] = getattr(mesh1, "x"), getattr(mesh1, "y")
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
        
    def times(self):
        if self.snapshot_time_table is None:
            raise ValueError("No data loaded. Run load_data() first.")
        return self.snapshot_time_table

    def create_mesh(
        self,
        slice=None,
        nr=50,
        ntheta=50,
        nphi=50
    ):
        """
        Create a mesh grid based on the slice definition provided by the user.
        If no slice is provided, create a full 3D mesh within the simulation domain.

        Parameters:
            slice (str, optional): The slice definition string (e.g., "r=[0.8,1.2],phi=0,theta=[0 deg,90 deg]").
            nr (int): Number of divisions in r.
            ntheta (int): Number of divisions in theta.
            nphi (int): Number of divisions in phi.

        Returns:
            tuple: Mesh grid (x, y, z) based on the slice definition or the full domain.
        """
        import numpy as np
        import re

        # If no slice is provided, create a full 3D mesh using the simulation domains
        if not slice:
            r = np.linspace(self.sim.domains.r.min(), self.sim.domains.r.max(), nr)
            theta = np.linspace(self.sim.domains.theta.min(), self.sim.domains.theta.max(), ntheta)
            phi = np.linspace(self.sim.domains.phi.min(), self.sim.domains.phi.max(), nphi)
            theta_grid, r_grid, phi_grid = np.meshgrid(theta, r, phi, indexing='ij')
            x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
            y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
            z = r_grid * np.cos(theta_grid)
            return x, y, z

        # Initialize default ranges
        r_range = [self.sim.domains.r.min(), self.sim.domains.r.max()]
        theta_range = [self.sim.domains.theta.min(), self.sim.domains.theta.max()]
        phi_range = [self.sim.domains.phi.min(), self.sim.domains.phi.max()]
        z_value = None

        # Regular expressions to extract parameters
        range_pattern = re.compile(r"(\w+)=\[(.+?)\]")  # Matches ranges like r=[0.8,1.2]
        value_pattern = re.compile(r"(\w+)=([-\d.]+)")  # Matches single values like phi=0 or z=0
        degree_pattern = re.compile(r"([-\d.]+) deg")   # Matches angles in degrees like -25 deg

        # Process ranges
        for match in range_pattern.finditer(slice):
            key, values = match.groups()
            values = [float(degree_pattern.sub(lambda m: str(float(m.group(1)) * np.pi / 180), v.strip())) for v in values.split(',')]
            if key == 'r':
                r_range = values
            elif key == 'phi':
                phi_range = values
            elif key == 'theta':
                theta_range = values

        # Process single values
        for match in value_pattern.finditer(slice):
            key, value = match.groups()
            value = float(degree_pattern.sub(lambda m: str(float(m.group(1)) * np.pi / 180), value))
            if key == 'z':
                z_value = value
            elif key == 'phi':
                phi_range = [value, value]
            elif key == 'theta':
                theta_range = [value, value]

        # 3D mesh: all ranges are intervals
        if (phi_range[0] != phi_range[1]) and (theta_range[0] != theta_range[1]):
            r = np.linspace(r_range[0], r_range[1], nr)
            theta = np.linspace(theta_range[0], theta_range[1], ntheta)
            phi = np.linspace(phi_range[0], phi_range[1], nphi)
            theta_grid, r_grid, phi_grid = np.meshgrid(theta, r, phi, indexing='ij')
            x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
            y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
            z = r_grid * np.cos(theta_grid)
            return x, y, z

        # 2D mesh: one angle is fixed (slice)
        elif phi_range[0] == phi_range[1]:  # Slice at constant phi (XZ plane)
            r = np.linspace(r_range[0], r_range[1], nr)
            theta = np.linspace(theta_range[0], theta_range[1], ntheta)
            phi = phi_range[0]
            theta_grid, r_grid = np.meshgrid(theta, r, indexing='ij')
            x = r_grid * np.sin(theta_grid) * np.cos(phi)
            y = r_grid * np.sin(theta_grid) * np.sin(phi)
            z = r_grid * np.cos(theta_grid)
            return x, y, z

        elif theta_range[0] == theta_range[1]:  # Slice at constant theta (XY plane)
            r = np.linspace(r_range[0], r_range[1], nr)
            phi = np.linspace(phi_range[0], phi_range[1], nphi)
            theta = theta_range[0]
            phi_grid, r_grid = np.meshgrid(phi, r, indexing='ij')
            x = r_grid * np.sin(theta) * np.cos(phi_grid)
            y = r_grid * np.sin(theta) * np.sin(phi_grid)
            z = r_grid * np.cos(theta)
            return x, y, z

        elif z_value is not None:  # Slice at constant z (XY plane in cartesian)
            r = np.linspace(r_range[0], r_range[1], nr)
            phi = np.linspace(phi_range[0], phi_range[1], nphi)
            r_grid, phi_grid = np.meshgrid(r, phi, indexing='ij')
            x = r_grid * np.cos(phi_grid)
            y = r_grid * np.sin(phi_grid)
            z = np.full_like(x, z_value)
            return x, y, z

        else:
            raise ValueError("Slice definition must include either 'z', 'phi', or 'theta'.")

    def evaluate(
            self, time, var1, var2=None, var3=None,
            interpolator="griddata", method="linear",
            rbf_kwargs=None, griddata_kwargs=None, idw_kwargs=None
        ):
        """
        Interpolates a field in 1D, 2D, or 3D using RBFInterpolator, griddata, LinearNDInterpolator, or IDW.
        Supports both grids and discrete points.

        Parameters:
            ...
            interpolator (str): Interpolation family, either "rbf", "griddata", "linearnd", or "idw". Default is "griddata".
            idw_kwargs (dict): Optional kwargs for IDW, e.g. {'power': 2, 'k': 8}
            ...
        """


        # --- Handle time input: explicit and robust: normalized time [0,1] or snapshot index ---
        if hasattr(self, "snapshot_time_table") and self.snapshot_time_table is not None:
            snaps = self.snapshot_time_table["Snapshot"].values
            min_snap, max_snap = snaps.min(), snaps.max()
            # If time is float in [0,1], treat as normalized time (directly)
            if isinstance(time, float) and 0 <= time <= 1:
                pass  # Use as normalized time directly
            # If time is int or float > 1, treat as snapshot or fractional snapshot
            elif (isinstance(time, int) or (isinstance(time, float) and time > 1)):
                if time < min_snap or time > max_snap:
                    raise ValueError(
                        f"Selected snapshot (time={time}) is outside the loaded range [{min_snap}, {max_snap}]."
                    )
                if isinstance(time, int) or np.isclose(time, np.round(time)):
                    # Exact snapshot
                    row = self.snapshot_time_table[self.snapshot_time_table["Snapshot"] == int(round(time))]
                    if not row.empty:
                        time = float(row["Normalized_time"].values[0])
                    else:
                        raise ValueError(f"Snapshot {int(round(time))} not found in snapshot_time_table.")
                else:
                    # Fractional snapshot: interpolate between neighbors
                    snap0 = int(np.floor(time))
                    snap1 = int(np.ceil(time))
                    if snap0 < min_snap or snap1 > max_snap:
                        raise ValueError(
                            f"Selected snapshot (time={time}) requires neighbors [{snap0}, {snap1}] outside the loaded range [{min_snap}, {max_snap}]."
                        )
                    row0 = self.snapshot_time_table[self.snapshot_time_table["Snapshot"] == snap0]
                    row1 = self.snapshot_time_table[self.snapshot_time_table["Snapshot"] == snap1]
                    if not row0.empty and not row1.empty:
                        t0 = float(row0["Normalized_time"].values[0])
                        t1 = float(row1["Normalized_time"].values[0])
                        factor = (time - snap0) / (snap1 - snap0)
                        time = (1 - factor) * t0 + factor * t1
                    else:
                        raise ValueError(f"Snapshots {snap0} or {snap1} not found in snapshot_time_table.")
            else:
                raise ValueError(
                    f"Invalid time value: {time}. Must be a normalized time in [0,1] or a snapshot index in [{min_snap},{max_snap}]."
                )
        else:
            if isinstance(time, int):
                raise ValueError("snapshot_time_table not found. Did you call load_data()?")

        if interpolator not in ["rbf", "griddata", "linearnd","idw"]:
            raise ValueError("Invalid method. Choose either 'rbf', 'griddata', 'idw', or 'linearnd'.")

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
        times = df_sorted["time"].values
        n_snaps = len(times)

        # Check if the input is a single point or a mesh
        is_scalar = np.isscalar(var1) and (var2 is None or np.isscalar(var2)) and (var3 is None or np.isscalar(var3))
        result_shape = () if is_scalar else var1.shape

        if rbf_kwargs is None:
            rbf_kwargs = {}
        if griddata_kwargs is None:
            griddata_kwargs = {}



        if idw_kwargs is None:
            idw_kwargs = {}


        def idw_interp(coords, values, xi):
            # Forzar a 2D: (N, D) y (M, D)
            coords = np.asarray(coords)
            xi = np.asarray(xi)
            if coords.ndim > 2:
                coords = coords.reshape(-1, coords.shape[-1])
            if xi.ndim > 2:
                xi = xi.reshape(-1, xi.shape[-1])
            values = np.asarray(values).ravel()
            power = idw_kwargs.get('power', 2)
            k = idw_kwargs.get('k', 8)
            tree = cKDTree(coords)
            dists, idxs = tree.query(xi, k=k)
            dists = np.where(dists == 0, 1e-10, dists)
            weights = 1 / dists**power
            weights /= weights.sum(axis=1, keepdims=True)
            return np.sum(values[idxs] * weights, axis=1)

        def rbf_interp(coords, values, xi):
            # Check if epsilon is required for the selected kernel
            kernels_requiring_epsilon = ["gaussian", "multiquadric", "inverse_multiquadric", "inverse_quadratic"]
            if method in kernels_requiring_epsilon and "epsilon" not in rbf_kwargs:
                raise ValueError(f"Kernel '{method}' requires 'epsilon' in rbf_kwargs.")
            interpolator_obj = RBFInterpolator(
                coords, values.ravel(),
                kernel=method,
                **rbf_kwargs
            )
            return interpolator_obj(xi)

        def griddata_interp(coords, values, xi):
                return griddata(coords, values.ravel(), xi, method=method, **griddata_kwargs)

        def linearnd_interp(coords, values, xi):
            interp_obj = LinearNDInterpolator(coords, values.ravel())
            return interp_obj(xi)

        def interp(idx, field, component=None):
            if var2 is None and var3 is None:  # 1D interpolation
                coord_x = np.array(df_sorted.iloc[idx]["var1_mesh"])
                if field == "gasv_mesh" and component is not None:
                    data = np.array(df_sorted.iloc[idx][field])[component]
                else:
                    data = np.array(df_sorted.iloc[idx][field])
                coords = coord_x.reshape(-1, 1)
                xi = var1.reshape(-1, 1) if not is_scalar else np.array([[var1]])
                if interpolator == "rbf":
                    return rbf_interp(coords, data, xi)
                elif interpolator == "linearnd":
                    return linearnd_interp(coords, data, xi)
                elif interpolator == "idw":
                    return idw_interp(coords, data, xi)
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
                if interpolator == "rbf":
                    return rbf_interp(coords, data, xi)
                elif interpolator == "linearnd":
                    return linearnd_interp(coords, data, xi)
                elif interpolator == "idw":
                    return idw_interp(coords, data, xi)
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
                if interpolator == "rbf":
                    return rbf_interp(coords, data, xi)
                elif interpolator == "linearnd":
                    return linearnd_interp(coords, data, xi)
                elif interpolator == "idw":
                    return idw_interp(coords, data, xi)
                else:
                    return griddata_interp(coords, data, xi)
                
        
        # --- Case 1: only a snapshot ---
        if n_snaps == 1:
            def eval_single(component=None):
                return interp(0, field_name, component)
            if field_name == "gasv_mesh":
                components = 3 if var3 is not None else 2 if var2 is not None else 1
                results = Parallel(n_jobs=-1)(
                    delayed(eval_single)(i) for i in range(components)
                )
                return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])
            else:
                # Trivial escalar case: parallelization over the single snapshot
                result = Parallel(n_jobs=-1)([delayed(eval_single)()])
                result = result[0]
                return result.item() if is_scalar else result.reshape(result_shape)

        # --- Case 2: Two snapshots, linear temporal interpolation ---
        elif n_snaps == 2:
            idx, idx_after = 0, 1
            t0, t1 = times[idx], times[idx_after]
            factor = (time - t0) / (t1 - t0) if abs(t1 - t0) > 1e-10 else 0
            factor = max(0, min(factor, 1))
            def temporal_interp(component=None):
                val0 = interp(idx, field_name, component)
                val1 = interp(idx_after, field_name, component)
                return (1 - factor) * val0 + factor * val1
            if field_name == "gasv_mesh":
                components = 3 if var3 is not None else 2 if var2 is not None else 1
                results = Parallel(n_jobs=-1)(
                    delayed(temporal_interp)(i) for i in range(components)
                )
                return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])
            else:
                # Escalar: paralelización sobre ambos snapshots
                results = Parallel(n_jobs=2)(
                    delayed(temporal_interp)() for _ in range(1)
                )
                result = results[0]
                return result.item() if is_scalar else result.reshape(result_shape)

        # --- Case 3: More than two snapshots, spline temporal interpolation ---
        else:
            def eval_all_snaps(component=None):
                return Parallel(n_jobs=-1)(
                    delayed(interp)(i, field_name, component) for i in range(n_snaps)
                )
            if field_name == "gasv_mesh":
                components = 3 if var3 is not None else 2 if var2 is not None else 1
                results = []
                for comp in range(components):
                    values = eval_all_snaps(component=comp)
                    values = np.stack([v if not is_scalar else np.array([v]) for v in values], axis=0)
                    f = interp1d(times, values, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
                    res = f(time)
                    results.append(res.item() if is_scalar else res.reshape(result_shape))
                return np.array(results)
            else:
                values = eval_all_snaps()
                values = np.stack([v if not is_scalar else np.array([v]) for v in values], axis=0)
                f = interp1d(times, values, axis=0, kind='linear', bounds_error=False, fill_value=np.nan)
                result = f(time)
                return result.item() if is_scalar else result.reshape(result_shape)