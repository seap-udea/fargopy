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
from pathlib import Path
import fargopy as fp

###############################################################
# Constants
###############################################################
# Map of coordinates into FARGO3D coordinates
# This dictionary maps the coordinates regular names (r, phi, theta, etc.) of
# different coordinate systems into the FARGO3D x, y, z

COORDS_MAP = dict(
    cartesian = dict(x='x',y='y',z='z'),
    cylindrical = dict(phi='x',r='y',z='z'),
    spherical = dict(phi='x',r='y',theta='z'),
)

###############################################################
# Classes
###############################################################
class Field(fargopy.Fargobj):
    """
    Represents a simulation field (scalar or vector) with coordinate system and domain information.

    Attributes
    ----------
    data : np.ndarray
        Numpy array with the field data.
    coordinates : str
        Type of coordinates ('cartesian', 'cylindrical', 'spherical').
    domains : object
        Domain information for each coordinate.
    type : str
        Field type ('scalar' or 'vector').

    Methods
    -------
    slice :
        Get a slice of the field along a given spatial direction.
    meshslice :
        Perform a slice and return the field slice and associated coordinate matrices for plotting.
    to_cartesian :
        Convert the field to cartesian coordinates (for vector fields).
    get_size :
        Return the size of the field data in MB.
    """

    def __init__(self,data=None,coordinates='cartesian',domains=None,type='scalar',**kwargs):
        """
        Initialize a Field object.

        Parameters
        ----------
        data : np.ndarray, optional
            Field data array.
        coordinates : str, optional
            Coordinate system ('cartesian', 'cylindrical', 'spherical').
        domains : object, optional
            Domain information for each coordinate.
        type : str, optional
            Field type ('scalar' or 'vector').
        **kwargs : dict
            Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.data = data
        self.coordinates = coordinates
        self.domains = domains
        self.type = type

    def meshslice(self,slice=None,component=0,verbose=False):
        """
        Perform a slice on a field and produce the corresponding field slice and
        associated coordinate matrices for plotting.

        Parameters
        ----------
        slice : str
            Slice definition string.
        component : int, optional
            Component index for vector fields (default: 0).
        verbose : bool, optional
            If True, print debug information.

        Returns
        -------
        tuple
            (sliced field, mesh dictionary with coordinates)
        """
        # Analysis of the slice 
        if slice is None:
            raise ValueError("You must provide a slice option.")

        # Degrees specification
        slice = slice.replace('deg','*fargopy.DEG')

        # Perform the slice
        slice_cmd = f"self._slice({slice},pattern=True,verbose={verbose})"
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

    def _slice(self,verbose=False,pattern=False,**kwargs):
        """
        Extract a slice of a 3-dimensional FARGO3D field.

        Parameters
        ----------
        verbose : bool, optional
            If True, print debug information.
        pattern : bool, optional
            If True, return the pattern of the slice (e.g., [:,:,:]).
        ir, iphi, itheta, ix, iy, iz : int or str, optional
            Index or range of indexes for the corresponding coordinate.
        r, phi, theta, x, y, z : float, list, or tuple, optional
            Value or range for slicing. The closest value in the domain is used.

        Returns
        -------
        np.ndarray or tuple
            Sliced field, and optionally the pattern string if pattern=True.

        Examples
        --------
        # 0D: Get the value of the field at iphi=0, itheta=-1, and close to r=0.82
        >>> gasvz.slice(iphi=0, itheta=-1, r=0.82)

        # 1D: Get all values in radial direction at iphi=0, itheta=-1
        >>> gasvz.slice(iphi=0, itheta=-1)

        # 2D: Get all values for values close to phi=0
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
        """
        Convert the field to cartesian coordinates.

        Returns
        -------
        Field or tuple of Field
            The field in cartesian coordinates (for vector fields, returns (vx, vy, vz)).
        """
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
        """
        Return the size of the field data in megabytes (MB).

        Returns
        -------
        float
            Size in MB.
        """
        return self.data.nbytes/1024**2

    def __str__(self):
        """
        String representation of the field data.

        Returns
        -------
        str
        """
        return str(self.data)
    
    def __repr__(self):
        """
        String representation of the field data.

        Returns
        -------
        str
        """
        return str(self.data)


# ###############################################################
# FieldInterpolator
# ###############################################################
# This class is used to load and interpolate fields from a FARGO3D simulation.
# It provides methods to load data, create meshes, and perform interpolation.
# It also handles 2D and 3D data loading based on the provided parameters.
#################################################################


class FieldInterpolator:
    """
    Loads and interpolates fields from a FARGO3D simulation. Provides methods to load data,
    create meshes, and perform interpolation in 1D, 2D, or 3D, supporting various interpolation methods.

    Attributes
    ----------
    sim : Simulation
        The simulation object.
    df : pd.DataFrame or None
        DataFrame containing loaded field data.
    snapshot_time_table : pd.DataFrame or None
        Table mapping snapshots to normalized time.
    snapshot : list or None
        List of loaded snapshots.
    slice : str or None
        Slice definition string.
    dim : int or None
        Dimensionality of the loaded data.
    """

    def __init__(self, sim, df=None):
        """
        Initialize a FieldInterpolator.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        df : pd.DataFrame, optional
            DataFrame with preloaded field data.
        """
        self.sim = sim
        self.snapshot_time_table = None
        self.snapshot = None
        self.slice = None
        self.dim=None
        self.df = df

    def __getattr__(self, name):
        """
        Delegate attribute access to the internal DataFrame if present.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute from the DataFrame if available.

        Raises
        ------
        AttributeError
            If the attribute is not found.
        """
        df = object.__getattribute__(self, 'df')
        if df is not None and hasattr(df, name):
            return getattr(df, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")   

    def load_data(self, field=None, slice=None, snapshots=1, cut=None):
        """
        Load field data in 2D or 3D depending on the provided parameters.

        Parameters
        ----------
        field : str, required
            Name of the field to load (e.g., 'gasdens', 'gasv', 'gasenergy').
        slice : str, optional
            Slice definition (e.g., 'phi=0', 'theta=45', or 'z=0,r=[0.8,1.2],phi=[-10 deg,10 deg]').
        snapshots : int, list, or tuple, optional
            Snapshot index or range to load.
        cut : tuple, optional
            Spatial cut for loading a subset of the field (e.g., for a sphere or cylinder).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the loaded data.
        """
        self.field = field
        self.slice = slice

        # Convert a single snapshot to a list
        if isinstance(snapshots, int):
            snapshots = [snapshots]
        self.snapshot = snapshots


        if slice is not None:
            self.dim = len(self.sim.load_field(
                fields='gasdens',
                slice=self.slice,
                snapshot=snapshots[0]
            ).data.shape)

        else:
            self.dim = 3

        # Handle the case where snapshots is a single value or a list with one value
        if len(snapshots) == 1:

            snaps = snapshots
            time_values = [0]  # Single snapshot corresponds to a single time value
        else:
            snaps = np.arange(snapshots[0], snapshots[1] + 1)
            time_values = np.linspace(0, 1, len(snaps))

        # Save table like dataframe
        self.snapshot_time_table = pd.DataFrame({
            "Snapshot": snaps,
            "Normalized_time": time_values
        })

        """
        Loads data in 2D or 3D depending on the provided parameters.

        Parameters
        ----------
        field (list of str, optional): List of fields to load (e.g., ["gasdens", "gasv"]).
        slice (str, optional): Slice definition, e.g., "phi=0", "theta=45", or "z=0,r=[0.8,1.2],phi=[-10 deg,10 deg]".
        snapshots (list or int, optional): List of snapshot indices or a single snapshot to load. Required for both 2D and 3D.
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        if field is None:
            raise ValueError("You must specify at least one field to load using the 'fields' parameter.")


        if not isinstance(snapshots, (int, list, tuple)):
            raise ValueError("'snapshots' must be an integer, a list, or a tuple.")

        if isinstance(snapshots, (list, tuple)) and len(snapshots) == 2:
            if snapshots[0] > snapshots[1]:
                raise ValueError("The range in 'snapshots' is invalid. The first value must be less than or equal to the second.")

        if not hasattr(self.sim, "domains") or self.sim.domains is None:
            raise ValueError("Simulation domains are not loaded. Ensure the simulation data is properly initialized.")

        def _rotation(X, Y, Z, phi0):
            X_rot =  X * np.cos(phi0) + Y * np.sin(phi0)
            Y_rot = -X * np.sin(phi0) + Y * np.cos(phi0)
            Z_rot = Z.copy()  # z no cambia

            return X_rot, Y_rot, Z_rot


        if self.dim<3:
        
            # Dynamically create DataFrame columns based on the fields
            columns = ['snapshot', 'time', 'var1_mesh', 'var2_mesh', 'var3_mesh']
            if field == "gasdens":
                columns.append('gasdens_mesh')
            if field == "gasv":
                columns.append('gasv_mesh')
            if field == 'gasenergy':
                columns.append('gasenergy_mesh')
            df_snapshots = pd.DataFrame(columns=columns)


            for i, snap in enumerate(snaps):
                row = {'snapshot': snap, 'time': time_values[i]}

                # Assign coordinates for all fields
                if field == 'gasdens':
                    
                    
                    gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')
                    gasd_slice, mesh = gasd.meshslice(slice=slice)
                    
                    if np.all(mesh.phi.ravel()== mesh.phi.ravel()[0]):
                        x_rot, y_rot, z_rot = _rotation(getattr(mesh, 'x'), getattr(mesh, 'y'), getattr(mesh, 'z'), mesh.phi.ravel()[0])
                        row['var1_mesh'] = x_rot
                        row['var2_mesh'] = y_rot
                        row['var3_mesh'] = z_rot
                        row['gasdens_mesh'] = gasd_slice


                    else:
                        row['var1_mesh'] = getattr(mesh, 'x')
                        row['var2_mesh'] = getattr(mesh, 'y')
                        row['var3_mesh'] = getattr(mesh, 'z')
                        row['gasdens_mesh'] = gasd_slice

                if field == "gasv":

                    gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                    gasvx, gasvy, gasvz = gasv.to_cartesian()

                    # Plane XZ: use vx and vz
                    vel1_slice, mesh = getattr(gasvx, f'meshslice')(slice=slice)
                    vel2_slice, mesh = getattr(gasvy, f'meshslice')(slice=slice)
                    vel3_slice, mesh = getattr(gasvz, f'meshslice')(slice=slice)

                    row['var1_mesh'] = getattr(mesh, 'x')
                    row['var2_mesh'] = getattr(mesh, 'y')
                    row['var3_mesh'] = getattr(mesh, 'z')
                    row['gasv_mesh'] = np.array([vel1_slice, vel2_slice, vel3_slice])

                if field == "gasenergy":
                    gasenergy = self.sim.load_field('gasenergy', snapshot=snap, type='scalar')
                    gasenergy_slice, mesh = gasenergy.meshslice(slice=slice)
                    

                    row["var1_mesh"] = getattr(mesh, "x")
                    row["var2_mesh"] = getattr(mesh, "y")
                    row["var3_mesh"] = getattr(mesh, "z")
                    row["gasenergy_mesh"] = gasenergy_slice

                # Convert the row to a DataFrame and concatenate it
                row_df = pd.DataFrame([row])
                to_concat = [df_snapshots, row_df]
                to_concat = [df for df in to_concat if not df.empty and not df.isna().all().all()]
                if to_concat:
                    df_snapshots = pd.concat(to_concat, ignore_index=True)

            self.df = df_snapshots
            return df_snapshots

        if self.dim==3:  # Load 3D data

            # Generate 3D mesh
            theta, r, phi = np.meshgrid(self.sim.domains.theta, self.sim.domains.r, self.sim.domains.phi, indexing='ij')
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # --- NEW: General 3D cut (cylinder or sphere) ---
            if cut is not None:
                mask = np.ones_like(x, dtype=bool)
                if len(cut) == 5:
                    xc, yc, zc, rc, hc = cut
                    r_xy = np.sqrt((x - xc)**2 + (y - yc)**2)
                    z_min = zc - hc/2
                    z_max = zc + hc/2
                    mask = (r_xy <= rc) & (z >= z_min) & (z <= z_max)
                elif len(cut) == 4:
                    xc, yc, zc, rs = cut
                    r_sph = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
                    mask = r_sph <= rs
                else:
                    raise ValueError("The 'cut' argument must have 4 (sphere) or 5 (cylinder) elements.")
            else:
                mask = None  # <--- No mask applied

            columns = ['snapshot', 'time', 'var1_mesh', 'var2_mesh', 'var3_mesh']
            if field == "gasdens":
                columns.append('gasdens_mesh')
            if field == "gasv":
                columns.append('gasv_mesh')
            if field == 'gasenergy':
                columns.append('gasenergy_mesh')

            df_snapshots = pd.DataFrame(columns=columns)

            for i, snap in enumerate(snaps):
                row = {'snapshot': snap, 'time': time_values[i]}
                if field == 'gasdens':
                    gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')
                    if mask is not None:
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x[mask], y[mask], z[mask]
                        row['gasdens_mesh'] = gasd.data[mask]
                    else:
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z
                        row['gasdens_mesh'] = gasd.data
                if field == "gasv":
                    gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                    gasvx, gasvy, gasvz = gasv.to_cartesian()
                    if mask is not None:
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x[mask], y[mask], z[mask]
                        row['gasv_mesh'] = np.array([gasvx.data[mask], gasvy.data[mask], gasvz.data[mask]])
                    else:
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z
                        row['gasv_mesh'] = np.array([gasvx.data, gasvy.data, gasvz.data])
                if field == "gasenergy":
                    gasenergy = self.sim.load_field('gasenergy', snapshot=snap, type='scalar')
                    if mask is not None:
                        row["gasenergy_mesh"] = gasenergy.data[mask]
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x[mask], y[mask], z[mask]
                    else:
                        row["gasenergy_mesh"] = gasenergy.data
                        row["var1_mesh"], row["var2_mesh"], row["var3_mesh"] = x, y, z

                row_df = pd.DataFrame([row])
                to_concat = [df_snapshots, row_df]
                to_concat = [df for df in to_concat if not df.empty and not df.isna().all().all()]
                if to_concat:
                    df_snapshots = pd.concat(to_concat, ignore_index=True)

            self.df = df_snapshots
            return df_snapshots
        
    def times(self):
        """
        Return the snapshot time table mapping snapshots to normalized time.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'Snapshot' and 'Normalized_time'.

        Raises
        ------
        ValueError
            If no data has been loaded.
        """
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

        Parameters
        ----------
        slice : str, optional
            The slice definition string (e.g., "r=[0.8,1.2],phi=0,theta=[0 deg,90 deg]").
        nr : int
            Number of divisions in r.
        ntheta : int
            Number of divisions in theta.
        nphi : int
            Number of divisions in phi.

        Returns
        -------
        tuple
            Mesh grid (x, y, z) based on the slice definition or the full domain.
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



    def _domain_mask(self, xi, slice=None):
        """
        Returns a boolean mask indicating which points in xi (cartesian) are inside the simulation domain,
        for slices in the XY (theta fixed) or XZ (phi fixed) plane.
        Handles both 1D and 2D input.

        Parameters
        ----------
        xi : np.ndarray
            Points to check (shape (N, D) or (N,)).
        slice : str, optional
            Slice definition string.

        Returns
        -------
        np.ndarray
            Boolean mask array.
        """
        xi = np.asarray(xi)
        ndim = xi.shape[1] if xi.ndim > 1 else 1

        # Get domain limits
        r_min = self.sim.domains.r.min()
        r_max = self.sim.domains.r.max()
        theta_min = self.sim.domains.theta.min()
        theta_max = self.sim.domains.theta.max()
        phi_min = self.sim.domains.phi.min()
        phi_max = self.sim.domains.phi.max()

        eps = 1e-7

        if ndim == 2:
            # XY plane: theta fixed
            if slice is not None and 'theta' in slice:
                # XY plane: z = 0, theta fixed, filter by r and phi
                x, y = xi[:, 0], xi[:, 1]
                z = np.zeros_like(x)
                r = np.sqrt(x**2 + y**2 + z**2)
                phi = np.arctan2(y, x)
                mask = (
                    (r >= r_min)  &
                    (r <= r_max)  )
                return mask
            
            
            # XZ plane: phi fixed
            elif slice is not None and 'phi' in slice:
                # XZ plane: y = 0, phi fixed, filter by r and theta
                x, z = xi[:, 0], xi[:, 1]
                y = np.zeros_like(x)
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arccos(z / np.clip(r, 1e-14, None))
                mask = (
                    ((r > r_min) | np.isclose(r, r_min, atol=eps)) &
                    ((r < r_max) | np.isclose(r, r_max, atol=eps)) &
                    ((theta > theta_min) | np.isclose(theta, theta_min, atol=eps)) &
                    ((theta < theta_max) | np.isclose(theta, theta_max, atol=eps))
                )
                return mask
            else:
                # Default: treat as XY (theta fixed)
                x, y = xi[:, 0], xi[:, 1]
                z = np.zeros_like(x)
                r = np.sqrt(x**2 + y**2 + z**2)
                phi = np.arctan2(y, x)
                mask = (
                    (r > r_min) &
                    (r < r_max) )
                return mask

        elif ndim == 1:
            # 1D input: could be r, theta, or phi
            xi_1d = xi
            if slice is not None:
                if 'r' in slice:
                    mask = (xi_1d >= r_min) & (xi_1d <= r_max)
                elif 'theta' in slice:
                    mask = (xi_1d >= theta_min) & (xi_1d <= theta_max)
                elif 'phi' in slice:
                    mask = (xi_1d >= phi_min) & (xi_1d <= phi_max)
                else:
                    mask = (xi_1d >= r_min) & (xi_1d <= r_max)
            else:
                mask = (xi_1d >= r_min) & (xi_1d <= r_max)
                return mask

        if ndim==3:
            mask = np.ones(xi.shape[0],dtype=bool)
            return mask




    def evaluate(
            self, time, var1, var2=None, var3=None, dataframe=None,
            interpolator="griddata", method="linear",
            rbf_kwargs=None, griddata_kwargs=None, idw_kwargs=None
        ):
        """
        Interpolates a field in 1D, 2D, or 3D using RBFInterpolator, griddata, LinearNDInterpolator, or IDW.
        Supports both grids and discrete points.

        Parameters
        ----------
        time : float or int
            Normalized time in [0,1] or snapshot index.
        var1, var2, var3 : np.ndarray or float
            Coordinates at which to interpolate.
        dataframe : pd.DataFrame, optional
            DataFrame to use for interpolation (default: self.df).
        interpolator : str, optional
            Interpolation method ('rbf', 'griddata', 'linearnd', 'idw').
        method : str, optional
            Interpolation kernel or method for the chosen interpolator.
        rbf_kwargs : dict, optional
            Additional kwargs for RBFInterpolator.
        griddata_kwargs : dict, optional
            Additional kwargs for griddata.
        idw_kwargs : dict, optional
            Additional kwargs for IDW.

        Returns
        -------
        np.ndarray
            Interpolated values at the requested coordinates.

        Raises
        ------
        ValueError
            If input parameters are invalid or required data is missing.
        """
        # Use the provided DataFrame or the internal one
        if dataframe is not None:
            dataframe = dataframe
        elif self.df is not None:
            dataframe = self.df
        else:
            raise ValueError("No DataFrame provided and self.df is not set.")
        

        

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
            norm_times = self.snapshot_time_table["Normalized_time"].values
            # If time is float in [0,1], check if it matches an exact snapshot
            if np.issubdtype(type(time), np.floating) and 0 <= time <= 1:
                idx_exact = np.where(np.isclose(norm_times, time, atol=1e-8))[0]
                if len(idx_exact) > 0:
                    # Use exact snapshot, no temporal interpolation
                    time = norm_times[idx_exact[0]]
                # If not exact, continue with normal temporal interpolation
            # If time is int or float > 1, treat as snapshot index or fractional snapshot
            elif np.issubdtype(type(time), np.integer) or (np.issubdtype(type(time), np.floating) and time > 1):
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
        if "gasdens_mesh" in dataframe.columns:
            field_name = "gasdens_mesh"
        elif "gasenergy_mesh" in dataframe.columns:
            field_name = "gasenergy_mesh"
        elif "gasv_mesh" in dataframe.columns:  # Velocity field
            field_name = "gasv_mesh"
        else:
            raise ValueError("No valid field found in the DataFrame for interpolation.")

        # Sort the DataFrame by time
        df_sorted = dataframe.sort_values("time")
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
            # Force to 2D: (N, D) and (M, D)
            coords = np.asarray(coords)
            xi = np.asarray(xi)
            if coords.ndim > 2:
                coords = coords.reshape(-1, coords.shape[-1])
            if xi.ndim > 2:
                xi = xi.reshape(-1, xi.shape[-1])
            values = np.asarray(values).ravel()
            power = idw_kwargs.get('power', 2)
            k = idw_kwargs.get('k', 8)

            # --- Apply domain mask: only interpolate inside the simulation domain ---
            mask = self._domain_mask(xi)  # Boolean mask: True if inside domain, False if outside

            # Prepare output array (default 0 outside domain)
            interp_values = np.zeros(xi.shape[0])

            # Only interpolate where mask is True
            if np.any(mask):
                tree = cKDTree(coords)
                dists, idxs = tree.query(xi[mask], k=k)
                dists = np.where(dists == 0, 1e-10, dists)
                weights = 1 / dists**power
                weights /= weights.sum(axis=1, keepdims=True)
                interp_values[mask] = np.sum(values[idxs] * weights, axis=1)

            return interp_values
        
        def rbf_interp(coords, values, xi):
            xi = np.asarray(xi)
            # Check if epsilon is required for the selected kernel
            kernels_requiring_epsilon = ["gaussian", "multiquadric", "inverse_multiquadric", "inverse_quadratic"]
            if method in kernels_requiring_epsilon and "epsilon" not in rbf_kwargs:
                raise ValueError(f"Kernel '{method}' requires 'epsilon' in rbf_kwargs.")

            # --- Apply domain mask: only interpolate inside the simulation domain ---
            mask = self._domain_mask(xi)  # Boolean mask: True if inside domain, False if outside

            # Prepare output array (default 0 outside domain)
            interp_values = np.zeros(xi.shape[0])

            # Only interpolate where mask is True
            if np.any(mask):
                interpolator_obj = RBFInterpolator(
                    coords, values.ravel(),
                    kernel=method,
                    **rbf_kwargs
                )
                interp_values[mask] = interpolator_obj(xi[mask])

            return interp_values

        
        def griddata_interp(coords, values, xi):
            

            # --- Apply domain mask: only interpolate inside the simulation domain ---
            mask = self._domain_mask(xi)  # Boolean mask: True if inside domain, False if outside

            # Prepare output array (default 0 outside domain)
            interp_values = np.zeros(xi.shape[0])
    
            # Only interpolate where mask is True

            if np.any(mask):
                interp_values[mask] = griddata(coords, values.ravel(), xi[mask], method=method, **griddata_kwargs)
            return interp_values
        
        def linearnd_interp(coords, values, xi):
            xi= np.asarray(xi)
            # --- Apply domain mask: only interpolate inside the simulation domain ---
            mask = self._domain_mask(xi)  # Boolean mask: True if inside domain, False if outside

            # Prepare output array (default 0 outside domain)
            interp_values = np.zeros(xi.shape[0])

            # Only interpolate where mask is True
            if np.any(mask):
                interp_obj = LinearNDInterpolator(coords, values.ravel())
                interp_values[mask] = interp_obj(xi[mask])

            return interp_values

        # --- Prepare the mesh for interpolation ---
        # New logic for slice_type:
        # - If theta is a single value (not in brackets), slice_type='theta'
        # - If phi is a single value (not in brackets), slice_type='phi'
        # - If both theta and phi are single values (not in brackets), slice_type='r' (1D cut in r)
        # - Otherwise, None

        slice_type = None
        if self.slice:
            slice_str = self.slice.replace(" ", "").lower()
            import re
            # Match theta=number (not in brackets)
            m_theta = re.search(r"theta=([^\[\],]+)(?![\]])", slice_str)
            m_phi = re.search(r"phi=([^\[\],]+)(?![\]])", slice_str)
            m_theta_bracket = re.search(r"theta=\[", slice_str)
            m_phi_bracket = re.search(r"phi=\[", slice_str)
            # Both theta and phi are fixed (not in brackets): 1D cut in r
            if m_theta and not m_theta_bracket and m_phi and not m_phi_bracket:
                slice_type = "r"
            # Only theta is fixed (not in brackets)
            elif m_theta and not m_theta_bracket:
                slice_type = "theta"
            # Only phi is fixed (not in brackets)
            elif m_phi and not m_phi_bracket:
                slice_type = "phi"
            else:
                slice_type = None
        

        # --- Prepare the points variables for interpolation ---
        if np.isscalar(var1):
            var1 = np.array([var1])
        if np.isscalar(var2):
            var2 = np.array([var2])
        if np.isscalar(var3):
            var3 = np.array([var3])


        def interp(idx, field, component=None):
            """Interpolates the field at the specified index (snapshot) and returns the interpolated values.
            If the field is "gasv_mesh" and a component is specified, it returns only that component.
            """
            

            coord_x = np.array(df_sorted.iloc[idx]["var1_mesh"])
            coord_y = np.array(df_sorted.iloc[idx]["var2_mesh"])
            coord_z = np.array(df_sorted.iloc[idx]["var3_mesh"])
            
            if self.dim == 3:
                coords=np.column_stack((
                    coord_x.ravel(),
                    coord_y.ravel(),
                    coord_z.ravel()))
                xi = np.column_stack((
                    var1.ravel(),
                    var2.ravel(),
                    var3.ravel()))

            elif self.dim == 2:
                
                if slice_type=='theta':
                    coords = np.column_stack((
                        coord_x.ravel(),
                        coord_y.ravel()))
                    xi = np.column_stack((
                        var1.ravel(),
                        var2.ravel()))

                elif slice_type=='phi':
                    coords = np.column_stack((
                        coord_x.ravel(),
                        coord_z.ravel()))
                    xi = np.column_stack((
                        var1.ravel(),
                        var3.ravel()))
                    
            elif self.dim==1:
                r=np.sqrt(coord_x**2 + coord_y**2 + coord_z**2)
                coords = r
                xi = np.asarray(var1)


            if field == "gasv_mesh" and component is not None:
                data = np.array(df_sorted.iloc[idx][field])[component]
            else:
                data = np.array(df_sorted.iloc[idx][field])

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
                results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(eval_single)(i) for i in range(components)
                )
                return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])
            else:
                # Trivial escalar case: parallelization over the single snapshot
                result = Parallel(n_jobs=-1, backend='threading')([delayed(eval_single)()])
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
                results = Parallel(n_jobs=-1, backend='threading')(
                    delayed(temporal_interp)(i) for i in range(components)
                )
                return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])
            else:
                # Escalar: paralelización sobre ambos snapshots
                results = Parallel(n_jobs=2, backend='threading')(
                    delayed(temporal_interp)() for _ in range(1)
                )
                result = results[0]
                return result.item() if is_scalar else result.reshape(result_shape)

        # --- Case 3: More than two snapshots, optimized linear temporal interpolation ---
        else:
            # Find the two closest snapshots for linear interpolation
            idx = np.searchsorted(times, time) - 1
            idx = max(0, min(idx, n_snaps - 2))  # Ensure idx is within bounds
            idx_after = idx + 1

            t0, t1 = times[idx], times[idx_after]
            factor = (time - t0) / (t1 - t0) if abs(t1 - t0) > 1e-10 else 0
            factor = max(0, min(factor, 1))

            def temporal_interp(component=None):
                # Precompute values for the two closest snapshots
                val0 = interp(idx, field_name, component)
                val1 = interp(idx_after, field_name, component)
                return (1 - factor) * val0 + factor * val1

            if field_name == "gasv_mesh":
                components = 3 if var3 is not None else 2 if var2 is not None else 1
                # Sequential computation for small number of components
                results = [temporal_interp(i) for i in range(components)]
                return np.array([res.item() if is_scalar else res.reshape(result_shape) for res in results])
            else:
                # Escalar: evitar paralelización innecesaria
                result = temporal_interp()
                return result.item() if is_scalar else result.reshape(result_shape)



    def plot(self, title="Field Plot", t=0, contour_levels=10, component='vz'):
        """
        Automatically determines the plane (XY, XZ, or 3D) and plots the field data.

        Parameters
        ----------
        title : str, optional
            Title of the plot.
        t : int, optional
            Index of the snapshot/time to plot.
        contour_levels : int, optional
            Number of contour levels for 2D plots.
        component : str, optional
            Component to plot for vector fields ('vx', 'vy', 'vz').
        """
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_field() first.")

        if component=='vz':
            comp = 2
        if component=='vy':
            comp = 1
        if component=='vx':
            comp = 0   

        df_names =  self.df.columns.tolist()
        # Load the original field (before slicing) to get the original mesh sizes
        d3 = self.sim.load_field("gasdens", snapshot=int(self.df['snapshot'][t]), interpolate=True)

        # Extract the mesh grids and field data after slicing
        var1 = self.df['var1_mesh'][t]
        var2 = self.df['var2_mesh'][t]
        var3 = self.df['var3_mesh'][t]
        field_data = np.log10(self.df[self.df.columns[-1]][t])  # Last column is the field data (e.g., gasdens_mesh)

        # Get the original shapes of the mesh grids before slicing
        original_shape = d3.var1_mesh[0].shape  # Assuming all var1, var2, var3 have the same shape originally

        # Get the shapes of the resulting mesh grids after applying the slice
        sliced_shape = var1.shape  # Assuming var1, var2, var3 have the same shape after slicing
        
        # Detect fixed angles in slice string
        slice_str = self.slice if hasattr(self, 'slice') and self.slice is not None else ""
        # Fixed theta: e.g. 'theta=1.56' (not theta=[...])
        fixed_theta = re.search(r'theta\s*=\s*([^\[\],]+)', slice_str)
        fixed_phi = re.search(r'phi\s*=\s*([^\[\],]+)', slice_str)

        if fixed_theta:
            plane = 'XY'
        elif fixed_phi:
            plane = 'XZ'
        else:
            plane = 'XY'  # Default/fallback
        # Check the number of dimensions in the sliced shape


        if len(sliced_shape) == 3:
            var1_flat = var1.ravel()
            var2_flat = var2.ravel()
            var3_flat = var3.ravel()
            data = np.log10(field_data.ravel())

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(var1_flat, var2_flat, var3_flat, c=data, cmap='Spectral_r', s=5)
            fig.colorbar(scatter, ax=ax, label=r"$\log_{10}(field)$")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)
            fp.Plot.fargopy_mark(ax)
            plt.show()
        
        
        elif len(sliced_shape) == 2:
            if plane == 'XY':
                fig, ax = plt.subplots(figsize=(10, 8))
                mesh = ax.pcolormesh(var1, var2, field_data, shading='auto', cmap='Spectral_r')
                fig.colorbar(mesh, label=r"$\log_{10}(field)$")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(title)
                fp.Plot.fargopy_mark(ax)
                plt.show()
            elif plane == 'XZ':
                fig, ax = plt.subplots(figsize=(10, 6))
                mesh = ax.pcolormesh(var1, var3, field_data, shading='auto', cmap='Spectral_r')
                fig.colorbar(mesh, label=rf"$\log_{10}(field)$")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                ax.set_title(title)
                fp.Plot.fargopy_mark(ax)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                mesh = ax.pcolormesh(var1, var2, field_data, shading='auto', cmap='Spectral_r')
                fig.colorbar(mesh, label=r"$\log_{10}(field)$")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(title)
                fp.Plot.fargopy_mark(ax)
                plt.show()

    def cut_sector(self, xc, yc, zc, rc, hc, dataframe=None):
        """
        Filter the DataFrame to keep only data inside a cylinder of radius rc and height hc
        centered at (xc, yc, zc). Returns a new filtered DataFrame.

        Parameters
        ----------
        xc, yc, zc : float
            Center coordinates of the cylinder.
        rc : float
            Cylinder radius.
        hc : float
            Cylinder height.
        dataframe : pd.DataFrame, optional
            DataFrame to filter (default: self.df).

        Returns
        -------
        pd.DataFrame
            Filtered DataFrame with only points inside the cylinder.
        """
        if dataframe is None:
            if self.df is None:
                raise ValueError("No DataFrame loaded. Run load_data() first or pasa un DataFrame.")
            dataframe = self.df

        df = dataframe.copy()
        # Asume que las columnas de malla son 'var1_mesh', 'var2_mesh', 'var3_mesh'
        mask_list = []
        for idx, row in df.iterrows():

            x = np.array(row['var1_mesh'])
            y = np.array(row['var2_mesh'])
            z = np.array(row['var3_mesh'])
            # Calcula la máscara booleana para el cilindro
            r_xy = np.sqrt((x - xc)**2 + (y - yc)**2)
            z_min = zc - hc/2
            z_max = zc + hc/2
            mask = (r_xy <= rc) & (z >= z_min) & (z <= z_max)
            # Si el campo es escalar
            filtered = {}
            filtered['snapshot'] = row['snapshot']
            filtered['time'] = row['time']
            filtered['var1_mesh'] = x[mask]
            filtered['var2_mesh'] = y[mask]
            filtered['var3_mesh'] = z[mask]
            # Filtra el campo correspondiente
            for col in df.columns:
                if col.endswith('_mesh') and col not in ['var1_mesh', 'var2_mesh', 'var3_mesh']:
                    data = np.array(row[col])
                    filtered[col] = data[mask]
            mask_list.append(filtered)
        # Convierte la lista de dicts a DataFrame
        filtered_df = pd.DataFrame(mask_list)
        return filtered_df