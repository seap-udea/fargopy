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


from joblib import Parallel, delayed, parallel_config


from ipywidgets import interact, FloatSlider, IntSlider
from celluloid import Camera
from IPython.display import HTML, Video

from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from tqdm import tqdm
from pathlib import Path
import fargopy as fp
from scipy.ndimage import gaussian_filter

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
        self._domain_limits = None
        self._df_sorted_cache = None
        self._slice_type = None
        self._slice_ranges = None

    def _reset_caches(self):
        """Clear cached dataframe and slice metadata before loading or evaluating new data."""
        self._df_sorted_cache = None
        self._slice_type = None
        self._slice_ranges = None

    def _cache_domain_limits(self):
        """Cache domain extrema for r, theta, and phi to avoid repeated property access."""
        if self._domain_limits is not None:
            return
        dom = self.sim.domains
        self._domain_limits = dict(
            r=(dom.r.min(), dom.r.max()),
            theta=(dom.theta.min(), dom.theta.max()),
            phi=(dom.phi.min(), dom.phi.max())
        )

    def _detect_slice_type(self, slice_str):
        """Return the canonical slice type ('theta', 'phi', 'r', or None) inferred from the user string."""
        if not slice_str:
            return None
        txt = slice_str.replace(" ", "").lower()
        m_theta = re.search(r"theta=([^\[\],]+)(?![\]])", txt)
        m_phi = re.search(r"phi=([^\[\],]+)(?![\]])", txt)
        if m_theta and not re.search(r"theta=\[", txt) and m_phi and not re.search(r"phi=\[", txt):
            return "r"
        if m_theta and not re.search(r"theta=\[", txt):
            return "theta"
        if m_phi and not re.search(r"phi=\[", txt):
            return "phi"
        return None

    def _parse_slice_ranges(self, slice_str):
        """Parse the slice expression into numeric (r, theta, phi, z) bounds expressed in radians when needed."""
        ranges = {"r": None, "theta": None, "phi": None, "z": None}
        if not slice_str:
            return ranges
        txt = slice_str.lower()

        def _to_float(value):
            value = value.strip()
            match = re.match(r"(-?\d+(?:\.\d+)?)\s*deg", value)
            return np.deg2rad(float(match.group(1))) if match else float(value)

        range_pattern = re.compile(r"(r|theta|phi|z)=\[(.+?)\]")
        value_pattern = re.compile(r"(r|theta|phi|z)=([^\[\],]+)")

        for key, vals in range_pattern.findall(txt):
            lo, hi = [_to_float(v) for v in vals.split(",")]
            ranges[key] = (min(lo, hi), max(lo, hi))
        for key, val in value_pattern.findall(txt):
            if ranges.get(key) is None:
                parsed = _to_float(val)
                ranges[key] = (parsed, parsed)
        return ranges

    def _get_sorted_dataframe(self, dataframe):
        """Return the dataframe sorted by normalized time, reusing a cached copy when possible."""
        if self._df_sorted_cache and self._df_sorted_cache[0] is dataframe:
            return self._df_sorted_cache[1]
        df_sorted = dataframe.sort_values("time")
        self._df_sorted_cache = (dataframe, df_sorted)
        return df_sorted

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

    def _run_parallel(self, tasks, backend='threading'):
        tasks = list(tasks)
        if not tasks:
            return []
        with parallel_config(n_jobs=-1, prefer='threads'):
            return Parallel(backend=backend)(tasks)
 
    def load_data(self, fields=None, slice=None, snapshots=1, cut=None, coords='cartesian'):
        """
        Load one or multiple fields ('gasdens', 'gasv', 'gasenergy') into a SINGLE
        unified DataFrame with shared coordinates (var1_mesh, var2_mesh, var3_mesh).
        
        This fixes the performance issue when interpolating multiple fields
        independently. Everything else in your code continues to work exactly the same.
        """

        # -------------------------
        # Validate arguments
        # -------------------------
        self._reset_caches()

        if fields is None:
            raise ValueError("You must specify at least one field.")
        if isinstance(fields, str):
            fields = [fields]

        self.fields = fields
        self.slice = slice

        # Convert snapshot into list
        if isinstance(snapshots, int):
            snapshots = [snapshots]
        self.snapshot = snapshots

        # Detect dimensionality from the sliced data (if a slice is provided)
        if slice is not None:
            test_field = self.sim._load_field_raw('gasdens', snapshot=snapshots[0], field_type='scalar')
            try:
                data_slice, mesh = test_field.meshslice(slice=slice)
                self.dim = len(np.array(data_slice).shape)
            except Exception:
                # Fallback: assume full 3D
                self.dim = 3
        else:
            self.dim = 3

        # Snapshot & time arrays
        if len(snapshots) == 1:
            snaps = snapshots
            time_values = [0]
        else:
            snaps = np.arange(snapshots[0], snapshots[1] + 1)
            time_values = np.linspace(0, 1, len(snaps))

        # Store snapshot-time table
        self.snapshot_time_table = pd.DataFrame({
            "Snapshot": snaps,
            "Normalized_time": time_values
        })

        # Slice handling
        if not hasattr(self.sim, "domains") or self.sim.domains is None:
            raise ValueError("Simulation domains are not loaded.")
        self._cache_domain_limits()
        self._slice_type = self._detect_slice_type(slice)
        self._slice_ranges = self._parse_slice_ranges(slice)

        # -------------------------
        # Helper for rotation (phi slices)
        # -------------------------
        def _rotation(X, Y, Z, phi0):
            X_rot =  X * np.cos(phi0) + Y * np.sin(phi0)
            Y_rot = -X * np.sin(phi0) + Y * np.cos(phi0)
            return X_rot, Y_rot, Z.copy()

        # =====================================================================
        # ========================   2D CASE   ================================
        # =====================================================================
        if self.dim < 3:

            # Collect rows and build DataFrame once to avoid repeated concat
            rows = []

            for i, snap in enumerate(snaps):

                row = {'snapshot': snap, 'time': time_values[i]}
                coords_assigned = False  # Only assign var1/var2/var3 once

                # Loop over requested fields
                for field in fields:

                    # -----------------
                    # GASDENS 2D
                    # -----------------
                    if field == 'gasdens':
                        gasd = self.sim._load_field_raw('gasdens', snapshot=snap, field_type='scalar')
                        data_slice, mesh = gasd.meshslice(slice=slice)

                        # assign coordinates only once
                        if not coords_assigned:
                            if coords == 'cartesian':
                                # rotate if phi is fixed
                                try:
                                    if np.all(mesh.phi.ravel() == mesh.phi.ravel()[0]):
                                        phi0 = mesh.phi.ravel()[0]
                                        x_rot, y_rot, z_rot = _rotation(mesh.x, mesh.y, mesh.z, phi0)
                                        row['var1_mesh'] = x_rot
                                        row['var2_mesh'] = y_rot
                                        row['var3_mesh'] = z_rot
                                    else:
                                        row['var1_mesh'] = mesh.x
                                        row['var2_mesh'] = mesh.y
                                        row['var3_mesh'] = mesh.z
                                except Exception:
                                    # Fallback if mesh lacks phi
                                    row['var1_mesh'] = mesh.x
                                    row['var2_mesh'] = mesh.y
                                    row['var3_mesh'] = mesh.z
                            else:
                                # original coordinate names as defined in simulation
                                vnames = getattr(self.sim.vars, 'VARIABLES', ['x', 'y', 'z'])
                                row['var1_mesh'] = getattr(mesh, vnames[0])
                                row['var2_mesh'] = getattr(mesh, vnames[1])
                                row['var3_mesh'] = getattr(mesh, vnames[2])
                            coords_assigned = True

                        row['gasdens_mesh'] = data_slice

                    # -----------------
                    # GASV 2D
                    # -----------------
                    if field == 'gasv':
                        gasv_raw = self.sim._load_field_raw('gasv', snapshot=snap, field_type='vector')
                        if coords == 'cartesian':
                            gasvx, gasvy, gasvz = gasv_raw.to_cartesian()
                            v1, mesh = gasvx.meshslice(slice=slice)
                            v2, mesh = gasvy.meshslice(slice=slice)
                            v3, mesh = gasvz.meshslice(slice=slice)

                            if not coords_assigned:
                                row['var1_mesh'] = mesh.x
                                row['var2_mesh'] = mesh.y
                                row['var3_mesh'] = mesh.z
                                coords_assigned = True

                            row['gasv_mesh'] = np.array([v1, v2, v3])
                        else:
                            v_slice, mesh = gasv_raw.meshslice(slice=slice)
                            v1, v2, v3 = v_slice[0], v_slice[1], v_slice[2]
                            if not coords_assigned:
                                vnames = getattr(self.sim.vars, 'VARIABLES', ['x', 'y', 'z'])
                                row['var1_mesh'] = getattr(mesh, vnames[0])
                                row['var2_mesh'] = getattr(mesh, vnames[1])
                                row['var3_mesh'] = getattr(mesh, vnames[2])
                                coords_assigned = True
                            row['gasv_mesh'] = np.array([v1, v2, v3])

                    # -----------------
                    # GASENERGY 2D
                    # -----------------
                    if field == 'gasenergy':
                        gasen = self.sim._load_field_raw('gasenergy', snapshot=snap, field_type='scalar')
                        data_slice, mesh = gasen.meshslice(slice=slice)

                        if not coords_assigned:
                            if coords == 'cartesian':
                                row['var1_mesh'] = mesh.x
                                row['var2_mesh'] = mesh.y
                                row['var3_mesh'] = mesh.z
                            else:
                                vnames = getattr(self.sim.vars, 'VARIABLES', ['x', 'y', 'z'])
                                row['var1_mesh'] = getattr(mesh, vnames[0])
                                row['var2_mesh'] = getattr(mesh, vnames[1])
                                row['var3_mesh'] = getattr(mesh, vnames[2])
                            coords_assigned = True

                        row['gasenergy_mesh'] = data_slice

                # collect row dicts and build DataFrame once
                rows.append(row)

            df_snapshots = pd.DataFrame(rows)
            self.df = df_snapshots
            return df_snapshots

        # =====================================================================
        # ========================   3D CASE   ================================
        # =====================================================================
        if self.dim == 3:

            # Build full mesh
            theta, r, phi = np.meshgrid(
                self.sim.domains.theta,
                self.sim.domains.r,
                self.sim.domains.phi,
                indexing='ij'
            )
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)

            # Apply spherical or cylindrical mask (optional)
            if cut is not None:
                if len(cut) == 5:
                    xc, yc, zc, rc, hc = cut
                    r_xy = np.sqrt((x - xc)**2 + (y - yc)**2)
                    zmin, zmax = zc - hc/2, zc + hc/2
                    mask = (r_xy <= rc) & (z >= zmin) & (z <= zmax)
                elif len(cut) == 4:
                    xc, yc, zc, rs = cut
                    r_sph = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
                    mask = r_sph <= rs
                else:
                    raise ValueError("cut must have 4 (sphere) or 5 (cylinder) elements.")
            else:
                mask = None

            # Collect rows and build DataFrame once to avoid repeated concat
            rows = []

            for i, snap in enumerate(snaps):

                row = {'snapshot': snap, 'time': time_values[i]}
                coords_assigned = False

                # Loop over requested fields
                for field in fields:

                    # -----------------
                    # GASDENS 3D
                    # -----------------
                    if field == "gasdens":
                        gasd = self.sim._load_field_raw('gasdens', snapshot=snap, field_type='scalar')

                        if not coords_assigned:
                            if coords == 'cartesian':
                                if mask is not None:
                                    row["var1_mesh"] = x[mask]
                                    row["var2_mesh"] = y[mask]
                                    row["var3_mesh"] = z[mask]
                                else:
                                    row["var1_mesh"] = x
                                    row["var2_mesh"] = y
                                    row["var3_mesh"] = z
                            else:
                                # original coordinate variables order
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r,phi=phi,theta=theta,x=x,y=y,z=z)
                                if mask is not None:
                                    row["var1_mesh"] = mapping[v0][mask]
                                    row["var2_mesh"] = mapping[v1][mask]
                                    row["var3_mesh"] = mapping[v2][mask]
                                else:
                                    row["var1_mesh"] = mapping[v0]
                                    row["var2_mesh"] = mapping[v1]
                                    row["var3_mesh"] = mapping[v2]
                            coords_assigned = True

                        row["gasdens_mesh"] = gasd.data[mask] if mask is not None else gasd.data
                    # -----------------
                    # GASV 3D
                    # -----------------
                    if field == "gasv":
                        gasv_raw = self.sim._load_field_raw('gasv', snapshot=snap, field_type='vector')
                        if coords == 'cartesian':
                            gasvx, gasvy, gasvz = gasv_raw.to_cartesian()

                            if not coords_assigned:
                                if mask is not None:
                                    row["var1_mesh"] = x[mask]
                                    row["var2_mesh"] = y[mask]
                                    row["var3_mesh"] = z[mask]
                                else:
                                    row["var1_mesh"] = x
                                    row["var2_mesh"] = y
                                    row["var3_mesh"] = z
                                coords_assigned = True

                            if mask is not None:
                                row["gasv_mesh"] = np.array([
                                    gasvx.data[mask],
                                    gasvy.data[mask],
                                    gasvz.data[mask]
                                ])
                            else:
                                row["gasv_mesh"] = np.array([
                                    gasvx.data,
                                    gasvy.data,
                                    gasvz.data
                                ])
                        else:
                            vdata = gasv_raw.data
                            if not coords_assigned:
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r,phi=phi,theta=theta,x=x,y=y,z=z)
                                if mask is not None:
                                    row["var1_mesh"] = mapping[v0][mask]
                                    row["var2_mesh"] = mapping[v1][mask]
                                    row["var3_mesh"] = mapping[v2][mask]
                                else:
                                    row["var1_mesh"] = mapping[v0]
                                    row["var2_mesh"] = mapping[v1]
                                    row["var3_mesh"] = mapping[v2]
                                coords_assigned = True

                            if mask is not None:
                                row["gasv_mesh"] = np.array([vdata[0][mask], vdata[1][mask], vdata[2][mask]])
                            else:
                                row["gasv_mesh"] = np.array([vdata[0], vdata[1], vdata[2]])

                    # -----------------
                    # GASENERGY 3D
                    # -----------------
                    if field == "gasenergy":
                        gasen = self.sim._load_field_raw('gasenergy', snapshot=snap, field_type='scalar')

                        if not coords_assigned:
                            if coords == 'cartesian':
                                if mask is not None:
                                    row["var1_mesh"] = x[mask]
                                    row["var2_mesh"] = y[mask]
                                    row["var3_mesh"] = z[mask]
                                else:
                                    row["var1_mesh"] = x
                                    row["var2_mesh"] = y
                                    row["var3_mesh"] = z
                            else:
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r,phi=phi,theta=theta,x=x,y=y,z=z)
                                if mask is not None:
                                    row["var1_mesh"] = mapping[v0][mask]
                                    row["var2_mesh"] = mapping[v1][mask]
                                    row["var3_mesh"] = mapping[v2][mask]
                                else:
                                    row["var1_mesh"] = mapping[v0]
                                    row["var2_mesh"] = mapping[v1]
                                    row["var3_mesh"] = mapping[v2]
                            coords_assigned = True

                        row["gasenergy_mesh"] = gasen.data[mask] if mask is not None else gasen.data

                # collect row dicts and build DataFrame once
                rows.append(row)

            df_snapshots = pd.DataFrame(rows)
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
        Build a boolean mask that keeps only coordinates within the simulation domain and
        enforces any user-specified radial/angle limits for XY (theta) or XZ (phi) slices.
        """
        self._cache_domain_limits()
        slice = slice or self.slice
        slice_ranges = self._slice_ranges or self._parse_slice_ranges(slice)
        r_bounds = slice_ranges.get('r')
        theta_bounds = slice_ranges.get('theta')
        phi_bounds = slice_ranges.get('phi')
        r_min, r_max = self._domain_limits['r']
        theta_min, theta_max = self._domain_limits['theta']
        phi_min, phi_max = self._domain_limits['phi']
        eps = 1e-7
        xi = np.asarray(xi)
        ndim = xi.shape[1] if xi.ndim > 1 else 1

        def _bounded(vals, bounds, default):
            if bounds is None:
                return vals >= default[0] - eps, vals <= default[1] + eps
            lo, hi = bounds
            return vals >= lo - eps, vals <= hi + eps

        def _phi_in_bounds(phi_vals):
            if phi_bounds is None:
                return np.ones_like(phi_vals, dtype=bool)
            lo, hi = phi_bounds
            if lo <= hi:
                return (phi_vals >= lo - eps) & (phi_vals <= hi + eps)
            return (phi_vals >= lo - eps) | (phi_vals <= hi + eps)

        if ndim == 2:
            # XY plane: theta fixed
            if slice is not None and 'theta' in slice:
                # XY plane: z = 0, theta fixed, filter by r and phi
                x, y = xi[:, 0], xi[:, 1]
                r = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                r_ge, r_le = _bounded(r, r_bounds, (r_min, r_max))
                mask = r_ge & r_le & _phi_in_bounds(phi)
                return mask
            
            # XZ plane: phi fixed
            elif slice is not None and 'phi' in slice:
                # XZ plane: y = 0, phi fixed, filter by r and theta
                x, z = xi[:, 0], xi[:, 1]
                r = np.sqrt(x**2 + z**2)
                theta = np.arccos(z / np.clip(r, 1e-14, None))
                r_ge, r_le = _bounded(r, r_bounds, (r_min, r_max))
                if theta_bounds:
                    lo, hi = theta_bounds
                    theta_mask = (theta >= lo - eps) & (theta <= hi + eps)
                else:
                    theta_mask = (
                        ((theta > theta_min) | np.isclose(theta, theta_min, atol=eps)) &
                        ((theta < theta_max) | np.isclose(theta, theta_max, atol=eps))
                    )
                return r_ge & r_le & theta_mask
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
            xi_1d = np.asarray(xi).ravel()

            # Decide which variable is the "free" one in the 1D cut.
            # Prefer explicit ranges (r=[...], theta=[...], phi=[...]); otherwise,
            # the free variable is the one that does NOT appear as a scalar in the slice.
            r_b = r_bounds
            th_b = theta_bounds
            ph_b = phi_bounds

            def _is_range(b):
                return (b is not None) and (abs(b[1] - b[0]) > 1e-12)

            if _is_range(r_b):
                free = 'r'
            elif _is_range(th_b):
                free = 'theta'
            elif _is_range(ph_b):
                free = 'phi'
            else:
                s = (slice or self.slice) or ""
                s_low = s.lower()
                has_r = re.search(r"\br\s*=", s_low) is not None
                has_th = re.search(r"\btheta\s*=", s_low) is not None
                has_ph = re.search(r"\bphi\s*=", s_low) is not None
                # the free variable is the one not present in the slice specification
                if not has_r:
                    free = 'r'
                elif not has_th:
                    free = 'theta'
                elif not has_ph:
                    free = 'phi'
                else:
                    free = 'r'

            # Build mask depending on which variable is free
            if free == 'r':
                lo, hi = (r_b if r_b is not None else (r_min, r_max))
                mask = (xi_1d >= lo - eps) & (xi_1d <= hi + eps)
                return mask

            if free == 'theta':
                lo, hi = (th_b if th_b is not None else (theta_min, theta_max))
                mask = (xi_1d >= lo - eps) & (xi_1d <= hi + eps)
                return mask

            # free == 'phi'
            lo, hi = (ph_b if ph_b is not None else (phi_min, phi_max))
            if lo <= hi:
                mask = (xi_1d >= lo - eps) & (xi_1d <= hi + eps)
            else:
                # wrap-around range (e.g. [5.5, 0.5] in radians)
                mask = (xi_1d >= lo - eps) | (xi_1d <= hi + eps)
            return mask

        if ndim==3:
            mask = np.ones(xi.shape[0],dtype=bool)
            return mask

    def evaluate(
            self, time, var1, var2=None, var3=None, dataframe=None,
            interpolator="griddata", method="linear",
            rbf_kwargs=None, griddata_kwargs=None, idw_kwargs=None,
            sigma_smooth=None, field=None, reflect=False
        ):
        """
        Evaluate the selected field at arbitrary spatial coordinates using
        multi-snapshot interpolation. Supports scalar and vector fields,
        1D/2D/3D geometry, time interpolation, and several interpolation
        backends. Designed for unified DataFrames (gasdens + gasv + others).

        Parameters
        ----------
        time : float or int
            Normalized time in [0,1] or snapshot index.

        var1, var2, var3 : array-like or float
            Evaluation coordinates (x,y,z for 3D). Scalars are accepted.

        dataframe : pandas.DataFrame, optional
            Custom DataFrame. If omitted, self.df is used.

        interpolator : {"griddata","rbf","linearnd","idw"}
            Backend used for spatial interpolation.

        method : str
            Kernel/method used by backend (e.g., 'linear' for griddata).

        sigma_smooth : float or None
            Optional Gaussian smoothing.

        field : {"gasdens","gasv","gasenergy"} or None
            Field to evaluate. If None and DF has >1 field → explicit error.

        Returns
        -------
        ndarray or float
            Interpolated value(s). Vector fields return shape (3,N) or (3,...).
        """

        # ===============================================================
        # Basic validation
        # ===============================================================
        if sigma_smooth is not None and sigma_smooth <= 0:
            raise ValueError("sigma_smooth must be None or positive.")

        df = dataframe if dataframe is not None else self.df
        if df is None:
            raise ValueError("No dataframe available.")

        # ===============================================================
        # FIELD SELECTION (safe and explicit)
        # ===============================================================
        field_map = {
            "gasdens": "gasdens_mesh",
            "gasv": "gasv_mesh",
            "gasenergy": "gasenergy_mesh"
        }

        if field is not None:
            if field in field_map:
                field = field_map[field]
            if field not in df.columns:
                raise ValueError(
                    f"Field '{field}' not in DF. Available: {list(df.columns)}"
                )
        else:
            # Autodetect only if exactly one exists
            candidates = [
                c for c in df.columns
                if c in ("gasdens_mesh","gasv_mesh","gasenergy_mesh")
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Multiple fields present {candidates}. "
                    "Specify field='gasdens', 'gasv', or 'gasenergy'."
                )
            field = candidates[0]

        # ===============================================================
        # Prepare snapshot ordering
        # ===============================================================
        df_sorted = self._get_sorted_dataframe(df)
        times = df_sorted["time"].values
        nsnaps = len(times)

        # Detect scalar inputs
        is_scalar = (
            np.isscalar(var1)
            and (var2 is None or np.isscalar(var2))
            and (var3 is None or np.isscalar(var3))
        )
        result_shape = () if is_scalar else np.asarray(var1).shape

        if np.isscalar(var1): var1 = np.array([var1])
        if np.isscalar(var2): var2 = np.array([var2])
        if np.isscalar(var3): var3 = np.array([var3])

        # Convenience: allow calling evaluate(var1=..., var2=...) for
        # 2D XZ slices where the expected coordinates are (var1,var3).
        # If the slice type is not 'theta' (i.e. XZ) and the user passed
        # a value for var2 but left var3=None, treat var2 as var3.
        try:
            slice_type_tmp = self._slice_type or self._detect_slice_type(self.slice)
        except Exception:
            slice_type_tmp = None
        if self.dim == 2 and slice_type_tmp is not None and slice_type_tmp != 'theta':
            if var3 is None and var2 is not None:
                var3 = var2
                var2 = None

        # ===============================================================
        # Smoothing helper
        # ===============================================================
        def _smooth(values):
            if sigma_smooth is None or np.isscalar(values):
                return values

            arr = np.asarray(values)
            if arr.ndim == 0:
                return values

            # Vector smoothing
            if field == "gasv_mesh" and arr.ndim >= 2:
                out = np.empty_like(arr)
                for k in range(arr.shape[0]):
                    out[k] = gaussian_filter(arr[k], sigma=sigma_smooth)
                return out

            return gaussian_filter(arr, sigma=sigma_smooth)

        # ===============================================================
        # Interpolation backends
        # ===============================================================

        def idw_interp(coords, values, xi):
            coords = np.asarray(coords)
            values = np.asarray(values).ravel()
            xi = np.asarray(xi)

            mask = self._domain_mask(xi)
            if reflect:
                mask = np.ones(xi.shape[0], dtype=bool)
            out = np.zeros(xi.shape[0])
            tree = cKDTree(coords)
            k = idw_kwargs.get("k", 8)
            power = idw_kwargs.get("power", 2)

            # If mask selects points, compute only there. Otherwise try for all xi.
            if np.any(mask):
                d, idxs = tree.query(xi[mask], k=k)
                d = np.where(d == 0, 1e-10, d)
                w = 1 / d**power
                w /= w.sum(axis=1, keepdims=True)
                out[mask] = np.sum(values[idxs] * w, axis=1)
                return out

            # Fallback: compute for all xi
            d, idxs = tree.query(xi, k=k)
            d = np.where(d == 0, 1e-10, d)
            w = 1 / d**power
            w /= w.sum(axis=1, keepdims=True)
            out = np.sum(values[idxs] * w, axis=1)
            return out


        def rbf_interp(coords, values, xi):
            coords = np.asarray(coords)
            values = np.asarray(values).ravel()
            xi = np.asarray(xi)

            mask = self._domain_mask(xi)
            if reflect:
                mask = np.ones(xi.shape[0], dtype=bool)
            out = np.full(xi.shape[0], np.nan)

            # Try interpolate where mask True
            try:
                obj = RBFInterpolator(coords, values, kernel=method, **(rbf_kwargs or {}))
            except Exception:
                return np.zeros(xi.shape[0])

            if np.any(mask):
                out[mask] = obj(xi[mask])
                # attempt to leave other positions as NaN
                return np.where(np.isfinite(out), out, np.nan)

            # Fallback: evaluate on all xi
            vals_all = obj(xi)
            return np.where(np.isfinite(vals_all), vals_all, np.nan)
    
 
        def griddata_interp(coords, values, xi):
            # --- Apply domain mask: only interpolate inside the simulation domain ---
            mask = self._domain_mask(xi)
            if reflect:
                mask = np.ones(xi.shape[0], dtype=bool)
            out = np.full(xi.shape[0], np.nan)

            # If mask has selected points, interpolate only there
            if np.any(mask):
                out[mask] = griddata(coords, values.ravel(), xi[mask], method=method)
                # leave outside as NaN -> caller can mask later
                return np.where(np.isfinite(out), out, np.nan)

            # Fallback: try interpolate for all xi (useful when domain mask selection fails)
            try:
                vals_all = griddata(coords, values.ravel(), xi, method=method)
                return np.where(np.isfinite(vals_all), vals_all, np.nan)
            except Exception:
                return np.zeros(xi.shape[0])
 
 
        def linearnd_interp(coords, values, xi):
            coords = np.asarray(coords)
            values = np.asarray(values).ravel()
            xi = np.asarray(xi)

            mask = self._domain_mask(xi)
            if reflect:
                mask = np.ones(xi.shape[0], dtype=bool)
            out = np.full(xi.shape[0], np.nan)
            obj = LinearNDInterpolator(coords, values)

            if np.any(mask):
                out[mask] = obj(xi[mask])
                return np.where(np.isfinite(out), out, np.nan)

            # Fallback: evaluate on all xi
            vals_all = obj(xi)
            return np.where(np.isfinite(vals_all), vals_all, np.zeros_like(vals_all))

        # ===============================================================
        # Main interpolation kernel
        # ===============================================================
        slice_type = self._slice_type or self._detect_slice_type(self.slice)

        def interp(idx, field_name, comp=None):
            row = df_sorted.iloc[idx]

            cx = np.array(row["var1_mesh"])
            cy = np.array(row["var2_mesh"])
            cz = np.array(row["var3_mesh"])

            # Build coordinate arrays
            if self.dim == 3:
                coords = np.column_stack((cx.ravel(), cy.ravel(), cz.ravel()))
                xi = np.column_stack((var1.ravel(), var2.ravel(), var3.ravel()))
            elif self.dim == 2:
                if slice_type == "theta":
                    coords = np.column_stack((cx.ravel(), cy.ravel()))
                    xi = np.column_stack((var1.ravel(), var2.ravel()))
                else:
                    coords = np.column_stack((cx.ravel(), cz.ravel()))
                    xi = np.column_stack((var1.ravel(), var3.ravel()))
            elif self.dim == 1:
                coords = np.sqrt(cx**2 + cy**2 + cz**2)
                xi = np.asarray(var1)

            # Select field
            data = row[field_name]

            # -------------------------------------------
            # UNIVERSAL VECTOR COMPONENT SELECTOR
            # -------------------------------------------
            if isinstance(data, np.ndarray) and comp is not None:
                if data.ndim == 2 and data.shape[0] == 3:
                    data = data[comp]
                elif data.ndim == 2 and data.shape[1] == 3:
                    data = data[:, comp]
                elif data.ndim == 3 and data.shape[0] == 3:
                    data = data[comp].ravel()
                elif data.ndim == 4 and data.shape[0] == 3:
                    data = data[comp].ravel()
                elif data.ndim == 4 and data.shape[-1] == 3:
                    data = data[..., comp].ravel()
                else:
                    raise ValueError(f"Cannot extract vector component from {data.shape}")

            # -------------------------------------------------
            # Reflection augmentation
            # If `reflect=True` we augment the interpolation
            # dataset reflecting across the equatorial plane z=0
            # (i.e. z -> -z). For 2D XZ cuts (coords (x,z)) we flip z.
            # For vector components, the component normal to the
            # reflection plane (vz) changes sign.
            # -------------------------------------------------
            if reflect:
                try:
                    # Normalize coords to shape (N, ndim)
                    coords_arr = np.asarray(coords)
                    ndim = coords_arr.shape[1] if coords_arr.ndim == 2 else 1
                    coords_orig = coords_arr.reshape(-1, ndim)
                    coords_ref = coords_orig.copy()

                    # Flip only the z coordinate (index -1 if 3D, index 1 if 2D XZ)
                    if coords_orig.shape[1] == 3:
                        coords_ref[:, 2] *= -1
                    elif coords_orig.shape[1] == 2:
                        # assume (x,z) layout for XZ cuts
                        coords_ref[:, 1] *= -1

                    # Prepare data values (flattened)
                    data_flat = np.asarray(data).ravel()

                    # For vector components, reflect sign for the
                    # component perpendicular to the plane (vz -> -vz)
                    if field_name == 'gasv_mesh' and comp is not None:
                        # comp: 2->vz (flip), others unchanged
                        if comp == 2:
                            data_ref = -data_flat
                        else:
                            data_ref = data_flat.copy()
                    else:
                        # Scalars or already selected components
                        data_ref = data_flat.copy()

                    # Augment coords and data for interpolation
                    coords = np.vstack([coords_orig, coords_ref])
                    data = np.concatenate([data_flat, data_ref])
                except Exception:
                    # On error, fallback to original coords/data
                    coords = np.asarray(coords)
                    data = np.asarray(data)

            # Dispatch backend
            if interpolator == "rbf":
                return rbf_interp(coords, data, xi)
            elif interpolator == "linearnd":
                return linearnd_interp(coords, data, xi)
            elif interpolator == "idw":
                return idw_interp(coords, data, xi)
            else:
                return griddata_interp(coords, data, xi)

        # ===============================================================
        # TIME INTERPOLATION
        # ===============================================================
        if nsnaps == 1:
            if field == "gasv_mesh":
                vals = [interp(0, field, c) for c in range(3)]
                arr = np.array([v.item() if is_scalar else v.reshape(result_shape) for v in vals])
                return _smooth(arr)
            v = interp(0, field)
            return _smooth(v.item() if is_scalar else v.reshape(result_shape))

        # Two snapshots
        if nsnaps == 2:
            i0, i1 = 0, 1
            t0, t1 = times[i0], times[i1]
            fac = (time - t0) / (t1 - t0) if abs(t1 - t0) > 1e-12 else 0
            fac = np.clip(fac, 0, 1)

            def blend(c=None):
                v0 = interp(i0, field, c)
                v1 = interp(i1, field, c)
                return (1 - fac) * v0 + fac * v1

            if field == "gasv_mesh":
                vals = [blend(c) for c in range(3)]
                arr = np.array([v.item() if is_scalar else v.reshape(result_shape) for v in vals])
                return _smooth(arr)

            v = blend()
            return _smooth(v.item() if is_scalar else v.reshape(result_shape))

        # Many snapshots
        i0 = np.searchsorted(times, time) - 1
        i0 = np.clip(i0, 0, nsnaps - 2)
        i1 = i0 + 1
        t0, t1 = times[i0], times[i1]
        fac = (time - t0) / (t1 - t0) if abs(t1 - t0) > 1e-12 else 0
        fac = np.clip(fac, 0, 1)

        def blend(c=None):
            v0 = interp(i0, field, c)
            v1 = interp(i1, field, c)
            return (1 - fac) * v0 + fac * v1

        if field == "gasv_mesh":
            vals = [blend(c) for c in range(3)]
            arr = np.array([v.item() if is_scalar else v.reshape(result_shape) for v in vals])
            return _smooth(arr)

        v = blend()
        return _smooth(v.item() if is_scalar else v.reshape(result_shape))


    def plot(self, title="Field Colormap", t=0, contour_levels=10, component='vz', smoothing_sigma=None):
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
        d3 = self.sim._load_field_raw("gasdens", snapshot=int(self.df['snapshot'][t]), field_type='scalar')

        # Extract the mesh grids and field data after slicing
        var1 = self.df['var1_mesh'][t]
        var2 = self.df['var2_mesh'][t]
        var3 = self.df['var3_mesh'][t]
        field_data = np.log10(self.df[self.df.columns[-1]][t])  # Last column is the field data (e.g., gasdens_mesh)

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

            fig = plt.figure(figsize=(8, 6))
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
            # Optional smoothing to remove interpolation artefacts (triangular edges)
            if smoothing_sigma is not None:
                try:
                    field_data = gaussian_filter(field_data, sigma=smoothing_sigma)
                except Exception:
                    # If smoothing fails, fall back to original data
                    field_data = field_data
            if plane == 'XY':
                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(var1, var2, field_data, shading='auto', cmap='Spectral_r')
                fig.colorbar(mesh, label=r"$\log_{10}(field)$")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(title)
                fp.Plot.fargopy_mark(ax)
                plt.show()
            elif plane == 'XZ':
                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(var1, var3, field_data, shading='auto', cmap='Spectral_r')
                fig.colorbar(mesh, label=rf"$\log_{10}(field)$")
                ax.set_xlabel("X")
                ax.set_ylabel("Z")
                ax.set_title(title)
                fp.Plot.fargopy_mark(ax)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
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
                raise ValueError("No DataFrame loaded. Run load_data() first or pass a DataFrame.")
            dataframe = self.df

        df = dataframe.copy()
        # Assume mesh columns are named 'var1_mesh', 'var2_mesh', 'var3_mesh'
        mask_list = []
        for idx, row in df.iterrows():

            x = np.array(row['var1_mesh'])
            y = np.array(row['var2_mesh'])
            z = np.array(row['var3_mesh'])
            # Compute boolean mask selecting points inside the cylinder
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
            # Filter the corresponding field columns
            for col in df.columns:
                if col.endswith('_mesh') and col not in ['var1_mesh', 'var2_mesh', 'var3_mesh']:
                    data = np.array(row[col])
                    filtered[col] = data[mask]
            mask_list.append(filtered)
        # Convert the list of dicts to a DataFrame
        filtered_df = pd.DataFrame(mask_list)
        return filtered_df