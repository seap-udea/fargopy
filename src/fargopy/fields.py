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
from scipy.interpolate import RegularGridInterpolator
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
    """Represents a simulation field (scalar or vector) with coordinate system and domain information.

    The ``Field`` object encapsulates the data arrays and associated
    coordinate meshes for a specific simulation variable. It supports
    slicing, coordinate transformation and simple visualization
    helpers.

    Attributes
    ----------
    data : np.ndarray
        Numpy array containing the physical data.
    coordinates : str
        Coordinate system type ('cartesian', 'cylindrical', 'spherical').
    domains : object
        Object containing domain-specific geometry (e.g., mesh arrays).
    type : str
        Field type ('scalar' or 'vector').

    Examples
    --------
    Load a field from a simulation object (returns a Field instance
    if interpolation is disabled or a FieldInterpolator otherwise):
    
    >>> fp.Simulation.load_field(fields='gasdens', snapshot=0, interpolate=False)
    
    Access data and mesh:
    
    >>> rho = field.data
    >>> xmesh = field.mesh.x
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
        """Perform a slice on a field and produce the corresponding field slice and
        associated coordinate matrices for plotting.

        Parameters
        ----------
        slice : str
            Slice definition string (e.g., 'z=0').
        component : int, optional
            Component index for vector fields (default: 0).
        verbose : bool, optional
            If True, print debug information.

        Returns
        -------
        tuple
            (sliced field, mesh dictionary with coordinates). The mesh dictionary
            contains coordinate arrays (x, y, z, r, phi, theta) corresponding
            to the slice.

        Examples
        --------
        Slice a field at z=0:
        
        >>> field_slice, mesh = field.meshslice(slice='z=0')
        
        Plot the slice:
        
        >>> plt.pcolormesh(mesh.x, mesh.y, field_slice)
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
            # Handle variable number of components (2 for 2D, 3 for 3D)
            ncomponents = self.data.shape[0]
            slice_components = []
            for i in range(ncomponents):
                component_cmd = f"self.data[{i},{pattern_str}]"
                slice_components.append(eval(component_cmd, {"self": self, "np": np}))
            slice = np.array(slice_components)

        if pattern:
            return slice,pattern_str
        return slice

    def to_cartesian(self):
        """
        Convert the field to cartesian coordinates.

        Returns
        -------
        Field or tuple of Field
            The field in cartesian coordinates. For scalar fields, returns the field itself.
            For vector fields, returns a tuple (vx, vy, vz).

        Examples
        --------
        >>> v = sim.load_field('gasv', snapshot=0)
        >>> vx, vy, vz = v.to_cartesian()
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
class FieldInterpolator(fargopy.Fargobj):
    """Loads and interpolates fields from a FARGO3D simulation.
    
    The ``FieldInterpolator`` facilitates loading, slicing, and interpolating
    simulation data across multiple snapshots and fields. It handles coordinate
    transformations and dimensionality reduction based on slice definitions.

    Attributesde
    ----------
    sim : Simulation
        The simulation object.
    df : pd.DataFrame or None
        DataFrame containing loaded field data organized by snapshot and time.
    snapshot_time_table : pd.DataFrame or None
        Table mapping snapshots to normalized time.
    snapshot : list or None
        List of loaded snapshots.
    slice : str or None
        Slice definition string used to load the data.
    dim : int or None
        Dimensionality of the loaded data (e.g., 2 for a 2D slice).

    Examples
    --------
    Load multiple fields from a snapshot with interpolation enabled:
    
    >>> data = sim.load_field(fields=['gasdens', 'gasv'], snapshot=4)
    
    Load a specific slice:
    
    >>> dens = sim.load_field(fields='gasdens', slice='r=[0.8,1.2],phi=[-25 deg,25 deg],theta=1.56', snapshot=4)
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
        self._is_regular_grid = None
        self._grid_axes = None
        self._axis_order = None
        self._original_coords = None

    def _reset_caches(self):
        """Clear cached dataframe and slice metadata before loading or evaluating new data."""
        self._df_sorted_cache = None
        self._slice_type = None
        self._slice_ranges = None

    def _cache_domain_limits(self):
        """Cache domain extrema for r, theta, phi, and z to avoid repeated property access."""
        if self._domain_limits is not None:
            return
        dom = self.sim.domains
        coords = self.sim.vars.COORDINATES
        
        if coords == 'spherical':
            self._domain_limits = dict(
                r=(dom.r.min(), dom.r.max()),
                theta=(dom.theta.min(), dom.theta.max()),
                phi=(dom.phi.min(), dom.phi.max()),
                z=(None, None)
            )
        elif coords == 'cylindrical':
            self._domain_limits = dict(
                r=(dom.r.min(), dom.r.max()),
                z=(dom.z.min(), dom.z.max()),
                phi=(dom.phi.min(), dom.phi.max()),
                theta=(None, None)
            )
        else:  # cartesian
            self._domain_limits = dict(
                x=(dom.x.min(), dom.x.max()),
                y=(dom.y.min(), dom.y.max()),
                z=(dom.z.min(), dom.z.max()),
                r=(None, None),
                theta=(None, None),
                phi=(None, None)
            )

    def _detect_slice_type(self, slice_str):
        """Return the canonical slice type ('theta', 'phi', 'r', 'z', or None) inferred from the user string.
        
        For spherical coordinates:
            - theta=constant -> XY plane slice
            - phi=constant -> XZ plane slice  
            - r=constant -> spherical shell
            
        For cylindrical coordinates:
            - z=constant -> XY plane slice (horizontal)
            - phi=constant -> RZ plane slice (vertical)
            - r=constant -> cylindrical surface
        """
        if not slice_str:
            return None
        txt = slice_str.replace(" ", "").lower()
        
        # Detect coordinate system
        coords = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'
        
        if coords == 'cylindrical':
            # For cylindrical coordinates, check for z, phi, r
            m_z = re.search(r"z=([^\[\],]+)(?![\]])", txt)
            m_phi = re.search(r"phi=([^\[\],]+)(?![\]])", txt)
            m_r = re.search(r"r=([^\[\],]+)(?![\]])", txt)
            
            # If both phi and r are fixed, we have a line along z
            if m_phi and not re.search(r"phi=\[", txt) and m_r and not re.search(r"r=\[", txt):
                return "z"
            # If z is fixed, we have XY plane (r-phi slice)
            if m_z and not re.search(r"z=\[", txt):
                return "z"
            # If phi is fixed, we have RZ plane
            if m_phi and not re.search(r"phi=\[", txt):
                return "phi"
            # If r is fixed, we have cylindrical surface
            if m_r and not re.search(r"r=\[", txt):
                return "r"
                
        elif coords == 'spherical':
            # Original spherical coordinate logic
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

    def _compute_cut_indices(self, cut, coords_system='spherical'):
        """
        Calculate structured grid indices for a geometric cut (sphere or cylinder).
        
        Instead of using boolean masks that destroy grid structure, this method
        computes the bounding box of the cut in spherical/cylindrical coordinates
        and returns index ranges that preserve the regular grid structure.
        
        Parameters
        ----------
        cut : tuple
            For sphere: (xc, yc, zc, rs) - center coordinates and radius
            For cylinder: (xc, yc, zc, rc, hc) - center, radius, height
        coords_system : str
            'spherical' or 'cylindrical'
            
        Returns
        -------
        dict
            Dictionary with keys 'r_slice', 'phi_slice', 'theta_slice' (or 'z_slice')
            containing slice objects or None if no cut should be applied.
            Also includes 'cut_mask' for fine-grained filtering if needed.
        """
        if cut is None:
            return None
            
        dom = self.sim.domains
        
        # Get coordinate arrays
        if coords_system == 'spherical':
            r_arr = dom.r
            phi_arr = dom.phi
            theta_arr = dom.theta
            
            # Build full 3D mesh to detect affected regions
            theta_mesh, r_mesh, phi_mesh = np.meshgrid(
                theta_arr, r_arr, phi_arr, indexing='ij'
            )
            x = r_mesh * np.sin(theta_mesh) * np.cos(phi_mesh)
            y = r_mesh * np.sin(theta_mesh) * np.sin(phi_mesh)
            z = r_mesh * np.cos(theta_mesh)
            
        elif coords_system == 'cylindrical':
            r_arr = dom.r
            phi_arr = dom.phi
            z_arr = dom.z
            
            # Build full 3D mesh
            z_mesh, r_mesh, phi_mesh = np.meshgrid(
                z_arr, r_arr, phi_arr, indexing='ij'
            )
            x = r_mesh * np.cos(phi_mesh)
            y = r_mesh * np.sin(phi_mesh)
            z = z_mesh
        else:
            raise ValueError(f"Unsupported coordinate system: {coords_system}")
        
        # Compute mask based on cut geometry
        if len(cut) == 5:  # Cylinder
            xc, yc, zc, rc, hc = cut
            r_xy = np.sqrt((x - xc)**2 + (y - yc)**2)
            zmin, zmax = zc - hc/2, zc + hc/2
            mask_3d = (r_xy <= rc) & (z >= zmin) & (z <= zmax)
        elif len(cut) == 4:  # Sphere
            xc, yc, zc, rs = cut
            r_sph = np.sqrt((x - xc)**2 + (y - yc)**2 + (z - zc)**2)
            mask_3d = r_sph <= rs
        else:
            raise ValueError("cut must have 4 (sphere) or 5 (cylinder) elements.")
        
        # Find bounding indices in each dimension where mask is True
        # This preserves grid structure by using index ranges instead of boolean masks
        
        if coords_system == 'spherical':
            # Check which indices have any True values along other dimensions
            mask_theta = np.any(mask_3d, axis=(1, 2))  # Any True along r, phi
            mask_r = np.any(mask_3d, axis=(0, 2))      # Any True along theta, phi
            mask_phi = np.any(mask_3d, axis=(0, 1))    # Any True along theta, r
            
            # Get index ranges
            theta_idx = np.where(mask_theta)[0]
            r_idx = np.where(mask_r)[0]
            phi_idx = np.where(mask_phi)[0]
            
            if len(theta_idx) == 0 or len(r_idx) == 0 or len(phi_idx) == 0:
                # Empty cut - return None
                return None
            
            # Create slices with some padding to ensure we capture the cut region
            # Use slice objects to maintain structure
            result = {
                'theta_slice': slice(theta_idx[0], theta_idx[-1] + 1),
                'r_slice': slice(r_idx[0], r_idx[-1] + 1),
                'phi_slice': slice(phi_idx[0], phi_idx[-1] + 1),
                'cut_mask': mask_3d,  # Store full mask for precise filtering if needed
                'coords_system': 'spherical'
            }
            
        elif coords_system == 'cylindrical':
            # Check which indices have any True values
            mask_z = np.any(mask_3d, axis=(1, 2))
            mask_r = np.any(mask_3d, axis=(0, 2))
            mask_phi = np.any(mask_3d, axis=(0, 1))
            
            z_idx = np.where(mask_z)[0]
            r_idx = np.where(mask_r)[0]
            phi_idx = np.where(mask_phi)[0]
            
            if len(z_idx) == 0 or len(r_idx) == 0 or len(phi_idx) == 0:
                return None
            
            result = {
                'z_slice': slice(z_idx[0], z_idx[-1] + 1),
                'r_slice': slice(r_idx[0], r_idx[-1] + 1),
                'phi_slice': slice(phi_idx[0], phi_idx[-1] + 1),
                'cut_mask': mask_3d,
                'coords_system': 'cylindrical'
            }
        
        return result

    def _detect_regular_grid(self, var1_mesh, var2_mesh, var3_mesh):
        """
        Detect if 3D mesh data represents a regular grid and extract coordinate axes.
        
        A regular grid has uniform spacing along each dimension and can be represented
        as a Cartesian product of 1D axes (meshgrid). This enables fast interpolation
        with RegularGridInterpolator.
        
        Parameters
        ----------
        var1_mesh, var2_mesh, var3_mesh : np.ndarray
            3D coordinate meshes (shape must match).
            
        Returns
        -------
        is_regular : bool
            True if the grid is regular with strictly monotonic axes.
        axes : tuple of np.ndarray or None
            Tuple (var1_axis, var2_axis, var3_axis) if regular, else None.
        axis_order : tuple of int or None
            Order indicating which mesh dimension varies along each coordinate.
            For example, (0, 1, 2) means var1 varies along dim 0, etc.
        """
        try:
            # Check if arrays are 3D
            if var1_mesh.ndim != 3 or var2_mesh.ndim != 3 or var3_mesh.ndim != 3:
                return False, None, None
            
            if var1_mesh.shape != var2_mesh.shape or var1_mesh.shape != var3_mesh.shape:
                return False, None, None
            
            shape = var1_mesh.shape
            
            # Detect which dimension each coordinate varies along
            # Check variation using peak-to-peak (ptp) along slices
            var1_vars = [
                np.ptp(var1_mesh[:, 0, 0]),
                np.ptp(var1_mesh[0, :, 0]),
                np.ptp(var1_mesh[0, 0, :])
            ]
            var2_vars = [
                np.ptp(var2_mesh[:, 0, 0]),
                np.ptp(var2_mesh[0, :, 0]),
                np.ptp(var2_mesh[0, 0, :])
            ]
            var3_vars = [
                np.ptp(var3_mesh[:, 0, 0]),
                np.ptp(var3_mesh[0, :, 0]),
                np.ptp(var3_mesh[0, 0, :])
            ]
            
            var1_dim = np.argmax(var1_vars)
            var2_dim = np.argmax(var2_vars)
            var3_dim = np.argmax(var3_vars)
            
            # Each coordinate must vary along a different dimension
            if len(set([var1_dim, var2_dim, var3_dim])) != 3:
                return False, None, None
            
            # Extract 1D axes
            if var1_dim == 0:
                var1_axis = var1_mesh[:, 0, 0]
            elif var1_dim == 1:
                var1_axis = var1_mesh[0, :, 0]
            else:
                var1_axis = var1_mesh[0, 0, :]
                
            if var2_dim == 0:
                var2_axis = var2_mesh[:, 0, 0]
            elif var2_dim == 1:
                var2_axis = var2_mesh[0, :, 0]
            else:
                var2_axis = var2_mesh[0, 0, :]
                
            if var3_dim == 0:
                var3_axis = var3_mesh[:, 0, 0]
            elif var3_dim == 1:
                var3_axis = var3_mesh[0, :, 0]
            else:
                var3_axis = var3_mesh[0, 0, :]
            
            # Check monotonicity (strictly increasing or decreasing)
            def is_strictly_monotonic(arr):
                diff = np.diff(arr)
                return np.all(diff > 0) or np.all(diff < 0)
            
            if not (is_strictly_monotonic(var1_axis) and 
                    is_strictly_monotonic(var2_axis) and 
                    is_strictly_monotonic(var3_axis)):
                return False, None, None
            
            # Verify grid structure: reconstruct mesh and check it matches
            axes_ordered = [None, None, None]
            axes_ordered[var1_dim] = var1_axis
            axes_ordered[var2_dim] = var2_axis
            axes_ordered[var3_dim] = var3_axis
            
            # Create test meshgrid
            test_mesh = np.meshgrid(axes_ordered[0], axes_ordered[1], axes_ordered[2], indexing='ij')
            
            # Map back to check
            test_var1 = test_mesh[var1_dim]
            test_var2 = test_mesh[var2_dim]
            test_var3 = test_mesh[var3_dim]
            
            # Check if test meshes match original (within tolerance)
            tol = 1e-10
            if not (np.allclose(test_var1, var1_mesh, rtol=tol, atol=tol) and
                    np.allclose(test_var2, var2_mesh, rtol=tol, atol=tol) and
                    np.allclose(test_var3, var3_mesh, rtol=tol, atol=tol)):
                return False, None, None
            
            return True, (var1_axis, var2_axis, var3_axis), (var1_dim, var2_dim, var3_dim)
            
        except Exception:
            return False, None, None
 
    def load_data(self, fields=None, slice=None, snapshots=1, cut=None, coords='cartesian'):
        """Load one or multiple fields into a unified DataFrame.

        This method loads simulation data for the specified fields and snapshots,
        potentially applying a spatial slice or cut. The data is stored in
        an internal DataFrame (`self.df`) for further processing or interpolation.

        Parameters
        ----------
        fields : str or list of str
            Name(s) of the fields to load (e.g., 'gasdens', 'gasv').
        slice : str, optional
            Slice definition (e.g., 'z=0', 'theta=1.57').
        snapshots : int or list of int, optional
            Snapshot number(s) to load. Can be a single integer or a range [start, end].
        cut : list, optional
            Geometric cut parameters (sphere or cylinder mask).
        coords : str, optional
            Coordinate system for vector transformation ('cartesian' by default).

        Returns
        -------
        pd.DataFrame
            The DataFrame containing the loaded data.

        Examples
        --------
        Load density and velocity for snapshot 10:

        >>> loader = fp.FieldInterpolator(sim)
        >>> df = loader.load_data(fields=['gasdens', 'gasv'], snapshot=10)

        Load a 2D slice at z=0 (midplane):

        >>> df_slice = loader.load_data(fields='gasdens', slice='z=0', snapshot=10)
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

        # Convert snapshot into list - handle int, numpy types, and iterables
        if snapshots is None:
            snapshots = [0]
        elif isinstance(snapshots, (int, np.integer)):
            snapshots = [int(snapshots)]
        elif not isinstance(snapshots, list):
            # Convert arrays, tuples, etc. to list
            try:
                snapshots = list(snapshots)
            except TypeError:
                snapshots = [int(snapshots)]
        
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
                            # Handle 2D (2 components) or 3D (3 components)
                            ncomponents = len(v_slice)
                            if ncomponents == 2:
                                v1, v2 = v_slice[0], v_slice[1]
                                if not coords_assigned:
                                    vnames = getattr(self.sim.vars, 'VARIABLES', ['x', 'y', 'z'])
                                    row['var1_mesh'] = getattr(mesh, vnames[0])
                                    row['var2_mesh'] = getattr(mesh, vnames[1])
                                    coords_assigned = True
                                row['gasv_mesh'] = np.array([v1, v2])
                            else:  # 3D
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

            # Determine coordinate system
            coords_sys = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'
            
            # Compute structured cut indices (preserves grid structure)
            cut_info = None
            if cut is not None:
                cut_info = self._compute_cut_indices(cut, coords_system=coords_sys)
                if cut_info is None:
                    raise ValueError("The specified cut does not intersect the simulation domain.")
            
            # Build mesh with optional structured cut
            if coords_sys == 'spherical':
                # Get coordinate arrays
                r_arr = self.sim.domains.r
                phi_arr = self.sim.domains.phi
                theta_arr = self.sim.domains.theta
                
                # Apply structured slicing if cut is specified
                if cut_info is not None:
                    theta_slice = cut_info['theta_slice']
                    r_slice = cut_info['r_slice']
                    phi_slice = cut_info['phi_slice']
                    
                    theta_arr = theta_arr[theta_slice]
                    r_arr = r_arr[r_slice]
                    phi_arr = phi_arr[phi_slice]
                
                # Build 3D structured mesh from sliced arrays
                theta, r, phi = np.meshgrid(theta_arr, r_arr, phi_arr, indexing='ij')
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
            elif coords_sys == 'cylindrical':
                # Get coordinate arrays
                r_arr = self.sim.domains.r
                phi_arr = self.sim.domains.phi
                z_arr = self.sim.domains.z
                
                # Apply structured slicing if cut is specified
                if cut_info is not None:
                    z_slice = cut_info['z_slice']
                    r_slice = cut_info['r_slice']
                    phi_slice = cut_info['phi_slice']
                    
                    z_arr = z_arr[z_slice]
                    r_arr = r_arr[r_slice]
                    phi_arr = phi_arr[phi_slice]
                
                # Build 3D structured mesh
                z, r, phi = np.meshgrid(z_arr, r_arr, phi_arr, indexing='ij')
                x = r * np.cos(phi)
                y = r * np.sin(phi)
            else:
                raise ValueError(f"Unsupported coordinate system: {coords_sys}")

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
                                row["var1_mesh"] = x
                                row["var2_mesh"] = y
                                row["var3_mesh"] = z
                            else:
                                # original coordinate variables order
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r, phi=phi, theta=theta, x=x, y=y, z=z)
                                row["var1_mesh"] = mapping[v0]
                                row["var2_mesh"] = mapping[v1]
                                row["var3_mesh"] = mapping[v2]
                            coords_assigned = True

                        # Apply structured slicing to data
                        if cut_info is not None:
                            if coords_sys == 'spherical':
                                data_sliced = gasd.data[
                                    cut_info['theta_slice'],
                                    cut_info['r_slice'],
                                    cut_info['phi_slice']
                                ]
                            else:  # cylindrical
                                data_sliced = gasd.data[
                                    cut_info['z_slice'],
                                    cut_info['r_slice'],
                                    cut_info['phi_slice']
                                ]
                            row["gasdens_mesh"] = data_sliced
                        else:
                            row["gasdens_mesh"] = gasd.data
                            
                    # -----------------
                    # GASV 3D
                    # -----------------
                    if field == "gasv":
                        gasv_raw = self.sim._load_field_raw('gasv', snapshot=snap, field_type='vector')
                        
                        if coords == 'cartesian':
                            gasvx, gasvy, gasvz = gasv_raw.to_cartesian()

                            if not coords_assigned:
                                row["var1_mesh"] = x
                                row["var2_mesh"] = y
                                row["var3_mesh"] = z
                                coords_assigned = True

                            # Apply structured slicing to vector components
                            if cut_info is not None:
                                if coords_sys == 'spherical':
                                    row["gasv_mesh"] = np.array([
                                        gasvx.data[cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        gasvy.data[cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        gasvz.data[cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']]
                                    ])
                                else:  # cylindrical
                                    row["gasv_mesh"] = np.array([
                                        gasvx.data[cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        gasvy.data[cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        gasvz.data[cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']]
                                    ])
                            else:
                                row["gasv_mesh"] = np.array([gasvx.data, gasvy.data, gasvz.data])
                        else:
                            vdata = gasv_raw.data
                            if not coords_assigned:
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r, phi=phi, theta=theta, x=x, y=y, z=z)
                                row["var1_mesh"] = mapping[v0]
                                row["var2_mesh"] = mapping[v1]
                                row["var3_mesh"] = mapping[v2]
                                coords_assigned = True

                            # Apply structured slicing to vector components
                            if cut_info is not None:
                                if coords_sys == 'spherical':
                                    row["gasv_mesh"] = np.array([
                                        vdata[0][cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        vdata[1][cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        vdata[2][cut_info['theta_slice'], cut_info['r_slice'], cut_info['phi_slice']]
                                    ])
                                else:  # cylindrical
                                    row["gasv_mesh"] = np.array([
                                        vdata[0][cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        vdata[1][cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']],
                                        vdata[2][cut_info['z_slice'], cut_info['r_slice'], cut_info['phi_slice']]
                                    ])
                            else:
                                row["gasv_mesh"] = np.array([vdata[0], vdata[1], vdata[2]])

                    # -----------------
                    # GASENERGY 3D
                    # -----------------
                    if field == "gasenergy":
                        gasen = self.sim._load_field_raw('gasenergy', snapshot=snap, field_type='scalar')

                        if not coords_assigned:
                            if coords == 'cartesian':
                                row["var1_mesh"] = x
                                row["var2_mesh"] = y
                                row["var3_mesh"] = z
                            else:
                                v0, v1, v2 = self.sim.vars.VARIABLES
                                mapping = dict(r=r, phi=phi, theta=theta, x=x, y=y, z=z)
                                row["var1_mesh"] = mapping[v0]
                                row["var2_mesh"] = mapping[v1]
                                row["var3_mesh"] = mapping[v2]
                            coords_assigned = True

                        # Apply structured slicing to data
                        if cut_info is not None:
                            if coords_sys == 'spherical':
                                data_sliced = gasen.data[
                                    cut_info['theta_slice'],
                                    cut_info['r_slice'],
                                    cut_info['phi_slice']
                                ]
                            else:  # cylindrical
                                data_sliced = gasen.data[
                                    cut_info['z_slice'],
                                    cut_info['r_slice'],
                                    cut_info['phi_slice']
                                ]
                            row["gasenergy_mesh"] = data_sliced
                        else:
                            row["gasenergy_mesh"] = gasen.data

                # collect row dicts and build DataFrame once
                rows.append(row)

            df_snapshots = pd.DataFrame(rows)
            
            # Detect if this is a regular grid
            # With structured cuts, the grid remains regular!
            if len(rows) > 0:
                first_row = rows[0]
                var1_test = first_row["var1_mesh"]
                var2_test = first_row["var2_mesh"]  
                var3_test = first_row["var3_mesh"]
                
                is_regular, axes, axis_order = self._detect_regular_grid(
                    var1_test, var2_test, var3_test
                )
                
                self._is_regular_grid = is_regular
                self._grid_axes = axes
                self._axis_order = axis_order
                self._original_coords = coords
            else:
                self._is_regular_grid = False
                self._grid_axes = None
                self._axis_order = None
                self._original_coords = coords
            
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
        nphi=50,
        nz=50
    ):
        """
        Create a mesh grid based on the slice definition provided by the user.
        If no slice is provided, create a full 3D mesh within the simulation domain.
        Supports both spherical and cylindrical coordinate systems.

        Parameters
        ----------
        slice : str, optional
            The slice definition string (e.g., "r=[0.8,1.2],phi=0,theta=[0 deg,90 deg]" for spherical
            or "r=[0.8,1.2],phi=0,z=[0,0.5]" for cylindrical).
        nr : int
            Number of divisions in r.
        ntheta : int
            Number of divisions in theta (spherical only).
        nphi : int
            Number of divisions in phi.
        nz : int
            Number of divisions in z (cylindrical only).

        Returns
        -------
        tuple
            Mesh grid (x, y, z) based on the slice definition or the full domain.
        """
        import numpy as np
        import re

        # Get coordinate system
        coords = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'

        # If no slice is provided, create a full 3D mesh using the simulation domains
        if not slice:
            if coords == 'spherical':
                r = np.linspace(self.sim.domains.r.min(), self.sim.domains.r.max(), nr)
                theta = np.linspace(self.sim.domains.theta.min(), self.sim.domains.theta.max(), ntheta)
                phi = np.linspace(self.sim.domains.phi.min(), self.sim.domains.phi.max(), nphi)
                theta_grid, r_grid, phi_grid = np.meshgrid(theta, r, phi, indexing='ij')
                x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
                y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
                z = r_grid * np.cos(theta_grid)
                return x, y, z
            elif coords == 'cylindrical':
                r = np.linspace(self.sim.domains.r.min(), self.sim.domains.r.max(), nr)
                z = np.linspace(self.sim.domains.z.min(), self.sim.domains.z.max(), nz)
                phi = np.linspace(self.sim.domains.phi.min(), self.sim.domains.phi.max(), nphi)
                z_grid, r_grid, phi_grid = np.meshgrid(z, r, phi, indexing='ij')
                x = r_grid * np.cos(phi_grid)
                y = r_grid * np.sin(phi_grid)
                return x, y, z_grid

        # Initialize default ranges based on coordinate system
        r_range = [self.sim.domains.r.min(), self.sim.domains.r.max()]
        phi_range = [self.sim.domains.phi.min(), self.sim.domains.phi.max()]
        
        if coords == 'spherical':
            theta_range = [self.sim.domains.theta.min(), self.sim.domains.theta.max()]
            z_range = None
        else:  # cylindrical
            theta_range = None
            z_range = [self.sim.domains.z.min(), self.sim.domains.z.max()]

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
                theta_range = values if coords == 'spherical' else theta_range
            elif key == 'z':
                z_range = values if coords == 'cylindrical' else z_range

        # Process single values
        z_value = None
        theta_value = None
        for match in value_pattern.finditer(slice):
            key, value = match.groups()
            value = float(degree_pattern.sub(lambda m: str(float(m.group(1)) * np.pi / 180), value))
            if key == 'z':
                z_value = value
                if coords == 'cylindrical':
                    z_range = [value, value]
            elif key == 'phi':
                phi_range = [value, value]
            elif key == 'theta':
                theta_value = value
                if coords == 'spherical':
                    theta_range = [value, value]

        # SPHERICAL COORDINATES
        if coords == 'spherical':
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

            elif z_value is not None:  # Slice at constant z (XY plane in Cartesian)
                r = np.linspace(r_range[0], r_range[1], nr)
                phi = np.linspace(phi_range[0], phi_range[1], nphi)
                r_grid, phi_grid = np.meshgrid(r, phi, indexing='ij')
                x = r_grid * np.cos(phi_grid)
                y = r_grid * np.sin(phi_grid)
                z = np.full_like(x, z_value)
                return x, y, z
            else:
                raise ValueError("Slice definition for spherical coordinates must include either 'z', 'phi', or 'theta'.")

        # CYLINDRICAL COORDINATES
        elif coords == 'cylindrical':
            # 3D mesh: all ranges are intervals
            if (phi_range[0] != phi_range[1]) and (z_range[0] != z_range[1]):
                r = np.linspace(r_range[0], r_range[1], nr)
                z = np.linspace(z_range[0], z_range[1], nz)
                phi = np.linspace(phi_range[0], phi_range[1], nphi)
                z_grid, r_grid, phi_grid = np.meshgrid(z, r, phi, indexing='ij')
                x = r_grid * np.cos(phi_grid)
                y = r_grid * np.sin(phi_grid)
                return x, y, z_grid

            # 2D mesh: one coordinate is fixed (slice)
            elif phi_range[0] == phi_range[1]:  # Slice at constant phi (RZ plane)
                r = np.linspace(r_range[0], r_range[1], nr)
                z = np.linspace(z_range[0], z_range[1], nz)
                phi = phi_range[0]
                z_grid, r_grid = np.meshgrid(z, r, indexing='ij')
                x = r_grid * np.cos(phi)
                y = r_grid * np.sin(phi)
                return x, y, z_grid

            elif z_range[0] == z_range[1]:  # Slice at constant z (XY plane)
                r = np.linspace(r_range[0], r_range[1], nr)
                phi = np.linspace(phi_range[0], phi_range[1], nphi)
                z = z_range[0]
                phi_grid, r_grid = np.meshgrid(phi, r, indexing='ij')
                x = r_grid * np.cos(phi_grid)
                y = r_grid * np.sin(phi_grid)
                z_out = np.full_like(x, z)
                return x, y, z_out
                
            else:
                raise ValueError("Slice definition for cylindrical coordinates must include either 'z', 'phi', or 'r'.")
        
        else:
            raise ValueError(f"Unsupported coordinate system: {coords}")



    def _domain_mask(self, xi, slice=None):
        """
        Build a boolean mask that keeps only coordinates within the simulation domain and
        enforces any user-specified radial/angle/z limits for various slice types.
        Supports both spherical and cylindrical coordinate systems.
        """
        self._cache_domain_limits()
        slice = slice or self.slice
        slice_ranges = self._slice_ranges or self._parse_slice_ranges(slice)
        
        # Get coordinate system
        coords = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'
        
        r_bounds = slice_ranges.get('r')
        theta_bounds = slice_ranges.get('theta')
        phi_bounds = slice_ranges.get('phi')
        z_bounds = slice_ranges.get('z')
        
        r_min, r_max = self._domain_limits['r']
        phi_min, phi_max = self._domain_limits['phi']
        
        # Get theta or z limits depending on coordinate system
        if coords == 'spherical':
            theta_min, theta_max = self._domain_limits['theta']
            z_min, z_max = None, None
        else:  # cylindrical
            theta_min, theta_max = None, None
            z_min, z_max = self._domain_limits['z']
        
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
            # Handle 2D slices based on coordinate system
            if coords == 'spherical':
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
                    r = np.sqrt(x**2 + y**2)
                    phi = np.arctan2(y, x)
                    mask = (r > r_min) & (r < r_max)
                    return mask
                    
            else:  # cylindrical
                # XY plane: z fixed
                if slice is not None and 'z' in slice:
                    # XY plane: z fixed, filter by r and phi
                    x, y = xi[:, 0], xi[:, 1]
                    r = np.sqrt(x**2 + y**2)
                    phi = np.arctan2(y, x)
                    r_ge, r_le = _bounded(r, r_bounds, (r_min, r_max))
                    mask = r_ge & r_le & _phi_in_bounds(phi)
                    return mask
                
                # RZ plane: phi fixed
                elif slice is not None and 'phi' in slice:
                    # RZ plane: phi fixed, filter by r and z
                    # Assuming xi[:, 0] is r-like and xi[:, 1] is z-like
                    x, y = xi[:, 0], xi[:, 1]
                    # Convert to cylindrical r, z
                    r = np.sqrt(x**2 + y**2)
                    # For phi-slice in cylindrical, we may receive (x,z) coordinates
                    # Need to determine which is which based on the slice
                    z_ge, z_le = _bounded(y, z_bounds, (z_min, z_max))
                    r_ge, r_le = _bounded(r, r_bounds, (r_min, r_max))
                    return r_ge & r_le & z_ge & z_le
                else:
                    # Default: treat as XY (z fixed)
                    x, y = xi[:, 0], xi[:, 1]
                    r = np.sqrt(x**2 + y**2)
                    phi = np.arctan2(y, x)
                    mask = (r > r_min) & (r < r_max)
                    return mask

        elif ndim == 1:
            # 1D input: could be r, theta, phi (spherical) or r, z, phi (cylindrical)
            xi_1d = np.asarray(xi).ravel()

            # Decide which variable is the "free" one in the 1D cut.
            # Prefer explicit ranges (r=[...], theta=[...], phi=[...], z=[...]); otherwise,
            # the free variable is the one that does NOT appear as a scalar in the slice.
            r_b = r_bounds
            th_b = theta_bounds
            ph_b = phi_bounds
            z_b = z_bounds

            def _is_range(b):
                return (b is not None) and (abs(b[1] - b[0]) > 1e-12)

            if _is_range(r_b):
                free = 'r'
            elif _is_range(th_b):
                free = 'theta'
            elif _is_range(ph_b):
                free = 'phi'
            elif _is_range(z_b):
                free = 'z'
            else:
                s = (slice or self.slice) or ""
                s_low = s.lower()
                has_r = re.search(r"\br\s*=", s_low) is not None
                has_th = re.search(r"\btheta\s*=", s_low) is not None
                has_ph = re.search(r"\bphi\s*=", s_low) is not None
                has_z = re.search(r"\bz\s*=", s_low) is not None
                # the free variable is the one not present in the slice specification
                if coords == 'cylindrical':
                    if not has_r:
                        free = 'r'
                    elif not has_z:
                        free = 'z'
                    elif not has_ph:
                        free = 'phi'
                    else:
                        free = 'r'
                else:  # spherical
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
                if theta_min is not None and theta_max is not None:
                    lo, hi = (th_b if th_b is not None else (theta_min, theta_max))
                    mask = (xi_1d >= lo - eps) & (xi_1d <= hi + eps)
                    return mask
                else:
                    return np.ones_like(xi_1d, dtype=bool)

            if free == 'z':
                if z_min is not None and z_max is not None:
                    lo, hi = (z_b if z_b is not None else (z_min, z_max))
                    mask = (xi_1d >= lo - eps) & (xi_1d <= hi + eps)
                    return mask
                else:
                    return np.ones_like(xi_1d, dtype=bool)

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
            self, var1, snapshot=None, time=None, var2=None, var3=None, dataframe=None,
            interpolator="griddata", method="linear",
            rbf_kwargs=None, griddata_kwargs=None, idw_kwargs=None,
            sigma_smooth=None, field=None, reflect=False, coords=None
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
            Evaluation coordinates. For 3D: typically (x, y, z) if coords='cartesian',
            (phi, r, theta) if coords='spherical', or (phi, r, z) if coords='cylindrical'.
            The coordinate system is determined by the 'coords' parameter.
            Scalars are accepted.

        dataframe : pandas.DataFrame, optional
            Custom DataFrame. If omitted, self.df is used.

        interpolator : {"griddata", "rbf", "linearnd", "idw", "regular_grid"}, default "griddata"
            Backend used for spatial interpolation:
            - "griddata": scipy.interpolate.griddata (default, works for irregular grids)
            - "rbf": Radial Basis Function interpolation
            - "linearnd": LinearNDInterpolator
            - "idw": Inverse Distance Weighting
            - "regular_grid": scipy.interpolate.RegularGridInterpolator (automatic for
              3D regular grids, fastest option when applicable)
            
            Note: For 3D data with regular grid structure, RegularGridInterpolator is
            automatically used by default regardless of the interpolator setting, with
            automatic fallback to griddata if the grid is irregular.

        method : str, default "linear"
            Kernel/method used by backend (e.g., 'linear', 'cubic' for griddata).

        sigma_smooth : float or None
            Optional Gaussian smoothing parameter.

        field : {"gasdens", "gasv", "gasenergy"} or None
            Field to evaluate. If None and DF has exactly one field, it is auto-selected.
            If multiple fields exist, explicit selection is required.

        reflect : bool, default False
            If True, enable symmetry-based reflection for interpolation.
            When interpolating at points outside the loaded domain, the method
            attempts to evaluate at the corresponding reflected point and uses that value.
            
            The reflection is performed in the coordinate system specified by the 'coords'
            parameter used in load_field():
            - Cartesian: reflects across z=0 (z -> -z)
            - Spherical: reflects across equatorial plane (theta -> π - theta)
            - Cylindrical: reflects across z=0 (z -> -z)
            
            This is useful for enforcing symmetry in simulations that are symmetric
            about the equatorial/midplane. Works efficiently with RegularGridInterpolator
            by only reflecting points that fall outside the domain (NaN values).

        coords : {"cartesian", "spherical", "cylindrical", None}, default None
            Target coordinate system for the evaluation points (var1, var2, var3).
            If None (default), uses the original coordinate system from load_data.
            
            - "cartesian": var1=x, var2=y, var3=z
            - "spherical": var1=phi, var2=r, var3=theta
            - "cylindrical": var1=phi, var2=r, var3=z
            
            The method automatically converts between coordinate systems as needed.
            This allows querying data in any coordinate system regardless of how
            it was originally loaded.

        Returns
        -------
        ndarray or float
            Interpolated value(s). Vector fields return shape (3, N) or (3, ...).

        Examples
        --------
        Evaluate density at Cartesian coordinates:
        
        >>> loader.evaluate(var1=x_points, var2=y_points, var3=z_points, 
        ...                 field='gasdens', time=0.5, coords='cartesian')
        
        Evaluate in spherical coordinates:
        
        >>> loader.evaluate(var1=phi_vals, var2=r_vals, var3=theta_vals,
        ...                 field='gasdens', time=0.5, coords='spherical')

        Notes
        -----
        For 3D regular grids (uniform spacing in each dimension), the method
        automatically uses RegularGridInterpolator for optimal performance,
        regardless of the 'interpolator' parameter. This provides significant
        speedup (often 10-100x) compared to griddata for large datasets.
        
        The coordinate transformation feature allows seamless querying in different
        coordinate systems. The original simulation data coordinate system is
        preserved internally, and transformations are applied automatically.
        """
        # ===============================================================
        # Time or snap
        # ===============================================================
        if time is None:
            if snapshot is None:
                raise ValueError("Must provide either 'time' or 'snapshot'.")
            time = snapshot
        
        # ===============================================================
        # Coordinate system handling
        # ===============================================================
        # Determine target coordinate system (default: original from load_data)
        if coords is None:
            coords_target = self._original_coords if self._original_coords else 'cartesian'
        else:
            coords_target = coords
        
        # Store original coordinate system from data
        coords_original = self._original_coords if self._original_coords else 'cartesian'
        
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
        # 2D XZ or RZ slices where the expected coordinates are (var1,var3).
        # If the slice is XZ/RZ (not XY) and user passed var2 but left var3=None,
        # treat var2 as var3.
        try:
            slice_type_tmp = self._slice_type or self._detect_slice_type(self.slice)
            coords_sys = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'
        except Exception:
            slice_type_tmp = None
            coords_sys = 'spherical'
        
        # Determine if this is an XY plane or XZ/RZ plane
        is_xy_plane = False
        if coords_sys == 'spherical':
            is_xy_plane = (slice_type_tmp == 'theta')
        else:  # cylindrical
            is_xy_plane = (slice_type_tmp == 'z')
        
        if self.dim == 2 and slice_type_tmp is not None and not is_xy_plane:
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
        # Coordinate transformation helpers
        # ===============================================================
        def transform_coords(var1_in, var2_in, var3_in, from_system, to_system):
            """Transform coordinates between cartesian, spherical, and cylindrical systems."""
            if from_system == to_system:
                return var1_in, var2_in, var3_in
            
            # Convert to arrays
            v1 = np.asarray(var1_in)
            v2 = np.asarray(var2_in) if var2_in is not None else np.zeros_like(v1)
            v3 = np.asarray(var3_in) if var3_in is not None else np.zeros_like(v1)
            
            # First convert to Cartesian (common intermediate)
            if from_system == 'cartesian':
                x, y, z = v1, v2, v3
            elif from_system == 'spherical':
                # spherical: var1=phi, var2=r, var3=theta
                phi, r, theta = v1, v2, v3
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
            elif from_system == 'cylindrical':
                # cylindrical: var1=phi, var2=r, var3=z
                phi, r_cyl, z = v1, v2, v3
                x = r_cyl * np.cos(phi)
                y = r_cyl * np.sin(phi)
            else:
                raise ValueError(f"Unknown coordinate system: {from_system}")
            
            # Convert from Cartesian to target system
            if to_system == 'cartesian':
                return x, y, z
            elif to_system == 'spherical':
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arccos(np.clip(z / (r + 1e-30), -1, 1))
                phi = np.arctan2(y, x)
                return phi, r, theta
            elif to_system == 'cylindrical':
                r_cyl = np.sqrt(x**2 + y**2)
                phi = np.arctan2(y, x)
                return phi, r_cyl, z
            else:
                raise ValueError(f"Unknown coordinate system: {to_system}")

        # ===============================================================
        # Interpolation backends
        # ===============================================================

        def regular_grid_interp(axes, values, xi, axis_order):
            """
            Fast interpolation for regular grids using RegularGridInterpolator.
            
            Parameters
            ----------
            axes : tuple of 1D arrays
                Coordinate axes (var1_axis, var2_axis, var3_axis).
            values : ndarray
                Data values on the regular grid.
            xi : ndarray
                Query points (N, 3).
            axis_order : tuple of int
                Dimension order (var1_dim, var2_dim, var3_dim).
                
            Returns
            -------
            ndarray
                Interpolated values at query points.
            """
            var1_axis, var2_axis, var3_axis = axes
            var1_dim, var2_dim, var3_dim = axis_order
            n_query = xi.shape[0]
            
            # Create axes in the order matching the data array dimensions
            axes_ordered = [None, None, None]
            axes_ordered[var1_dim] = var1_axis
            axes_ordered[var2_dim] = var2_axis
            axes_ordered[var3_dim] = var3_axis
            
            # Handle reversed axes (decreasing instead of increasing)
            flips = [False, False, False]
            for i, ax in enumerate(axes_ordered):
                if len(ax) > 1 and np.diff(ax)[0] < 0:
                    axes_ordered[i] = np.flip(ax)
                    flips[i] = True
            
            # Flip data array accordingly
            values_fixed = values.copy()
            for i, flip in enumerate(flips):
                if flip:
                    values_fixed = np.flip(values_fixed, axis=i)
            
            # Create interpolator
            try:
                interp_obj = RegularGridInterpolator(
                    tuple(axes_ordered),
                    values_fixed,
                    method=method if method in ['linear', 'nearest'] else 'linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
            except Exception as e:
                # Fallback to None if creation fails
                return None
            
            # Prepare query points: xi has columns [var1, var2, var3]
            # We need to reorder them to [dim0_coord, dim1_coord, dim2_coord]
            xi_ordered_list = [None, None, None]
            xi_ordered_list[var1_dim] = xi[:, 0]
            xi_ordered_list[var2_dim] = xi[:, 1]
            xi_ordered_list[var3_dim] = xi[:, 2]
            xi_ordered = np.column_stack(xi_ordered_list)
            
            # Interpolate with automatic chunking for large datasets
            chunk_size = 5_000_000
            
            if n_query > chunk_size:
                # Use chunking for large queries to avoid memory issues
                result = np.empty(n_query, dtype=np.float64)
                n_chunks = int(np.ceil(n_query / chunk_size))
                
                for i in range(n_chunks):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, n_query)
                    chunk_xi = xi_ordered[start_idx:end_idx]
                    
                    try:
                        result[start_idx:end_idx] = interp_obj(chunk_xi)
                    except Exception as e:
                        return None
                
                return result
            else:
                # Small query - no chunking needed
                try:
                    return interp_obj(xi_ordered)
                except Exception as e:
                    return None

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

            # =========================================================
            # COORDINATE TRANSFORMATION (SIMPLIFIED)
            # =========================================================
            # Only apply transformation if explicitly requested and different from original
            # Default: use coordinates as-is (no transformation)
            if (self.dim == 3 and 
                coords is not None and 
                coords_target != coords_original and 
                var1 is not None):
                # User explicitly requested different coordinate system
                try:
                    var1_eval = var1
                    var2_eval = var2 if var2 is not None else np.zeros_like(var1)
                    var3_eval = var3 if var3 is not None else np.zeros_like(var1)
                    
                    # Transform coordinates
                    var1_transformed, var2_transformed, var3_transformed = transform_coords(
                        var1_eval, var2_eval, var3_eval,
                        from_system=coords_target,
                        to_system=coords_original
                    )
                    var1_use = var1_transformed
                    var2_use = var2_transformed
                    var3_use = var3_transformed
                except Exception as e:
                    # If transformation fails, use original coordinates
                    var1_use = var1
                    var2_use = var2
                    var3_use = var3
            else:
                # No transformation needed - use original coordinates
                var1_use = var1
                var2_use = var2
                var3_use = var3

            # Build coordinate arrays
            if self.dim == 3:
                points_array = np.column_stack((cx.ravel(), cy.ravel(), cz.ravel()))
                xi = np.column_stack((var1_use.ravel(), var2_use.ravel(), var3_use.ravel()))
                
                # =========================================================
                # TRY REGULAR GRID INTERPOLATOR FOR 3D (FAST PATH)
                # =========================================================
                # Use RegularGridInterpolator if grid is regular (much faster)
                # With reflect=True, we still use RegularGridInterpolator but handle
                # reflection by evaluating reflected points when primary evaluation fails
                use_regular_grid = (self._is_regular_grid and 
                                   self._grid_axes is not None and 
                                   self._axis_order is not None)
                
                if use_regular_grid:
                    # Select field data
                    data_3d = row[field_name]
                    
                    # Handle vector component selection before interpolation
                    if isinstance(data_3d, np.ndarray) and comp is not None:
                        if data_3d.ndim == 4 and data_3d.shape[0] == 3:
                            data_3d = data_3d[comp]
                        elif data_3d.ndim == 2 and data_3d.shape[0] == 3:
                            # Might be already flattened (for vector in masked case)
                            expected_size = cx.size
                            if data_3d.shape[1] == expected_size:
                                data_3d = data_3d[comp].reshape(cx.shape)
                    
                    # Try regular grid interpolation
                    if isinstance(data_3d, np.ndarray) and data_3d.ndim == 3:
                        result_rg = regular_grid_interp(
                            self._grid_axes, data_3d, xi, self._axis_order
                        )
                        
                        if result_rg is not None:
                            # Success with regular grid interpolator!
                            
                            # Handle reflection if requested
                            if reflect:
                                # Strategy: For all points, attempt reflection to fill gaps
                                # Priority: use primary value if available, otherwise use reflected value
                                # Special case: for points very close to symmetry plane, try both
                                
                                # Calculate ALL reflected coordinates
                                xi_reflected = xi.copy()
                                
                                # Determine symmetry plane location and tolerance
                                if coords_original == 'spherical':
                                    # Symmetry plane is at theta = π/2 (equatorial plane)
                                    # Reflect theta: theta -> π - theta (index 2)
                                    xi_reflected[:, 2] = np.pi - xi_reflected[:, 2]
                                    symmetry_coord = xi[:, 2]  # theta values
                                    symmetry_plane = np.pi / 2
                                    # Tolerance: consider "near symmetry plane" as within ~2 grid spacings
                                    if self._grid_axes is not None and len(self._grid_axes) > 2:
                                        coord_axis = self._grid_axes[2] if self._axis_order[2] == 2 else self._grid_axes[self._axis_order.index(2)]
                                        if len(coord_axis) > 1:
                                            grid_spacing = np.min(np.abs(np.diff(coord_axis)))
                                            tolerance = 2.0 * grid_spacing
                                        else:
                                            tolerance = 0.05  # fallback ~3 degrees
                                    else:
                                        tolerance = 0.05
                                elif coords_original == 'cylindrical':
                                    # Symmetry plane is at z = 0 (midplane)
                                    # Reflect z: z -> -z (index 2)
                                    xi_reflected[:, 2] *= -1
                                    symmetry_coord = xi[:, 2]  # z values
                                    symmetry_plane = 0.0
                                    if self._grid_axes is not None and len(self._grid_axes) > 2:
                                        coord_axis = self._grid_axes[2] if self._axis_order[2] == 2 else self._grid_axes[self._axis_order.index(2)]
                                        if len(coord_axis) > 1:
                                            grid_spacing = np.min(np.abs(np.diff(coord_axis)))
                                            tolerance = 2.0 * grid_spacing
                                        else:
                                            tolerance = 0.1
                                    else:
                                        tolerance = 0.1
                                else:  # cartesian
                                    # Symmetry plane is at z = 0 (midplane)
                                    # Reflect z: z -> -z (index 2)
                                    xi_reflected[:, 2] *= -1
                                    symmetry_coord = xi[:, 2]  # z values
                                    symmetry_plane = 0.0
                                    tolerance = 0.1
                                
                                # Identify points near symmetry plane
                                distance_to_plane = np.abs(symmetry_coord - symmetry_plane)
                                near_plane_mask = distance_to_plane < tolerance
                                
                                # Interpolate at ALL reflected points
                                result_reflected = regular_grid_interp(
                                    self._grid_axes, data_3d, xi_reflected, self._axis_order
                                )
                                
                                if result_reflected is not None:
                                    # Apply sign flip for perpendicular component
                                    if field_name == 'gasv_mesh' and comp == 2:
                                        result_reflected = result_reflected * (-1)
                                    
                                    # Identify NaN values in primary interpolation
                                    nan_mask = np.isnan(result_rg)
                                    
                                    # Strategy 1: Fill NaN values with reflected values
                                    valid_reflected = ~np.isnan(result_reflected)
                                    fill_mask = nan_mask & valid_reflected
                                    result_rg[fill_mask] = result_reflected[fill_mask]
                                    
                                    # Strategy 2: For points near symmetry plane with both values available,
                                    # use average to smooth transition (optional, more robust)
                                    both_valid = (~nan_mask) & valid_reflected & near_plane_mask
                                    if np.any(both_valid):
                                        # Average primary and reflected values for points very close to plane
                                        result_rg[both_valid] = 0.5 * (result_rg[both_valid] + result_reflected[both_valid])
                                    
                                    # Strategy 3: Fill remaining NaN values in the gap at symmetry plane
                                    # This handles cases where data doesn't quite reach the symmetry plane
                                    # (e.g., FARGO3D data ending at theta=89.92° instead of 90°)
                                    still_nan = np.isnan(result_rg)
                                    if np.any(still_nan):
                                        # Only fill points very close to symmetry plane (in the gap)
                                        gap_points = still_nan & (distance_to_plane < tolerance)
                                        
                                        if np.any(gap_points):
                                            # Use nearest neighbor interpolation ONLY for gap points
                                            # This is much better than global nearest: it's a smart local fill
                                            from scipy.spatial import cKDTree
                                            
                                            # Get all valid points (original or reflected)
                                            all_valid = ~still_nan
                                            if np.any(all_valid):
                                                # Build KDTree from valid points
                                                valid_coords = xi[all_valid]
                                                valid_values = result_rg[all_valid]
                                                tree = cKDTree(valid_coords)
                                                
                                                # Find nearest valid point for each gap point
                                                gap_coords = xi[gap_points]
                                                distances, indices = tree.query(gap_coords, k=1)
                                                
                                                # Only fill if nearest point is very close (within tolerance)
                                                # This prevents filling with distant values
                                                close_enough = distances < tolerance
                                                gap_indices = np.where(gap_points)[0]
                                                fill_indices = gap_indices[close_enough]
                                                
                                                if len(fill_indices) > 0:
                                                    result_rg[fill_indices] = valid_values[indices[close_enough]]
                            
                            return result_rg
            elif self.dim == 2:
                # Detect coordinate system
                coords_sys = self.sim.vars.COORDINATES if hasattr(self.sim, 'vars') else 'spherical'
                
                # Determine which plane we're working with
                # Spherical: theta=const → XY plane, phi=const → XZ plane
                # Cylindrical: z=const → XY plane, phi=const → RZ plane
                if coords_sys == 'spherical':
                    if slice_type == "theta":
                        points_array = np.column_stack((cx.ravel(), cy.ravel()))
                        xi = np.column_stack((var1_use.ravel(), var2_use.ravel()))
                    else:  # phi or r
                        points_array = np.column_stack((cx.ravel(), cz.ravel()))
                        xi = np.column_stack((var1_use.ravel(), var3_use.ravel()))
                else:  # cylindrical
                    if slice_type == "z":
                        points_array = np.column_stack((cx.ravel(), cy.ravel()))
                        xi = np.column_stack((var1_use.ravel(), var2_use.ravel()))
                    else:  # phi or r
                        points_array = np.column_stack((cx.ravel(), cz.ravel()))
                        xi = np.column_stack((var1_use.ravel(), var3_use.ravel()))
            elif self.dim == 1:
                points_array = np.sqrt(cx**2 + cy**2 + cz**2)
                xi = np.asarray(var1_use)

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
            # If `reflect=True` we augment the interpolation dataset
            # by reflecting across the equatorial plane.
            # 
            # The reflection is done in the ORIGINAL coordinate system
            # specified by _original_coords (from load_field):
            # - Cartesian: z -> -z (reflection across z=0)
            # - Spherical: theta -> π - theta (reflection across equatorial plane)
            # - Cylindrical: z -> -z (reflection across z=0)
            # 
            # For vector components, the component perpendicular to the
            # reflection plane changes sign.
            # -------------------------------------------------
            if reflect:
                try:
                    # Normalize coords to shape (N, ndim)
                    coords_arr = np.asarray(points_array)
                    ndim = coords_arr.shape[1] if coords_arr.ndim == 2 else 1
                    coords_orig = coords_arr.reshape(-1, ndim)
                    coords_ref = coords_orig.copy()

                    # Determine which coordinate to reflect based on original system
                    original_system = coords_original  # Set earlier in evaluate()
                    
                    if original_system == 'spherical':
                        # For spherical coords (phi, r, theta):
                        # Reflect across equatorial plane: theta -> π - theta
                        if coords_orig.shape[1] == 3:
                            # Index 2 is theta
                            coords_ref[:, 2] = np.pi - coords_ref[:, 2]
                        elif coords_orig.shape[1] == 2:
                            # For 2D slices, check if it's an XZ-like plane
                            # If slice_type is 'phi' (meridional plane), reflect theta
                            if slice_type == 'phi':
                                coords_ref[:, 1] = np.pi - coords_ref[:, 1]  # theta coordinate
                    
                    elif original_system == 'cylindrical':
                        # For cylindrical coords (phi, r, z):
                        # Reflect across z=0: z -> -z
                        if coords_orig.shape[1] == 3:
                            # Index 2 is z
                            coords_ref[:, 2] *= -1
                        elif coords_orig.shape[1] == 2:
                            # For 2D slices, if it's RZ plane (phi slice), flip z
                            if slice_type == 'phi':
                                coords_ref[:, 1] *= -1  # z coordinate
                    
                    else:  # cartesian (default)
                        # For cartesian coords (x, y, z):
                        # Reflect across z=0: z -> -z
                        if coords_orig.shape[1] == 3:
                            coords_ref[:, 2] *= -1
                        elif coords_orig.shape[1] == 2:
                            # Assume (x,z) layout for XZ cuts
                            coords_ref[:, 1] *= -1

                    # Prepare data values (flattened)
                    data_flat = np.asarray(data).ravel()

                    # For vector components, reflect sign for the component
                    # perpendicular to the reflection plane
                    if field_name == 'gasv_mesh' and comp is not None:
                        # The perpendicular component depends on the coordinate system:
                        # - Cartesian: vz (comp=2)
                        # - Spherical: v_theta (comp=2)
                        # - Cylindrical: vz (comp=2)
                        # In all cases, comp=2 is the perpendicular component
                        if comp == 2:
                            data_ref = -data_flat
                        else:
                            data_ref = data_flat.copy()
                    else:
                        # Scalars or already selected components
                        data_ref = data_flat.copy()

                    # Augment coords and data for interpolation
                    points_array = np.vstack([coords_orig, coords_ref])
                    data = np.concatenate([data_flat, data_ref])
                except Exception:
                    # On error, fallback to original coords/data
                    points_array = np.asarray(points_array)
                    data = np.asarray(data)

            # Dispatch backend
            
            if interpolator == "rbf":
                return rbf_interp(points_array, data, xi)
            elif interpolator == "linearnd":
                return linearnd_interp(points_array, data, xi)
            elif interpolator == "idw":
                return idw_interp(points_array, data, xi)
            else:
                return griddata_interp(points_array, data, xi)

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


    def plot(self, t=0, contour_levels=10, component='vz', smoothing_sigma=None, field=None):
        """
        Automatically determines the plane (XY, XZ, or 3D) and plots the field data.

        Parameters
        ----------
        t : int, optional
            Index of the snapshot/time to plot.
        contour_levels : int, optional
            Number of contour levels for 2D plots.
        component : str, optional
            Component to plot for vector fields ('vx', 'vy', 'vz').
        field : str or None, optional
            Which field to plot when multiple fields were loaded (e.g. 'gasdens', 'gasv', 'gasenergy').
            If None and exactly one field is present, that field is plotted. If None and multiple
            candidate fields exist, a ValueError is raised instructing the user to pick one.
        """
        
        if self.df is None:
            raise ValueError("No data loaded. Run load_field() first.")

        if component=='vz':
            comp = 2
        if component=='vy':
            comp = 1
        if component=='vx':
            comp = 0   
        df_names = self.df.columns.tolist()

        # Map short names to dataframe column names
        field_map = {
            'gasdens': 'gasdens_mesh',
            'gasv': 'gasv_mesh',
            'gasenergy': 'gasenergy_mesh'
        }

        # Detect candidate fields present in the DataFrame
        candidates = [c for c in df_names if c in ('gasdens_mesh', 'gasv_mesh', 'gasenergy_mesh')]

        # Resolve user-requested field
        if field is not None:
            # allow short names or full column names
            if field in field_map:
                field_col = field_map[field]
            else:
                field_col = field
            if field_col not in df_names:
                raise ValueError(f"Requested field '{field}' not present. Available: {candidates}")
        else:
            if len(candidates) == 1:
                field_col = candidates[0]
            else:
                raise ValueError(
                    f"Multiple fields present {candidates}. Specify which to plot using field='gasdens' or 'gasv'."
                )

        # Extract the mesh grids and field data after slicing
        var1 = self.df['var1_mesh'][t]
        var2 = self.df['var2_mesh'][t]
        var3 = self.df['var3_mesh'][t]

        # Load the original field (before slicing) if needed elsewhere
        d3 = self.sim._load_field_raw('gasdens', snapshot=int(self.df['snapshot'][t]), field_type='scalar')

        # Prepare field_data according to selected field
        raw_field = self.df[field_col][t]
        is_vector = (field_col == 'gasv_mesh')
        if is_vector:
            # choose component or compute magnitude if needed
            data_arr = np.asarray(raw_field)
            # handle common memory layouts: (3, ... ) or (..., 3)
            if data_arr.ndim >= 1 and data_arr.shape[0] == 3:
                # shape (3, N...) -> select component
                field_data = data_arr[comp]
            elif data_arr.ndim >= 1 and data_arr.shape[-1] == 3:
                field_data = data_arr[..., comp]
            else:
                # fallback: try to interpret as magnitude
                try:
                    field_data = np.sqrt(np.sum(data_arr**2, axis=0))
                except Exception:
                    field_data = data_arr
            # do not apply log to vector components
        else:
            # scalar field: apply safe log10 for plotting
            field_data = np.asarray(raw_field)
            # avoid log of non-positive numbers
            with np.errstate(divide='ignore'):
                field_data = np.log10(np.where(field_data <= 0, np.nan, field_data))

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
            data = field_data.ravel()

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(var1_flat, var2_flat, var3_flat, c=data, cmap='Spectral_r', s=5)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.ax.tick_params(labelsize=12)
            if is_vector:
                cbar.set_label(rf"{component} [units]", size=14)
            else:
                cbar.set_label(rf"$\log_{{10}}(field)$", size=14)
            ax.set_xlabel("X",size=14)
            ax.set_ylabel("Y",size=14)
            ax.set_zlabel("Z",size=14)
            
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
                cbar = fig.colorbar(mesh)
                cbar.ax.tick_params(labelsize=12)
                if is_vector:
                    cbar.set_label(rf"{component} [units]", size=14)
                else:
                    cbar.set_label(rf"$\log_{{10}}(field)$", size=14)
                ax.set_xlabel("X",size=14)
                ax.set_ylabel("Y",size=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                fp.Plot.fargopy_mark(ax)
                plt.show()
            elif plane == 'XZ':
                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(var1, var3, field_data, shading='auto', cmap='Spectral_r')
                cbar = fig.colorbar(mesh)
                cbar.ax.tick_params(labelsize=12)
                if is_vector:
                    cbar.set_label(rf"{component} [units]", size=14)
                else:
                    cbar.set_label(rf"$\log_{{10}}(field)$", size=14)
                ax.set_xlabel("X",size=14)
                ax.set_ylabel("Z",size=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                fp.Plot.fargopy_mark(ax)
                plt.show()
            else:
                fig, ax = plt.subplots(figsize=(8, 6))
                mesh = ax.pcolormesh(var1, var2, field_data, shading='auto', cmap='Spectral_r')
                cbar = fig.colorbar(mesh)
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label(rf"$\log_{{10}}(field)$", size=14)
                ax.set_xlabel("X",size=14)
                ax.set_ylabel("Y",size=14)
                ax.tick_params(axis='both', which='major', labelsize=12)
                
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