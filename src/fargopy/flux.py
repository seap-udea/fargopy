###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import fargopy as fp


class Surface:
    """Analytic surface tessellation and helpers for flux and mass analysis.

    The ``Surface`` class provides tools to define analytic control surfaces
    (sphere, cylinder, plane) and tessellate them into small patches.
    These surfaces are used to calculate physical quantities like mass flux
    (accretion rates) and enclosed mass by integrating simulation fields.

    Attributes
    ----------
    type : str
        Surface type ('sphere', 'cylinder', 'plane').
    radius : float
        Radius or characteristic dimension (e.g., hill radius factor).
    centers : np.ndarray
        (N, 3) array of patch centroids.
    normals : np.ndarray
        (N, 3) array of outward-facing unit normals for each patch.
    areas : np.ndarray
        (N,) array of patch areas.

    Examples
    --------
    Define a spherical surface around a planet:
    
    >>> surface = fp.Flux.Surface(type='sphere', radius=0.5, subdivisions=2)
    
    Calculate mass flux through the surface:
    
    >>> flux = surface.mass_flux(sim, field_density='gasdens', field_velocity='gasv')
    """

    def __init__(self, type="sphere", radius=1.0, height=None, subdivisions=1,
                 center=(0.0, 0.0, 0.0), z_cut=None, x_axis=1, y_axis=0, z_axis=0,
                 width=None, length=None, coords='cartesian'):
        """Initialize a Surface instance.

        Parameters
        ----------
        type : str, optional
            Surface type. One of ``'sphere'``, ``'cylinder'`` or ``'plane'``.
        radius : float, optional
            Radius for spheres or radial extent for planes/cylinders.
        height : float, optional
            Cylinder height; required when ``type=='cylinder'``.
        subdivisions : int, optional
            For ``'sphere'``: number of recursive icosahedron subdivisions.
            For ``'cylinder'`` and ``'plane'``: number of divisions along
            each logical axis used to build patches.
        center : tuple of float, optional
            Cartesian coordinates ``(x, y, z)`` of the surface center.
        z_cut : float, optional
            Optional z-plane clipping value. When provided, portions with
            z < ``z_cut`` are removed for spherical tessellations.
        x_axis, y_axis, z_axis : int, optional
            Axis flags (0 or 1) that select the plane normal for
            ``type=='plane'``. Exactly one flag must be 1.
        width, length : float, optional
            Explicit span of the plane along the two in-plane axes. If
            omitted, each defaults to ``2 * radius``.
        coords : str, optional
            Default coordinate system for surface geometry ('cartesian', 'cylindrical', 
            or 'spherical'). This affects the output of properties like 
            ``centers_in_coords`` and ``normals_in_coords``. The tessellation is 
            always computed in Cartesian coordinates internally, but results can be 
            automatically converted to the specified system. Default is 'cartesian'.
        """
        self.type = type
        self.radius = radius
        self.height = height
        self.subdivisions = subdivisions
        self.center = np.array(center)
        self.z_cut = z_cut
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.z_axis = z_axis
        # Plane dimensions: if not provided, fall back to diameter defined by radius
        self.width = width if width is not None else 2.0 * self.radius
        self.length = length if length is not None else 2.0 * self.radius
        if self.width <= 0 or self.length <= 0:
            raise ValueError("width and length must be positive numbers")
        self.volume = None
        
        # Coordinate system for output
        self.coords = coords.lower()
        if self.coords not in ['cartesian', 'cylindrical', 'spherical']:
            raise ValueError(f"Invalid coords '{coords}'. Must be 'cartesian', 'cylindrical', or 'spherical'.")

        # Attributes for tessellation (internal storage always in cartesian)
        self._centers = None
        self._normals = None
        self.areas = None
        self.triangles = None
        self.num_triangles = 0
        self.triangle_index = 0

        if self.type == "sphere":
            self.num_triangles = 20 * (4 ** self.subdivisions)
            self.triangles = np.zeros((self.num_triangles, 3, 3))
            self._centers = np.zeros((self.num_triangles, 3))
            self.areas = np.zeros(self.num_triangles)
            self._tessellate_sphere()
        elif self.type == "cylinder":
            self._tessellate_cylinder()
        elif self.type == "plane":
            self._tessellate_plane()
        else:
            raise ValueError("Unsupported surface type. Use 'sphere', 'cylinder', or 'plane'.")

    def _tessellate_sphere(self):
        """Construct a spherical triangle tessellation.

        The method builds an icosahedron and recursively subdivides its
        faces to produce approximately uniform triangular patches on the
        sphere of radius ``self.radius``. If ``self.z_cut`` is defined,
        triangles crossing the plane ``z = z_cut`` are clipped so that
        only the portion with ``z >= z_cut`` remains.
        """
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        patterns = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ]
        vertices = np.array([Surface._normalize(np.array(p)) * self.radius for p in patterns])
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]
        # Reset triangle index and arrays
        self.num_triangles = 20 * (4 ** self.subdivisions)
        self.triangle_index = 0
        self.triangles = np.zeros((self.num_triangles, 3, 3))
        self._centers = np.zeros((self.num_triangles, 3))
        self.areas = np.zeros(self.num_triangles)

        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            self._subdivide_triangle(v1, v2, v3, self.subdivisions)

        self._calculate_polygon_centers()

        # If z_cut is provided, clip each triangle against the plane instead of
        # simply filtering by centroid position.
        if self.z_cut is not None:
            def clip_triangle_with_plane(tri, z_plane):
                # tri: (3,3) array
                # returns list of triangles (each (3,3)) above or on the plane
                verts = tri
                z = verts[:, 2]
                above = z >= z_plane
                if np.all(above):
                    return [verts]
                elif np.all(~above):
                    return []
                # Identify configuration
                idx_above = np.where(above)[0]
                idx_below = np.where(~above)[0]
                v = verts
                tris = []
                if len(idx_above) == 2 and len(idx_below) == 1:
                    # Two above, one below: split into two triangles
                    a, b = idx_above
                    c = idx_below[0]
                    va, vb, vc = v[a], v[b], v[c]
                    # Intersect edges ac and bc with plane
                    def interp(v1, v2):
                        dz = v2[2] - v1[2]
                        if dz == 0: return v1
                        t = (z_plane - v1[2]) / dz
                        return v1 + t * (v2 - v1)
                    vab = va
                    vbb = vb
                    vac = interp(va, vc)
                    vbc = interp(vb, vc)
                    # Triangle 1: va, vb, vbc
                    tris.append(np.array([va, vb, vbc]))
                    # Triangle 2: va, vbc, vac
                    tris.append(np.array([va, vbc, vac]))
                elif len(idx_above) == 1 and len(idx_below) == 2:
                    # One above, two below: split into one triangle
                    a = idx_above[0]
                    b, c = idx_below
                    va, vb, vc = v[a], v[b], v[c]
                    # Intersect edges ab and ac with plane
                    def interp(v1, v2):
                        dz = v2[2] - v1[2]
                        if dz == 0: return v1
                        t = (z_plane - v1[2]) / dz
                        return v1 + t * (v2 - v1)
                    vab = interp(va, vb)
                    vac = interp(va, vc)
                    # Triangle: va, vab, vac
                    tris.append(np.array([va, vab, vac]))
                return tris

            new_triangles = []
            for tri in self.triangles:
                clipped = clip_triangle_with_plane(tri, self.z_cut)
                new_triangles.extend(clipped)
            self.triangles = np.array(new_triangles)
            self.num_triangles = len(self.triangles)
            self._centers = np.mean(self.triangles, axis=1)
            self.areas = np.array([self._calculate_triangle_area(*tri) for tri in self.triangles])
            self._calculate_normals()
        else:
            self._calculate_all_triangle_areas()
            self._calculate_normals()

        self.volume = self.areas * (self.radius / 3)
        

    def _tessellate_cylinder(self):
        """Discretize a right circular cylinder into top, bottom and lateral patches.

        The cylinder is split into a regular grid on the top and bottom
        circular faces and into strips along the azimuth and height for
        the lateral surface. The method sets per-patch centers, normals
        and areas on attributes named ``top_centers``, ``bottom_centers``,
        ``lateral_centers`` and their corresponding ``_normals`` and
        ``_areas`` attributes.
        """
        theta = np.linspace(0, 2 * np.pi, self.subdivisions, endpoint=False)
        r = np.linspace(0, self.radius, self.subdivisions)
        R, Theta = np.meshgrid(r, theta, indexing='ij')
        X = R * np.cos(Theta) + self.center[0]
        Y = R * np.sin(Theta) + self.center[1]
        Z_top = np.full_like(X, self.center[2] + self.height / 2)
        Z_bottom = np.full_like(X, self.center[2] - self.height / 2)

        self.top_centers = np.stack([X.ravel(), Y.ravel(), Z_top.ravel()], axis=1)
        self.bottom_centers = np.stack([X.ravel(), Y.ravel(), Z_bottom.ravel()], axis=1)
        self.top_normals = np.tile([0, 0, 1], (self.top_centers.shape[0], 1))
        self.bottom_normals = np.tile([0, 0, -1], (self.bottom_centers.shape[0], 1))
        self.top_areas = (self.radius / self.subdivisions) ** 2 * np.pi
        self.bottom_areas = self.top_areas

        theta = np.linspace(0, 2 * np.pi, self.subdivisions, endpoint=False)
        z = np.linspace(-self.height / 2, self.height / 2, self.subdivisions) + self.center[2]  # Adjust z by center
        Theta, Z = np.meshgrid(theta, z, indexing='ij')
        X = self.radius * np.cos(Theta) + self.center[0]
        Y = self.radius * np.sin(Theta) + self.center[1]

        self.lateral_centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        self.lateral_normals = np.stack([np.cos(Theta).ravel(), np.sin(Theta).ravel(), np.zeros_like(Z).ravel()], axis=1)
        self.lateral_areas = (2 * np.pi * self.radius / self.subdivisions) * (self.height / self.subdivisions)

    def _tessellate_plane(self):
        """Discretize an axis-aligned plane into rectangular patches.

        The plane is centered at ``self.center`` and aligned with the axis
        selected by the triple ``(x_axis, y_axis, z_axis)``. The in-plane
        spans are ``self.width`` and ``self.length`` which are split into
        ``self.subdivisions`` cells along each axis. The method sets
        ``self.centers``, ``self.normals``, ``self.areas`` and related
        attributes for downstream use.
        """
        axis_flags = np.array([self.x_axis, self.y_axis, self.z_axis], dtype=int)
        if np.any((axis_flags != 0) & (axis_flags != 1)) or axis_flags.sum() != 1:
            raise ValueError("Plane normal must align with exactly one axis using 0/1 flags.")
        if self.subdivisions <= 0:
            raise ValueError("subdivisions must be >= 1 for plane tessellation.")
        normal_axis = int(np.argmax(axis_flags))
        plane_axes = [idx for idx in range(3) if idx != normal_axis]
        #        lin = np.linspace(-self.radius, self.radius, self.subdivisions + 1)
        #        centers_axis = 0.5 * (lin[:-1] + lin[1:])
        #        grid_a, grid_b = np.meshgrid(centers_axis, centers_axis, indexing='ij')
        #        num_cells = self.subdivisions ** 2
        #        centers = np.zeros((num_cells, 3))
        #        centers[:, plane_axes[0]] = grid_a.ravel() + self.center[plane_axes[0]]
        #        centers[:, plane_axes[1]] = grid_b.ravel() + self.center[plane_axes[1]]
        #        centers[:, normal_axis] = self.center[normal_axis]
        #        self.centers = centers
        #        normal_vector = np.zeros(3)
        #        normal_vector[normal_axis] = 1.0
        #        self.normals = np.tile(normal_vector, (num_cells, 1))
        #        cell_edge = (2 * self.radius) / self.subdivisions
        #        self.areas = np.full(num_cells, cell_edge ** 2)
        #        self.num_triangles = num_cells
        #        self.triangles = None
        # Use width and length (span) to build grid along the two in-plane axes
        lin_a = np.linspace(-self.width/2.0, self.width/2.0, self.subdivisions + 1)
        lin_b = np.linspace(-self.length/2.0, self.length/2.0, self.subdivisions + 1)
        centers_a = 0.5 * (lin_a[:-1] + lin_a[1:])
        centers_b = 0.5 * (lin_b[:-1] + lin_b[1:])
        grid_a, grid_b = np.meshgrid(centers_a, centers_b, indexing='ij')
        num_cells = self.subdivisions ** 2
        centers = np.zeros((num_cells, 3))
        centers[:, plane_axes[0]] = grid_a.ravel() + self.center[plane_axes[0]]
        centers[:, plane_axes[1]] = grid_b.ravel() + self.center[plane_axes[1]]
        centers[:, normal_axis] = self.center[normal_axis]
        self._centers = centers
        normal_vector = np.zeros(3)
        normal_vector[normal_axis] = 1.0
        self._normals = np.tile(normal_vector, (num_cells, 1))
        cell_edge_a = (self.width) / self.subdivisions
        cell_edge_b = (self.length) / self.subdivisions
        self.areas = np.full(num_cells, cell_edge_a * cell_edge_b)
        self.num_triangles = num_cells
        self.triangles = None

    @staticmethod
    def _normalize(v):
        """Return a unit-length copy of the input vector.

        Parameters
        ----------
        v : array_like
            Input vector.

        Returns
        -------
        ndarray
            Unit-normalized copy of ``v``.
        """
        return v / np.linalg.norm(v)

    def _subdivide_triangle(self, v1, v2, v3, depth):
        """Recursively subdivide a triangle and store leaf triangles.

        Parameters
        ----------
        v1, v2, v3 : array_like
            Triangle vertices in Cartesian coordinates (on the unit sphere
            before scaling by ``self.radius``).
        depth : int
            Number of remaining subdivision levels. When ``depth==0`` the
            triangle is written into ``self.triangles``.
        """
        if depth == 0:
            self.triangles[self.triangle_index] = [v1 + self.center, v2 + self.center, v3 + self.center]
            self.triangle_index += 1
            return
        v12 = Surface._normalize((v1 + v2) / 2) * self.radius
        v23 = Surface._normalize((v2 + v3) / 2) * self.radius
        v31 = Surface._normalize((v3 + v1) / 2) * self.radius
        self._subdivide_triangle(v1, v12, v31, depth - 1)
        self._subdivide_triangle(v12, v2, v23, depth - 1)
        self._subdivide_triangle(v31, v23, v3, depth - 1)
        self._subdivide_triangle(v12, v23, v31, depth - 1)

    def _calculate_polygon_centers(self):
        """Compute and cache centroids for all stored triangles.

        The computed centroids are assigned to ``self._centers``.
        """
        self._centers = np.mean(self.triangles, axis=1)

    @staticmethod
    def _calculate_triangle_area(v1, v2, v3):
        """Compute the area of a triangle using the cross product.

        Parameters
        ----------
        v1, v2, v3 : array_like
            Triangle vertex coordinates.

        Returns
        -------
        float
            Triangle area.
        """
        side1 = v2 - v1
        side2 = v3 - v1
        cross_product = np.cross(side1, side2)
        area = np.linalg.norm(cross_product) / 2
        return area

    def _calculate_all_triangle_areas(self):
        """Evaluate and cache areas for every triangle in ``self.triangles``.

        This fills ``self.areas`` using :meth:`_calculate_triangle_area`.
        """
        for i, (v1, v2, v3) in enumerate(self.triangles):
            self.areas[i] = self._calculate_triangle_area(v1, v2, v3)

    def _calculate_normals(self):
        """Compute outward-facing unit normals for each stored triangle.

        The method enforces that each normal points away from ``self.center``;
        if the computed normal points inward it is flipped.
        Results are stored in ``self._normals``.
        """
        self._normals = np.zeros((self.num_triangles, 3))
        for i, tri in enumerate(self.triangles):
            AB = tri[1] - tri[0]
            AC = tri[2] - tri[0]
            normal = np.cross(AB, AC)
            normal /= np.linalg.norm(normal)
            centroid = np.mean(tri, axis=0)
            to_centroid = centroid - self.center
            if np.dot(normal, to_centroid) < 0:
                normal = -normal
            self._normals[i] = normal

    def tessellate(self):
        """Recompute the tessellation from current instance parameters.

        This method is a convenience wrapper that dispatches to the
        appropriate internal tessellation implementation depending on
        ``self.type``.
        """
        if self.type == "sphere":
            self._tessellate_sphere()
        elif self.type == "cylinder":
            self._tessellate_cylinder()
        elif self.type == "plane":
            self._tessellate_plane()
        else:
            raise ValueError("Unsupported surface type. Use 'sphere', 'cylinder', or 'plane'.")

    def get_centers_in_coords(self, coords_system='cartesian'):
        """
        Get surface centers in the specified coordinate system.
        
        This method converts the tessellation centers from Cartesian coordinates
        to the requested coordinate system. This is essential for fast interpolation
        when the simulation data is stored in spherical or cylindrical coordinates,
        as it allows using RegularGridInterpolator instead of slow unstructured methods.
        
        Parameters
        ----------
        coords_system : str, optional
            Target coordinate system: 'cartesian', 'spherical', or 'cylindrical'.
            Default is 'cartesian'.
            
        Returns
        -------
        tuple of np.ndarray
            For cartesian: (x, y, z)
            For spherical: (phi, r, theta)
            For cylindrical: (phi, r, z)
            Each array has shape (n_points,) corresponding to surface centers.
            
        Notes
        -----
        The coordinate transformations are:
        - Spherical: phi = arctan2(y, x), r = sqrt(x² + y² + z²), theta = arccos(z/r)
        - Cylindrical: phi = arctan2(y, x), r = sqrt(x² + y²), z = z
        
        Examples
        --------
        >>> surface = Surface(type='sphere', radius=1.0, subdivisions=2)
        >>> phi, r, theta = surface.get_centers_in_coords('spherical')
        >>> # Use these for fast interpolation with spherical simulation data
        """
        if self.type == "sphere":
            centers = self._centers
        elif self.type == "cylinder":
            if not hasattr(self, 'top_centers'):
                raise ValueError("Cylinder not tessellated. Call tessellate() first.")
            centers = np.concatenate([self.top_centers, self.bottom_centers, self.lateral_centers], axis=0)
        elif self.type == "plane":
            if self._centers is None:
                raise ValueError("Plane not tessellated. Call tessellate() first.")
            centers = self._centers
        else:
            raise ValueError(f"Unsupported surface type: {self.type}")
        
        x = centers[:, 0]
        y = centers[:, 1]
        z = centers[:, 2]
        
        if coords_system == 'cartesian':
            return x, y, z
        elif coords_system == 'spherical':
            r = np.sqrt(x**2 + y**2 + z**2)
            # Avoid division by zero for points at origin
            r_safe = np.where(r > 1e-15, r, 1e-15)
            theta = np.arccos(np.clip(z / r_safe, -1, 1))
            phi = np.arctan2(y, x)
            return phi, r, theta
        elif coords_system == 'cylindrical':
            r_cyl = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return phi, r_cyl, z
        else:
            raise ValueError(f"Unknown coordinate system: {coords_system}")
    
    def get_normals_in_coords(self, coords_system='cartesian'):
        """
        Get surface normals in the specified coordinate system.
        
        Transforms the unit normal vectors from Cartesian to the requested
        coordinate system. This is useful for flux calculations when working
        in native simulation coordinates.
        
        Parameters
        ----------
        coords_system : str, optional
            Target coordinate system: 'cartesian', 'spherical', or 'cylindrical'.
            
        Returns
        -------
        tuple of np.ndarray
            For cartesian: (nx, ny, nz)
            For spherical: (n_phi, n_r, n_theta)
            For cylindrical: (n_phi, n_r, n_z)
            Each array has shape (n_points,).
            
        Notes
        -----
        The transformation uses Jacobian matrices for the coordinate change.
        For spherical: [n_phi, n_r, n_theta]^T = J^T [nx, ny, nz]^T
        For cylindrical: [n_phi, n_r, n_z]^T = J^T [nx, ny, nz]^T
        """
        if self.type == "sphere":
            normals = self._normals
            centers = self._centers
        elif self.type == "cylinder":
            if not hasattr(self, 'top_normals'):
                raise ValueError("Cylinder not tessellated. Call tessellate() first.")
            normals = np.concatenate([self.top_normals, self.bottom_normals, self.lateral_normals], axis=0)
            centers = np.concatenate([self.top_centers, self.bottom_centers, self.lateral_centers], axis=0)
        elif self.type == "plane":
            if self._normals is None:
                raise ValueError("Plane not tessellated. Call tessellate() first.")
            normals = self._normals
            centers = self._centers
        else:
            raise ValueError(f"Unsupported surface type: {self.type}")
        
        nx = normals[:, 0]
        ny = normals[:, 1]
        nz = normals[:, 2]
        
        if coords_system == 'cartesian':
            return nx, ny, nz
        elif coords_system == 'spherical':
            # Get position coordinates for Jacobian
            x = centers[:, 0]
            y = centers[:, 1]
            z = centers[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            r_safe = np.where(r > 1e-15, r, 1e-15)
            r_xy = np.sqrt(x**2 + y**2)
            r_xy_safe = np.where(r_xy > 1e-15, r_xy, 1e-15)
            
            # Transform: n_i = (∂x^j/∂q^i) n_j where q = (phi, r, theta)
            # For spherical: x = r sin(theta) cos(phi), y = r sin(theta) sin(phi), z = r cos(theta)
            sin_theta = r_xy / r_safe
            cos_theta = z / r_safe
            sin_phi = y / r_xy_safe
            cos_phi = x / r_xy_safe
            
            # Jacobian transpose (transforms contravariant vectors)
            n_phi = -r_safe * sin_theta * sin_phi * nx + r_safe * sin_theta * cos_phi * ny
            n_r = sin_theta * cos_phi * nx + sin_theta * sin_phi * ny + cos_theta * nz
            n_theta = r_safe * cos_theta * cos_phi * nx + r_safe * cos_theta * sin_phi * ny - r_safe * sin_theta * nz
            
            return n_phi, n_r, n_theta
        elif coords_system == 'cylindrical':
            # Get position coordinates
            x = centers[:, 0]
            y = centers[:, 1]
            r_cyl = np.sqrt(x**2 + y**2)
            r_cyl_safe = np.where(r_cyl > 1e-15, r_cyl, 1e-15)
            sin_phi = y / r_cyl_safe
            cos_phi = x / r_cyl_safe
            
            # Transform for cylindrical
            n_phi = -r_cyl_safe * sin_phi * nx + r_cyl_safe * cos_phi * ny
            n_r = cos_phi * nx + sin_phi * ny
            n_z = nz
            
            return n_phi, n_r, n_z
        else:
            raise ValueError(f"Unknown coordinate system: {coords_system}")

    @property
    def centers(self):
        """
        Get surface centers in the coordinate system specified by self.coords.
        
        This property automatically returns coordinates in the system specified
        during Surface initialization. For Cartesian, returns shape (N, 3) array.
        For cylindrical/spherical, returns shape (N, 3) array with columns
        representing (phi, r, z/theta).
        
        Returns
        -------
        np.ndarray
            Array of shape (N, 3) where N is the number of surface patches.
            - If coords='cartesian': columns are (x, y, z)
            - If coords='cylindrical': columns are (phi, r, z)
            - If coords='spherical': columns are (phi, r, theta)
            
        Examples
        --------
        >>> sphere = fp.flux.Surface(type='sphere', radius=1.0, subdivisions=2, coords='spherical')
        >>> centers = sphere.centers  # Returns (N, 3) array with (phi, r, theta) columns
        >>> phi = centers[:, 0]
        >>> r = centers[:, 1]
        >>> theta = centers[:, 2]
        
        Notes
        -----
        Internally, coordinates are always stored in Cartesian form. This property
        performs the coordinate transformation on-the-fly based on self.coords.
        """
        if self.coords == 'cartesian':
            # Return internal storage directly for cartesian
            if self.type == "sphere":
                return self._centers
            elif self.type == "cylinder":
                return np.concatenate([self.top_centers, self.bottom_centers, self.lateral_centers], axis=0)
            elif self.type == "plane":
                return self._centers
        else:
            # Transform to requested coordinate system
            return np.column_stack(self.get_centers_in_coords(self.coords))
    
    @property
    def normals(self):
        """
        Get surface normals in the coordinate system specified by self.coords.
        
        This property automatically returns normal vectors in the system specified
        during Surface initialization. For Cartesian, returns shape (N, 3) array.
        For cylindrical/spherical, returns shape (N, 3) array with columns
        representing (n_phi, n_r, n_z/n_theta).
        
        Returns
        -------
        np.ndarray
            Array of shape (N, 3) where N is the number of surface patches.
            - If coords='cartesian': columns are (nx, ny, nz)
            - If coords='cylindrical': columns are (n_phi, n_r, n_z)
            - If coords='spherical': columns are (n_phi, n_r, n_theta)
            
        Examples
        --------
        >>> sphere = fp.flux.Surface(type='sphere', radius=1.0, coords='spherical')
        >>> normals = sphere.normals  # Returns (N, 3) array with (n_phi, n_r, n_theta) columns
        >>> n_r = normals[:, 1]  # Radial component (should be ~1 for sphere)
        
        Notes
        -----
        Internally, normals are always stored in Cartesian form. This property
        performs the coordinate transformation on-the-fly based on self.coords.
        Normals are always unit vectors regardless of coordinate system.
        """
        if self.coords == 'cartesian':
            # Return internal storage directly for cartesian
            if self.type == "sphere":
                return self._normals
            elif self.type == "cylinder":
                return np.concatenate([self.top_normals, self.bottom_normals, self.lateral_normals], axis=0)
            elif self.type == "plane":
                return self._normals
        else:
            # Transform to requested coordinate system
            return np.column_stack(self.get_normals_in_coords(self.coords))

    @property
    def centers_in_coords(self):
        """
        Get surface centers in the default coordinate system specified by self.coords.
        
        This property provides a convenient way to access surface centers without
        manually calling get_centers_in_coords(). The coordinate system is determined
        by the ``coords`` attribute set during initialization or modified later.
        
        Returns
        -------
        tuple of np.ndarray
            Coordinates in the system specified by self.coords:
            - Cartesian: (x, y, z)
            - Spherical: (phi, r, theta) 
            - Cylindrical: (phi, r, z)
            
        Examples
        --------
        >>> surface = fp.Flux.Surface(type='sphere', radius=1.0, subdivisions=2, coords='spherical')
        >>> phi, r, theta = surface.centers_in_coords
        >>> # Use directly for interpolation
        >>> rho = fields.evaluate(var1=phi, var2=r, var3=theta, field='gasdens')
        
        Notes
        -----
        This is equivalent to calling get_centers_in_coords(self.coords) but more concise.
        Change self.coords to switch the output coordinate system dynamically.
        """
        return self.get_centers_in_coords(self.coords)
    
    @property
    def normals_in_coords(self):
        """
        Get surface normals in the default coordinate system specified by self.coords.
        
        This property provides a convenient way to access surface normals without
        manually calling get_normals_in_coords(). The coordinate system is determined
        by the ``coords`` attribute.
        
        Returns
        -------
        tuple of np.ndarray
            Normal vector components in the system specified by self.coords:
            - Cartesian: (nx, ny, nz)
            - Spherical: (n_phi, n_r, n_theta)
            - Cylindrical: (n_phi, n_r, n_z)
            
        Examples
        --------
        >>> surface = fp.Flux.Surface(type='sphere', radius=1.0, coords='cylindrical')
        >>> n_phi, n_r, n_z = surface.normals_in_coords
        
        Notes
        -----
        This is equivalent to calling get_normals_in_coords(self.coords) but more concise.
        Normals are always unit vectors regardless of coordinate system.
        """
        return self.get_normals_in_coords(self.coords)

    def generate_dataframe(self):
        """Return tessellation metadata as a pandas :class:`DataFrame`.

        The returned DataFrame contains one row per surface patch with
        columns ``'Center'``, ``'Normal'`` and ``'Area'``. Coordinates are
        represented as Python lists to ensure JSON-serializable cell values.
        """
        data = []
        for i, (center, normal, area) in enumerate(zip(self.centers, self.normals, self.areas)):
            data.append({
                "Center": center.tolist(),
                "Normal": normal.tolist(),
                "Area": area
            })
        return pd.DataFrame(data)

    def total_mass_mtc(self, sim, field='gasdens', n_samples=10000, snapshot=[0,1],
                   interpolator='regular_grid', method='linear', cut_r=None,
                   follow_planet=True, planet_index=0, random_seed=None, coords=None):
        """Estimate enclosed mass using Monte Carlo sampling.

        The method samples ``n_samples`` points uniformly within the
        spherical region defined by this surface (accounting for ``z_cut``
        when present), interpolates the requested density field at the
        sample points and returns an estimate of the enclosed mass.

        Parameters
        ----------
        sim : object
            Simulation object providing ``load_field`` and ``load_planets``
            methods (FARGOpy simulation API).
        field : str, optional
            Density field name to sample (default ``'gasdens'``).
        n_samples : int, optional
            Number of Monte Carlo samples per snapshot.
        snapshot : int or [start, end], optional
            Snapshot index or inclusive snapshot range.
        interpolator : str, optional
            Interpolator backend passed to the field evaluator.
        method : str, optional
            Interpolation method passed to the field evaluator.
        cut_r : float, optional
            Radial cut passed to ``sim.load_field`` to limit data transfer.
        follow_planet : bool, optional
            If True follow the planet position when sampling (useful for
            Hill-sphere based regions).
        planet_index : int, optional
            Index of the planet to follow when ``follow_planet`` is True.
        random_seed : int, optional
            Random seed for reproducibility of Monte Carlo sampling.
            If None (default), results will vary between runs.
            Set to a fixed integer for reproducible results.
        coords : str, optional
            Coordinate system to use for field loading and interpolation.
            If None, uses ``self.coords``. Options are 'cartesian', 'cylindrical', 
            or 'spherical'. Using the simulation's native coordinate system 
            enables faster RegularGridInterpolator.

        Returns
        -------
        float or numpy.ndarray
            Estimated enclosed mass. Returns a single float if a single
            snapshot is requested, otherwise an array of estimates.
        """
        # Determine coordinate system to use
        if coords is None:
            coords = self.coords
        coords = coords.lower()
        
        if isinstance(snapshot, int):
            times = [snapshot]
        else:
            times = np.linspace(snapshot[0], snapshot[1], snapshot[1]-snapshot[0]+1)
        mass = np.zeros(len(times))
        
        planet = sim.load_planets(snapshot=0)[planet_index]
        factor = self.radius/planet.hill_radius
        
        # Warn if n_samples is too low for accurate Monte Carlo estimates
        if n_samples < 5000:
            import warnings
            warnings.warn(
                f"Monte Carlo sampling with n_samples={n_samples} may produce noisy results. "
                f"Consider increasing n_samples to 10000 or more for smoother estimates.",
                UserWarning
            )
        
        # Set random seed for reproducibility if requested
        if random_seed is not None:
            np.random.seed(random_seed)

        for i, t in enumerate(tqdm(times, desc="Calculating total mass (Monte Carlo)")):
            # Update surface if following planet
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])
                if hasattr(planet, 'hill_radius'):
                    self.radius = factor * planet.hill_radius
                self.tessellate()

            # Generate random points inside the sphere (in spherical coordinates)
            u = np.random.uniform(0, 1, int(n_samples))
            v = np.random.uniform(0, 1, int(n_samples))
            w = np.random.uniform(0, 1, int(n_samples))

            r_sph = self.radius * np.cbrt(u)
            theta_sph = np.arccos(1 - 2 * v)
            phi_sph = 2 * np.pi * w

            # Convert to Cartesian for geometry checks
            x = r_sph * np.sin(theta_sph) * np.cos(phi_sph) + self.center[0]
            y = r_sph * np.sin(theta_sph) * np.sin(phi_sph) + self.center[1]
            z = r_sph * np.cos(theta_sph) + self.center[2]

            # Apply z_cut if specified
            if self.z_cut is not None:
                mask = z > self.z_cut
                x, y, z = x[mask], y[mask], z[mask]
                r_sph, theta_sph, phi_sph = r_sph[mask], theta_sph[mask], phi_sph[mask]
                n_effective = len(x)
                h = self.radius + self.center[2] - self.z_cut
                h = np.clip(h, 0, 2*self.radius)
                volume = (1/3) * np.pi * h**2 * (3*self.radius - h)
            else:
                n_effective = int(n_samples)
                volume = (4/3) * np.pi * self.radius**3

            # Load field in the specified coordinate system for fast interpolation
            if cut_r is not None:
                field_interp = sim.load_field(
                    fields=[field],
                    snapshot=[int(t)],
                    cut=(self.center[0], self.center[1], self.center[2], cut_r),
                    coords=coords
                )
            else:
                field_interp = sim.load_field(
                    fields=[field],
                    snapshot=[int(t)],
                    cut=(self.center[0], self.center[1], self.center[2], self.radius*2),
                    coords=coords
                )

            # Convert sample points to the specified coordinate system
            if coords == 'spherical':
                var1, var2, var3 = fp.transform_coords('cartesian', 'spherical', x, y, z)
            elif coords == 'cylindrical':
                var1, var2, var3 = fp.transform_coords('cartesian', 'cylindrical', x, y, z)
            else:
                # Cartesian
                var1, var2, var3 = x, y, z

            # Interpolate in native coordinates (uses fast RegularGridInterpolator)
            rho = field_interp.evaluate(
                time=t,
                var1=var1,
                var2=var2,
                var3=var3,
                interpolator=interpolator,
                method=method
            )
            
            # Compute mass estimate
            # Filter NaN values (points outside domain)
            valid_rho = rho[~np.isnan(rho)]
            
            if len(valid_rho) == 0:
                # All points are outside domain - mass is zero
                mass[i] = 0.0
            else:
                # Monte Carlo estimator: V * mean(rho)
                # If some points are NaN, we need to account for the fraction that's valid
                fraction_valid = len(valid_rho) / n_effective
                avg_rho = np.mean(valid_rho)
                mass[i] = avg_rho * volume * fraction_valid

        if len(mass) == 1:
            return mass[0]
        return mass


    def mass_flux(self, sim, field_density='gasdens', field_velocity='gasv',
                snapshot=[0, 1], interpolator='regular_grid', method='linear',
                follow_planet=True, planet_index=0,
                correct_normals=True, relative_velocity=False, coords=None):
        """Compute mass flux through the surface patches.

        The instantaneous mass flux for each patch is computed as::

            dΦ = ρ * (v_rel · n_out) * dA

        and the returned value is the sum over all patches. Velocities can
        optionally be converted to the planet rest frame by enabling
        ``relative_velocity``.

        Parameters
        ----------
        sim : object
            Simulation object exposing ``load_field`` and ``load_planets``.
        field_density : str, optional
            Density field name (default ``'gasdens'``).
        field_velocity : str, optional
            Velocity field name (default ``'gasv'``).
        snapshot : [start, end], optional
            Inclusive snapshot range to evaluate.
        interpolator, method : str, optional
            Passed to the field evaluator for interpolation.
        follow_planet : bool, optional
            If True follow the planet position and scale the surface
            (useful for Hill-sphere analysis).
        planet_index : int, optional
            Index of the planet to follow.
        correct_normals : bool, optional
            If True ensure per-patch normals point outward from the
            surface center before computing the flux.
        relative_velocity : bool, optional
            If True subtract the planet velocity from the interpolated
            fluid velocity prior to flux computation.
        coords : str, optional
            Coordinate system to use for field loading and interpolation.
            If None, uses ``self.coords`` (default coordinate system set during 
            initialization). Options are 'cartesian', 'cylindrical', or 'spherical'.
            Using the simulation's native coordinate system enables faster 
            RegularGridInterpolator instead of unstructured griddata.

        Returns
        -------
        numpy.ndarray
            Array of flux values, one per requested snapshot.

        Examples
        --------
        Compute accretion rate (mass flux) onto a planet:
        
        >>> surface = fp.Flux.Surface(type='sphere', radius=0.5, subdivisions=3, coords='spherical')
        >>> mdot = surface.mass_flux(sim, field_density='gasdens', field_velocity='gasv', follow_planet=True)
        >>> plt.plot(mdot)
        """
        
        # Check if simulation is 3D (required for flux calculations)
        if hasattr(sim, 'vars') and sim.vars.DIM == 2:
            raise ValueError(
                "mass_flux() requires a 3D simulation. "
                "Your simulation is 2D. Flux calculations through 3D surfaces cannot be performed in 2D domains."
            )

        # Determine coordinate system to use
        if coords is None:
            coords = self.coords
        coords = coords.lower()
        
        # Validate coordinate system
        if coords not in ['cartesian', 'cylindrical', 'spherical']:
            raise ValueError(f"Invalid coords '{coords}'. Must be 'cartesian', 'cylindrical', or 'spherical'.")

        steps = snapshot[1] - snapshot[0] + 1
        times = np.linspace(snapshot[0], snapshot[1], steps)
        flux = np.zeros(len(times))

        # Initial planet for scaling
        planet0 = sim.load_planets()[planet_index]
        factor = self.radius / planet0.hill_radius

        for i, t in enumerate(tqdm(times, desc="Calculating mass flux")):

            # -------------------------------------------------------------
            #Update the surface position and scale if following the planet
            # -------------------------------------------------------------
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])

                if hasattr(planet, 'hill_radius'):
                    self.radius = factor * planet.hill_radius

                self.tessellate()
            elif relative_velocity:
                # Need planet velocity but not position
                planet = sim.load_planets(snapshot=int(t))[planet_index]

            # Planet velocity (for relative velocities)
            if relative_velocity:
                vpx, vpy, vpz = planet.vel.x, planet.vel.y, planet.vel.z
            else:
                vpx, vpy, vpz = 0.0, 0.0, 0.0

            # -------------------------------------------------------------
            # Select geometric properties of the surface
            # Always use internal Cartesian coordinates for geometry
            # -------------------------------------------------------------
            if self.type == "sphere":
                centers = self._centers  # Always use Cartesian for geometric calculations
                normals = self._normals  # Always use Cartesian for dot products
                areas = self.areas
                surface_cut = (self.center[0], self.center[1], self.center[2], 2*self.radius)

            elif self.type == "cylinder":
                centers = np.concatenate([self.top_centers,
                                        self.bottom_centers,
                                        self.lateral_centers], axis=0)
                normals = np.concatenate([self.top_normals,
                                        self.bottom_normals,
                                        self.lateral_normals], axis=0)
                areas = np.concatenate([
                    np.full(self.top_centers.shape[0], self.top_areas),
                    np.full(self.bottom_centers.shape[0], self.bottom_areas),
                    np.full(self.lateral_centers.shape[0], self.lateral_areas)
                ])
                surface_cut = (self.center[0], self.center[1], self.center[2],
                            2*self.radius, 2*self.height)

            elif self.type == "plane":
                centers = self._centers  # Always use Cartesian for geometric calculations
                normals = self._normals  # Always use Cartesian for dot products
                areas = self.areas
                surface_cut = (self.center[0], self.center[1], self.center[2], 2*self.radius)

            else:
                raise ValueError("Unsupported surface type.")

            # -------------------------------------------------------------
            # Load both fields simultaneously into a single DataFrame
            # Use specified coordinate system for optimal interpolation
            # -------------------------------------------------------------
            fields = sim.load_field(
                fields=[field_density, field_velocity],
                snapshot=[int(t)],
                cut=surface_cut,
                coords=coords  # Load in requested coords for fast RegularGridInterpolator
            )
            
            # Convert surface centers to the specified coordinate system
            var1_eval, var2_eval, var3_eval = self.get_centers_in_coords(coords)

            # -------------------------------------------------------------
            # Interpolate density
            # -------------------------------------------------------------
            rho = fields.evaluate(
                time=t,
                var1=var1_eval,
                var2=var2_eval,
                var3=var3_eval,
                interpolator=interpolator,
                method=method,
                field="gasdens"
            )

            # -------------------------------------------------------------
            # Interpolate velocity vector  
            # -------------------------------------------------------------
            vel = fields.evaluate(
                time=t,
                var1=var1_eval,
                var2=var2_eval,
                var3=var3_eval,
                interpolator=interpolator,
                method=method,
                field="gasv"
            )

            # Shape fix (3,N → N,3)
            if vel.ndim == 2 and vel.shape[0] == 3:
                vel = vel.T
            
            # -------------------------------------------------------------
            # Convert velocity to Cartesian for dot product with normals
            # Normals are always in Cartesian system
            # -------------------------------------------------------------
            if coords != 'cartesian':
                # Use transform_velocity to convert from coords to cartesian
                # vel is in the format expected by the coordinate system:
                # - spherical: [v_phi, v_r, v_theta]
                # - cylindrical: [v_phi, v_r, v_z]
                
                # Get position in the original coordinate system for transformation
                pos_coords = self.get_centers_in_coords(coords)
                
                # Transform velocity components to Cartesian
                if vel.shape[0] == len(centers):
                    # vel has shape (N, 3)
                    vel_tuple = (vel[:, 0], vel[:, 1], vel[:, 2])
                    pos_tuple = pos_coords
                    
                    vx, vy, vz = fp.transform_velocity(coords, 'cartesian', vel_tuple, pos_tuple)
                    vel = np.column_stack([vx, vy, vz])
                else:
                    raise ValueError(f"Unexpected velocity shape: {vel.shape}")


            # -------------------------------------------------------------
            # Ensure normals point outward
            # -------------------------------------------------------------
            if correct_normals:
                to_centers = centers - self.center
                flip = (np.einsum('ij,ij->i', normals, to_centers) < 0)
                normals[flip] *= -1

            # -------------------------------------------------------------
            # Convert to velocity in planet's rest frame
            # -------------------------------------------------------------
            if relative_velocity:
                vel[:, 0] -= vpx
                vel[:, 1] -= vpy
                vel[:, 2] -= vpz

            # -------------------------------------------------------------
            # Compute flux for each surface element
            # -------------------------------------------------------------
            v_dot_n = np.einsum('ij,ij->i', vel, normals)
            dF = rho * v_dot_n * areas
            flux[i] = np.nansum(dF)

        return flux



    def total_mass(self,
                sim,
                field='gasdens',
                snapshot=[0,1],
                follow_planet=True,
                planet_index=0,
                return_resolution=False):
        """Compute enclosed mass by direct grid integration.

        The method integrates the requested density field on the simulation
        grid accounting for the region geometry defined by this Surface instance.
        Supports both spherical and cylindrical coordinate systems.
        ``self.type`` must be either ``'sphere'`` or ``'cylinder'``; the 
        integration mask is constructed accordingly.

        Parameters
        ----------
        sim : object
            Simulation object providing access to the raw grid and fields.
        field : str, optional
            Density field name (default ``'gasdens'``).
        snapshot : int or [start, end], optional
            Snapshot index or inclusive snapshot range to integrate.
        follow_planet : bool, optional
            If True update the integration center and radius from the
            specified planet's position/hill radius.
        planet_index : int, optional
            Planet index to follow when ``follow_planet`` is True.
        return_resolution : bool, optional
            If True return detailed resolution metadata for each snapshot
            alongside the integrated mass.

        Returns
        -------
        float or numpy.ndarray or list
            If ``return_resolution`` is False and a single snapshot is
            requested, returns a float. If multiple snapshots are
            requested, returns a numpy array of masses. If
            ``return_resolution`` is True a list of dictionaries with
            per-snapshot metadata is returned.

        Examples
        --------
        Compute total mass inside a Hill sphere:
        
        >>> surface = fp.Flux.Surface(type='sphere', radius=1.0) # radius is factor of Hill radius
        >>> mass = surface.total_mass(sim, field='gasdens', follow_planet=True)
        """

        # --------------------
        # Handle snapshot list
        # --------------------
        if isinstance(snapshot, int):
            times = [snapshot]
        else:
            s0, s1 = snapshot
            times = np.arange(s0, s1+1)

        masses = []
        resolutions = []

        # ----------------------------
        # Detect coordinate system from simulation
        # ----------------------------
        coords = sim.vars.COORDINATES if hasattr(sim, 'vars') else 'spherical'
        
        # Check if simulation is 3D (required for volume integration)
        if hasattr(sim, 'vars') and sim.vars.DIM == 2:
            raise ValueError(
                "total_mass() requires a 3D simulation. "
                "Your simulation is 2D. Volume integration cannot be performed in 2D domains."
            )

        # ----------------------------
        # Load grid info based on coordinate system
        # ----------------------------
        r_arr = sim.domains.r
        ph_arr = sim.domains.phi
        
        if coords == 'spherical':
            th_arr = sim.domains.theta
            TH, RR, PH = np.meshgrid(th_arr, r_arr, ph_arr, indexing='ij')
            X = RR * np.sin(TH) * np.cos(PH)
            Y = RR * np.sin(TH) * np.sin(PH)
            Z = RR * np.cos(TH)
            
            # Precompute cell volumes (spherical metric)
            dr = np.diff(r_arr)
            dth = np.diff(th_arr)
            dph = np.diff(ph_arr)
            
            dr_full = np.empty_like(r_arr); dr_full[:-1] = dr; dr_full[-1] = dr[-1]
            dth_full = np.empty_like(th_arr); dth_full[:-1] = dth; dth_full[-1] = dth[-1]
            dph_full = np.empty_like(ph_arr); dph_full[:-1] = dph; dph_full[-1] = dph[-1]
            
            DR = dr_full[np.newaxis, :, np.newaxis]
            DTH = dth_full[:, np.newaxis, np.newaxis]
            DPH = dph_full[np.newaxis, np.newaxis, :]
            
            dV = (RR**2) * np.sin(TH) * DR * DTH * DPH
            
        elif coords == 'cylindrical':
            z_arr = sim.domains.z
            ZZ, RR, PH = np.meshgrid(z_arr, r_arr, ph_arr, indexing='ij')
            X = RR * np.cos(PH)
            Y = RR * np.sin(PH)
            Z = ZZ
            
            # Precompute cell volumes (cylindrical metric)
            dr = np.diff(r_arr)
            dz = np.diff(z_arr)
            dph = np.diff(ph_arr)
            
            dr_full = np.empty_like(r_arr); dr_full[:-1] = dr; dr_full[-1] = dr[-1]
            dz_full = np.empty_like(z_arr); dz_full[:-1] = dz; dz_full[-1] = dz[-1]
            dph_full = np.empty_like(ph_arr); dph_full[:-1] = dph; dph_full[-1] = dph[-1]
            
            DR = dr_full[np.newaxis, :, np.newaxis]
            DZ = dz_full[:, np.newaxis, np.newaxis]
            DPH = dph_full[np.newaxis, np.newaxis, :]
            
            dV = RR * DR * DZ * DPH
        
        else:
            raise ValueError(f"Unsupported coordinate system: {coords}")

        # ----------------------------------------
        # Detect geometry
        # ----------------------------------------
        geom = self.type.lower()

        if geom not in ['sphere', 'cylinder']:
            raise ValueError("Surface.type must be 'sphere' or 'cylinder'")

        # Loop over snapshots
        for t in tqdm(times, desc="Calculating total mass"):

            # Follow planet
            if follow_planet:
                planet = sim.load_planets(snapshot=t)[planet_index]
                xp, yp, zp = planet.pos.x, planet.pos.y, planet.pos.z

                # Update center and radius according to Hill radius
                factor = np.round(self.radius / sim.load_planets(snapshot=t)[planet_index].hill_radius,2)
                self.center = (xp, yp, zp)
                self.radius = factor * planet.hill_radius

            else:
                xp, yp, zp = self.center

            Xc = X - xp
            Yc = Y - yp
            Zc = Z - zp

            # ---------------------------------------
            # Apply geometry mask
            # ---------------------------------------
            if geom == 'sphere':
                Rlim = self.radius
                mask = (Xc**2 + Yc**2 + Zc**2) <= Rlim**2

                # If z_cut exists → semisphere
                if hasattr(self, 'z_cut') and (self.z_cut is not None):
                    mask &= (Z >= self.z_cut)

                Hlim = None

            elif geom == 'cylinder':
                Rlim = self.radius
                Hlim = self.height   # provided by Surface(type='cylinder', height=...)

                Rcyl = np.sqrt(Xc**2 + Yc**2)
    
                mask = (Rcyl <= Rlim) & (np.abs(Zc) <= Hlim) & (np.abs(Zc) >= -Hlim)

            # ---------------------------------------
            # Load density for this snapshot
            # ---------------------------------------
            rho = sim.load_field(field, snapshot=t).gasdens_mesh[0]
            
            # Enclosed mass
            M = np.sum(rho[mask] * dV[mask])
            masses.append(M)
            
            # Resolution info
            if return_resolution:
                idx_dim1, idx_r, idx_ph = np.where(mask)
                
                dim1_name = 'N_theta' if coords == 'spherical' else 'N_z'

                resolutions.append({
                    "snapshot": t,
                    "mass": M,
                    "geometry": geom,
                    "R_extent": Rlim,
                    "H_extent": Hlim,
                    dim1_name: len(np.unique(idx_dim1)),
                    "N_r":     len(np.unique(idx_r)),
                    "N_phi":   len(np.unique(idx_ph)),
                    "N_total": mask.sum()
                })

        # Return logic
        if return_resolution:
            return resolutions
        if len(masses) == 1:
            return masses[0]
        return np.array(masses)
