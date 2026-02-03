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
                 width=None, length=None):
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

        # Attributes for tessellation
        self.centers = None
        self.normals = None
        self.areas = None
        self.triangles = None
        self.num_triangles = 0
        self.triangle_index = 0

        if self.type == "sphere":
            self.num_triangles = 20 * (4 ** self.subdivisions)
            self.triangles = np.zeros((self.num_triangles, 3, 3))
            self.centers = np.zeros((self.num_triangles, 3))
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
        self.centers = np.zeros((self.num_triangles, 3))
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
            self.centers = np.mean(self.triangles, axis=1)
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
        self.centers = centers
        normal_vector = np.zeros(3)
        normal_vector[normal_axis] = 1.0
        self.normals = np.tile(normal_vector, (num_cells, 1))
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

        The computed centroids are assigned to ``self.centers``.
        """
        self.centers = np.mean(self.triangles, axis=1)

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
        Results are stored in ``self.normals``.
        """
        self.normals = np.zeros((self.num_triangles, 3))
        for i, tri in enumerate(self.triangles):
            AB = tri[1] - tri[0]
            AC = tri[2] - tri[0]
            normal = np.cross(AB, AC)
            normal /= np.linalg.norm(normal)
            centroid = np.mean(tri, axis=0)
            to_centroid = centroid - self.center
            if np.dot(normal, to_centroid) < 0:
                normal = -normal
            self.normals[i] = normal

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
                   interpolator='griddata', method='linear', cut_r=None,
                   follow_planet=True, planet_index=0):
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

        Returns
        -------
        float or numpy.ndarray
            Estimated enclosed mass. Returns a single float if a single
            snapshot is requested, otherwise an array of estimates.
        """
        if isinstance(snapshot, int):
            times = [snapshot]
        else:
            times = np.linspace(snapshot[0], snapshot[1], snapshot[1]-snapshot[0]+1)
        mass = np.zeros(len(times))
        
        planet = sim.load_planets(snapshot=0)[planet_index]
        factor = self.radius/planet.hill_radius 

        for i, t in enumerate(tqdm(times, desc="Calculating total mass")):
            # Update surface if following planet
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])
                if hasattr(planet, 'hill_radius'):
                    self.radius = factor * planet.hill_radius
                self.tessellate()

            # Generate random points inside the sphere
            u = np.random.uniform(0, 1, int(n_samples))
            v = np.random.uniform(0, 1, int(n_samples))
            w = np.random.uniform(0, 1, int(n_samples))

            r = self.radius * np.cbrt(u)
            theta = np.arccos(1 - 2 * v)
            phi = 2 * np.pi * w

            x = r * np.sin(theta) * np.cos(phi) + self.center[0]
            y = r * np.sin(theta) * np.sin(phi) + self.center[1]
            z = r * np.cos(theta) + self.center[2]

            # Apply z_cut if specified
            if self.z_cut is not None:
                mask = z > self.z_cut
                x, y, z = x[mask], y[mask], z[mask]
                n_effective = len(x)
                h = self.radius + self.center[2] - self.z_cut
                h = np.clip(h, 0, 2*self.radius)
                volume = (1/3) * np.pi * h**2 * (3*self.radius - h)
            else:
                n_effective = int(n_samples)
                volume = (4/3) * np.pi * self.radius**3

            # Interpolate density at the random points
            if cut_r is not None:
                field_interp = sim.load_field(
                    fields=[field],
                    snapshot=[int(t)],
                    cut=(self.center[0], self.center[1], self.center[2], cut_r),
                )
            else:
                field_interp = sim.load_field(
                    fields=[field],
                    snapshot=[int(t)],
                    cut=(self.center[0], self.center[1], self.center[2], self.radius*2),
                )

            rho = field_interp.evaluate(
                time=t,
                var1=x,
                var2=y,
                var3=z,
                interpolator=interpolator,
                method=method
            )
            avg_rho = np.mean(rho[~np.isnan(rho)])
            mass[i] = avg_rho * volume

        if len(mass) == 1:
            return mass[0]
        return mass


    def mass_flux(self, sim, field_density='gasdens', field_velocity='gasv',
                snapshot=[0, 1], interpolator='griddata', method='linear',
                follow_planet=True, planet_index=0,
                correct_normals=True, relative_velocity=False):
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

        Returns
        -------
        numpy.ndarray
            Array of flux values, one per requested snapshot.

        Examples
        --------
        Compute accretion rate (mass flux) onto a planet:
        
        >>> surface = fp.Flux.Surface(type='sphere', radius=0.5, subdivisions=3)
        >>> mdot = surface.mass_flux(sim, field_density='gasdens', field_velocity='gasv', follow_planet=True)
        >>> plt.plot(mdot)
        """

        steps = snapshot[1] - snapshot[0] + 1
        times = np.linspace(snapshot[0], snapshot[1], steps)
        flux = np.zeros(len(times))

        # Initial planet for scaling
        planet0 = sim.load_planets()[planet_index]
        factor = self.radius / planet0.hill_radius

        for i, t in enumerate(tqdm(times, desc="Calculating mass flux")):

            # -------------------------------------------------------------
            # Update the surface position and scale if following the planet
            # -------------------------------------------------------------
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])

                if hasattr(planet, 'hill_radius'):
                    self.radius = factor * planet.hill_radius

                self.tessellate()

            # Planet velocity (for relative velocities)
            vpx, vpy, vpz = planet.vel.x, planet.vel.y, planet.vel.z

            # -------------------------------------------------------------
            # Select geometric properties of the surface
            # -------------------------------------------------------------
            if self.type == "sphere":
                centers = self.centers
                normals = self.normals
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
                centers = self.centers
                normals = self.normals
                areas = self.areas
                surface_cut = (self.center[0], self.center[1], self.center[2], 2*self.radius)

            else:
                raise ValueError("Unsupported surface type.")

            # -------------------------------------------------------------
            # Load both fields simultaneously into a single DataFrame
            # -------------------------------------------------------------
            fields = sim.load_field(
                fields=[field_density, field_velocity],
                snapshot=[int(t)],
                cut=surface_cut
            )
        

            # -------------------------------------------------------------
            # Interpolate density
            # -------------------------------------------------------------
            rho = fields.evaluate(
                time=t,
                var1=centers[:, 0],
                var2=centers[:, 1],
                var3=centers[:, 2],
                interpolator=interpolator,
                method=method,
                field="gasdens"
            )

            # -------------------------------------------------------------
            # Interpolate velocity vector
            # -------------------------------------------------------------
            vel = fields.evaluate(
                time=t,
                var1=centers[:, 0],
                var2=centers[:, 1],
                var3=centers[:, 2],
                interpolator=interpolator,
                method=method,
                field="gasv"
            )

            # Shape fix (3,N → N,3)
            if vel.ndim == 2 and vel.shape[0] == 3:
                vel = vel.T

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
        spherical-polar grid accounting for the region geometry defined by
        this Surface instance. ``self.type`` must be either ``'sphere'`` or
        ``'cylinder'``; the integration mask is constructed accordingly.

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
        # Load grid info once
        # ----------------------------
        gas0 = sim.load_field(field, snapshot=times[0], interpolate=False)
        r_arr  = gas0.domains.r
        th_arr = gas0.domains.theta
        ph_arr = gas0.domains.phi

        TH, RR, PH = np.meshgrid(th_arr, r_arr, ph_arr, indexing='ij')

        X = RR * np.sin(TH) * np.cos(PH)
        Y = RR * np.sin(TH) * np.sin(PH)
        Z = RR * np.cos(TH)

        # ----------------------------------------
        # Precompute cell volumes (FARGO3D metric)
        # ----------------------------------------
        dr  = np.diff(r_arr)
        dth = np.diff(th_arr)
        dph = np.diff(ph_arr)

        dr_full  = np.empty_like(r_arr);     dr_full[:-1]  = dr;  dr_full[-1]  = dr[-1]
        dth_full = np.empty_like(th_arr);    dth_full[:-1] = dth; dth_full[-1] = dth[-1]
        dph_full = np.empty_like(ph_arr);    dph_full[:-1] = dph; dph_full[-1] = dph[-1]

        DR  = dr_full[np.newaxis, :, np.newaxis]
        DTH = dth_full[:, np.newaxis, np.newaxis]
        DPH = dph_full[np.newaxis, np.newaxis, :]

        dV = (RR**2) * np.sin(TH) * DR * DTH * DPH

        # ----------------------------------------
        # Detect geometry
        # ----------------------------------------
        geom = self.type.lower()

        if geom not in ['sphere', 'cylinder']:
            raise ValueError("Surface.type must be 'sphere' or 'cylinder'")

        # Loop over snapshots
        for t in times:

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
            rho = sim.load_field(field, snapshot=t, interpolate=False).data

            # Enclosed mass
            M = np.sum(rho[mask] * dV[mask])
            masses.append(M)

            # Resolution info
            if return_resolution:
                idx_th, idx_r, idx_ph = np.where(mask)

                resolutions.append({
                    "snapshot": t,
                    "mass": M,
                    "geometry": geom,
                    "R_extent": Rlim,
                    "H_extent": Hlim,
                    "N_theta": len(np.unique(idx_th)),
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
