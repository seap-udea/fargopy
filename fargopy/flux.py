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
    """
    Class to generate and manage 3D surfaces (e.g., spheres, cylinders) for integration and analysis
    of simulation data. Provides tessellation, geometric properties, and methods for mass and flux
    calculations over the defined surface.
    """

    def __init__(self, type="sphere", radius=1.0, height=None, subdivisions=1, center=(0.0, 0.0, 0.0), z_cut=None):
        """
        Initialize a Surface object.

        Parameters
        ----------
        type : str
            Type of surface ('sphere' or 'cylinder').
        radius : float
            Radius of the surface.
        height : float, optional
            Height of the cylinder (required if type='cylinder').
        subdivisions : int
            Number of subdivisions for tessellation (higher means finer mesh).
        center : tuple of float
            Center coordinates (x, y, z) of the surface.
        z_cut : float, optional
            If specified, only include triangles with center z >= z_cut (for hemispheres).
        """
        self.type = type
        self.radius = radius
        self.height = height
        self.subdivisions = subdivisions
        self.center = np.array(center)
        self.z_cut = z_cut  # New parameter for z-cut

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
        else:
            raise ValueError("Unsupported surface type. Use 'sphere' or 'cylinder'.")

    def _tessellate_sphere(self):
        """
        Tessellate the sphere using recursive subdivision of an icosahedron.
        Sets up the triangles, centers, and areas arrays.
        Applies z_cut if specified.
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

        # Apply z_cut filter if specified
        if self.z_cut is not None:
            valid_indices = self.centers[:, 2] >= self.z_cut
            self.centers = self.centers[valid_indices]
            self.triangles = self.triangles[valid_indices]
            self.areas = self.areas[valid_indices]
            # Ensure normals are also filtered

        self._calculate_all_triangle_areas()
        self._calculate_normals()
        self.volume = self.areas * (self.radius / 3)
        self.normals = self.normals[valid_indices] 

    def _tessellate_cylinder(self):
        """
        Tessellate a cylinder into top, bottom, and lateral surfaces.
        Sets up arrays for centers, normals, and areas for each part.
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

    @staticmethod
    def _normalize(v):
        """
        Normalize a vector.

        Parameters
        ----------
        v : np.ndarray
            Input vector.

        Returns
        -------
        np.ndarray
            Normalized vector.
        """
        return v / np.linalg.norm(v)

    def _subdivide_triangle(self, v1, v2, v3, depth):
        """
        Recursively subdivide a triangle for tessellation.

        Parameters
        ----------
        v1, v2, v3 : np.ndarray
            Vertices of the triangle.
        depth : int
            Subdivision depth.
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
        """
        Calculate the centroid of each triangle in the tessellation.
        """
        self.centers = np.mean(self.triangles, axis=1)

    @staticmethod
    def _calculate_triangle_area(v1, v2, v3):
        """
        Calculate the area of a triangle given its vertices.

        Parameters
        ----------
        v1, v2, v3 : np.ndarray
            Vertices of the triangle.

        Returns
        -------
        float
            Area of the triangle.
        """
        side1 = v2 - v1
        side2 = v3 - v1
        cross_product = np.cross(side1, side2)
        area = np.linalg.norm(cross_product) / 2
        return area

    def _calculate_all_triangle_areas(self):
        """
        Calculate the area for all triangles in the tessellation.
        """
        for i, (v1, v2, v3) in enumerate(self.triangles):
            self.areas[i] = self._calculate_triangle_area(v1, v2, v3)

    def _calculate_normals(self):
        """
        Calculate the normal vector for each triangle in the tessellation.
        Ensures normals point outward from the surface.
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
        """
        Re-tessellate the surface (sphere or cylinder) based on current parameters.
        """
        if self.type == "sphere":
            self._tessellate_sphere()
        elif self.type == "cylinder":
            self._tessellate_cylinder()
        else:
            raise ValueError("Unsupported surface type. Use 'sphere' or 'cylinder'.")

    def generate_dataframe(self):
        """
        Generate a pandas DataFrame with tessellation data (centers, normals, areas).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 'Center', 'Normal', 'Area'.
        """
        data = []
        for i, (center, normal, area) in enumerate(zip(self.centers, self.normals, self.areas)):
            data.append({
                "Center": center.tolist(),
                "Normal": normal.tolist(),
                "Area": area
            })
        return pd.DataFrame(data)

    def total_mass(self, sim, field='gasdens', n_samples=10000, snapshot=[0,1], interpolator='griddata', method='linear', cut_r=None, follow_planet=True, planet_index=0):
        """
        Estimate the total mass inside the surface using Monte Carlo sampling.

        Parameters
        ----------
        sim : Simulation
            Simulation object (must have load_field method).
        field : str
            Density field name (default 'gasdens').
        n_samples : int
            Number of random points for Monte Carlo integration.
        snapshot : int or list
            Snapshot(s) to evaluate the field.
        interpolator : str
            Interpolation algorithm.
        method : str
            Interpolation method.
        cut_r : float, optional
            Cut radius for loading field data.
        follow_planet : bool
            If True, update surface to follow planet at each snapshot.
        planet_index : int
            Index of the planet to follow.

        Returns
        -------
        float or np.ndarray
            Estimated total mass (array if multiple snapshots).
        """
        if isinstance(snapshot, int):
            times = [snapshot]
        else:
            times = np.linspace(snapshot[0], snapshot[1], snapshot[1]-snapshot[0]+1)
        mass = np.zeros(len(times))

        for i, t in enumerate(tqdm(times, desc="Calculating total mass")):
            # Update surface if following planet
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])
                if hasattr(planet, 'hill_radius'):
                    self.radius = planet.hill_radius
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
                    interpolate=True
                )
            else:
                field_interp = sim.load_field(
                    fields=[field],
                    snapshot=[int(t)],
                    cut=(self.center[0], self.center[1], self.center[2], self.radius*2),
                    interpolate=True
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

    def mass_flux(self, sim, field_density='gasdens', field_velocity='gasv', snapshot=[0, 1], interpolator='griddata', method='linear', follow_planet=True, planet_index=0):
        """
        Compute the total mass flux through the surface for a range of snapshots.

        Parameters
        ----------
        sim : Simulation
            Simulation object (must have load_field and load_planets methods).
        field_density : str
            Name of the density field (default 'gasdens').
        field_velocity : str
            Name of the velocity field (default 'gasv').
        snapshot : list
            Range of snapshots (e.g., [0, 1]).
        interpolator : str
            Interpolation algorithm.
        method : str
            Interpolation method.
        follow_planet : bool
            If True, update surface to follow planet at each snapshot.
        planet_index : int
            Index of the planet to follow.

        Returns
        -------
        np.ndarray
            Array with the total mass flux for each snapshot.
        """
        steps = snapshot[1] - snapshot[0] + 1
        times = np.linspace(snapshot[0], snapshot[1], steps)
        flux = np.zeros(len(times))

        for i, t in enumerate(tqdm(times, desc="Calculating mass flux")):
            # Update surface parameters if following a planet
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])
                if hasattr(planet, 'hill_radius'):
                    self.radius = planet.hill_radius
                self.tessellate()  # Recompute tessellation for new center/radius

            # Select centers, normals, and areas according to surface type
            if self.type == "sphere":
                centers = self.centers
                normals = self.normals
                areas = self.areas
                surface_cut = (self.center[0], self.center[1], self.center[2], 2*self.radius)
            elif self.type == "cylinder":
                centers = np.concatenate([self.top_centers, self.bottom_centers, self.lateral_centers], axis=0)
                normals = np.concatenate([self.top_normals, self.bottom_normals, self.lateral_normals], axis=0)
                areas = np.concatenate([
                    np.full(self.top_centers.shape[0], self.top_areas),
                    np.full(self.bottom_centers.shape[0], self.bottom_areas),
                    np.full(self.lateral_centers.shape[0], self.lateral_areas)
                ])
                surface_cut = (self.center[0], self.center[1], self.center[2], 2*self.radius, 2*self.height)
            else:
                raise ValueError("Unsupported surface type. Use 'sphere' or 'cylinder'.")

            # Load fields for this snapshot and cut
            fields = sim.load_field(
                fields=[field_density, field_velocity],
                snapshot=[int(t)],
                interpolate=True,
                cut=surface_cut
            )

            # Evaluate density at the surface centers
            rho = fields[0].evaluate(
                time=t,
                var1=centers[:, 0],
                var2=centers[:, 1],
                var3=centers[:, 2],
                interpolator=interpolator,
                method=method
            )
            # Evaluate velocity at the surface centers
            vel = fields[1].evaluate(
                time=t,
                var1=centers[:, 0],
                var2=centers[:, 1],
                var3=centers[:, 2],
                interpolator=interpolator,
                method=method
            )
            if vel.shape[0] == 3 and vel.shape[1] == len(rho):
                vel = vel.T  # (N, 3)
            v_dot_n = np.einsum('ij,ij->i', vel, normals)
            dF = rho * v_dot_n * areas
            flux[i] = np.nansum(dF)

        return flux


# class Analyzer:
#     """
#     General class for performing calculations and integrals on 3D surfaces or 2D slices
#     using simulation data. Handles field loading, interpolation, and integration.
#     """

#     def __init__(self, simulation, surface=None, slice=None, fields=None, snapshots=(1, 10), interpolator='griddata', method='linear', interp_kwargs=None):
#         """
#         Initialize an Analyzer object.

#         Parameters
#         ----------
#         simulation : Simulation
#             The simulation object (e.g., fp.Simulation).
#         surface : Surface, optional
#             The 3D surface object for 3D calculations.
#         slice : str, optional
#             The 2D slice specification for 2D calculations.
#         fields : list of str
#             List of fields to load (e.g., ['gasdens', 'gasv']).
#         snapshots : tuple
#             Range of snapshots to load (e.g., (1, 10)).
#         interpolator : str
#             Interpolation algorithm.
#         method : str
#             Interpolation method.
#         interp_kwargs : dict, optional
#             Extra kwargs for the interpolator.
#         """
#         self.sim = simulation
#         self.surface = surface
#         self.slice = slice
#         self.fields = fields
#         self.snapshots = snapshots
#         self.interpolator = interpolator
#         self.method = method
#         self.interp_kwargs = interp_kwargs or {}
#         self.time = None
#         self.interpolated_fields = None

#         # Load fields with interpolation
#         self.load_fields()

#     def load_fields(self):
#         """
#         Load and interpolate the fields based on the provided configuration.
#         Ensures self.interpolated_fields is always a list, even for a single field.
#         """
#         if self.surface is not None:  # 3D case
#             self.interpolated_fields = self.sim.load_field(
#                 fields=self.fields,
#                 snapshot=self.snapshots,
#                 interpolate=True
#             )
#             # Ensure it's always a list
#             if not isinstance(self.interpolated_fields, (list, tuple)):
#                 self.interpolated_fields = [self.interpolated_fields]
#         elif self.slice is not None:  # 2D case
#             self.interpolated_fields = self.sim.load_field(
#                 fields=self.fields,
#                 slice=self.slice,
#                 snapshot=self.snapshots,
#                 interpolate=True
#             )
#             if not isinstance(self.interpolated_fields, (list, tuple)):
#                 self.interpolated_fields = [self.interpolated_fields]
#         else:
#             raise ValueError("Either a surface (3D) or a slice (2D) must be specified.")

#     def evaluate_fields(
#         self, time, coordinates,
#         griddata_kwargs=None, rbf_kwargs=None, idw_kwargs=None, linearnd_kwargs=None
#     ):
#         """
#         Evaluate interpolated fields at a given time and coordinates, allowing specific kwargs for each interpolator.

#         Parameters
#         ----------
#         time : float
#             The time at which to evaluate.
#         coordinates : tuple of np.ndarray
#             The coordinates (x, y, z) or (x, z).
#         griddata_kwargs : dict, optional
#             Optional kwargs for griddata.
#         rbf_kwargs : dict, optional
#             Optional kwargs for RBF.
#         idw_kwargs : dict, optional
#             Optional kwargs for IDW.
#         linearnd_kwargs : dict, optional
#             Optional kwargs for LinearND.

#         Returns
#         -------
#         dict
#             Dictionary with the field values.
#         """
#         results = {}
#         for field, interp in zip(self.fields, self.interpolated_fields):
#             # Prepare kwargs in the same format as FieldInterpolator.evaluate
#             eval_kwargs = {}
#             if griddata_kwargs is not None:
#                 eval_kwargs["griddata_kwargs"] = griddata_kwargs
#             if rbf_kwargs is not None:
#                 eval_kwargs["rbf_kwargs"] = rbf_kwargs
#             if idw_kwargs is not None:
#                 eval_kwargs["idw_kwargs"] = idw_kwargs
#             if linearnd_kwargs is not None:
#                 eval_kwargs["linearnd_kwargs"] = linearnd_kwargs

#             field_values = interp.evaluate(
#                 time=time,
#                 var1=coordinates[0],
#                 var2=coordinates[1],
#                 var3=coordinates[2] if len(coordinates) > 2 else None,
#                 interpolator=self.interpolator,
#                 method=self.method,
#                 **eval_kwargs
#             )

#             if field == 'gasv':
#                 results[field] = np.array(field_values).T
#             else:
#                 results[field] = field_values

#         return results
    

#     def calculate_integral(self, integrand, time_steps, dtype):
#         """
#         Calculates an integral based on the provided integrand and integration type.

#         Parameters
#         ----------
#         integrand : callable
#             Function defining the integrand, accepting field values as keyword arguments.
#         time_steps : int
#             Number of time steps for the calculation.
#         dtype : str
#             Type of integration: 'area', 'volume', or 'line'.

#         Returns
#         -------
#         np.ndarray
#             Array of results for each time step.
#         """
#         self.time = np.linspace(0, 1, time_steps)
#         results = np.zeros(len(self.time))

#         if self.surface is not None:  # 3D case
#             xc, yc, zc = self.surface.centers[:, 0], self.surface.centers[:, 1], self.surface.centers[:, 2]
#             # Select weights according to the integration type
#             if dtype == 'volume':
#                 weights = self.surface.volume
#             elif dtype == 'area':
#                 weights = self.surface.areas
#             else:
#                 raise ValueError("For 3D, dtype must be 'area' or 'volume'.")
#             for i, t in enumerate(tqdm(self.time, desc="Calculating integral")):
#                 field_values = self.evaluate_fields(t, (xc, yc, zc))
#                 integrand_values = integrand(**field_values)
#                 results[i] = np.sum(integrand_values * weights)

#         elif self.slice is not None:  # 2D case
#             n_points = len(self.surface.centers)
#             angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
#             x = self.surface.center[0] + self.surface.radius * np.cos(angles)
#             y = self.surface.center[1] + self.surface.radius * np.sin(angles)
#             # Select weights according to the integration type
#             if dtype == 'line':
#                 dl = 2 * np.pi * self.surface.radius / n_points
#                 weights = dl
#             elif dtype == 'area':
#                 weights = np.ones(n_points)  # You can define area elements if needed
#             else:
#                 raise ValueError("For 2D, dtype must be 'line' or 'area'.")
#             for i, t in enumerate(tqdm(self.time, desc="Calculating integral")):
#                 field_values = self.evaluate_fields(t, (x, y))
#                 integrand_values = integrand(**field_values)
#                 results[i] = np.sum(integrand_values * weights)

#         else:
#             raise ValueError("Either a surface (3D) or a slice (2D) must be specified.")

#         return results
    
#     def total_masss(sim, time_steps, surface):
#         """
#         Calculate the total mass within the defined surface over time.

#         Parameters
#         ----------
#         sim : Simulation
#             Simulation object.
#         time_steps : tuple
#             Limits of time steps for the calculation.
#         surface : Surface
#             Surface object (must have .type, .center, .radius, .height if cylinder).

#         Returns
#         -------
#         np.ndarray
#             Array of total mass at each time step.
#         """

#         # Detect surface type and build the cut tuple
#         if surface.type == "sphere":
#             # cut = (xc, yc, zc, r)
#             surface_cut = (surface.center[0], surface.center[1], surface.center[2], surface.radius)
#             volume = surface.volume
#             centers = surface.centers
#         elif surface.type == "cylinder":
#             # cut = (xc, yc, zc, rc, hc)
#             surface_cut = (surface.center[0], surface.center[1], surface.center[2], surface.radius, surface.height)
#             # For cylinder, you may want to use lateral_centers and lateral_areas, or combine all
#             # Here, we use all points (top, bottom, lateral) and sum their contributions
#             centers = np.concatenate([surface.top_centers, surface.bottom_centers, surface.lateral_centers], axis=0)
#             volume = np.concatenate([
#                 np.full(surface.top_centers.shape[0], surface.top_areas),
#                 np.full(surface.bottom_centers.shape[0], surface.bottom_areas),
#                 np.full(surface.lateral_centers.shape[0], surface.lateral_areas)
#             ])
#         else:
#             raise ValueError("Unsupported surface type. Use 'sphere' or 'cylinder'.")

#         steps = time_steps[1] - time_steps[0]
#         time = np.linspace(time_steps[0], time_steps[1], steps)
#         mass = np.zeros(len(time))

#         gasdens = sim.load_field(
#             fields=['gasdens'],
#             snapshot=time_steps,
#             cut=surface_cut,
#             interpolate=True
#         )

#         for i, t in enumerate(tqdm(time, desc="Calculating total mass")):
#             rho = gasdens.evaluate(
#                 time=t,
#                 var1=centers[:, 0],
#                 var2=centers[:, 1],
#                 var3=centers[:, 2]
#             )
#             mass[i] = np.sum(rho * volume)

#         return mass