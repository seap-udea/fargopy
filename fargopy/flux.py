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

        # Si hay z_cut, hacer clipping geométrico en vez de solo filtrar por centroide
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

        planet = sim.load_planets()[planet_index]
        factor = self.radius/planet.hill_radius
        for i, t in enumerate(tqdm(times, desc="Calculating mass flux")):
            # Update surface parameters if following a planet
            if follow_planet:
                planet = sim.load_planets(snapshot=int(t))[planet_index]
                self.center = np.array([planet.pos.x, planet.pos.y, planet.pos.z])
                if hasattr(planet, 'hill_radius'):
                    self.radius = factor*planet.hill_radius
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


