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
    Factory class to generate and manage surfaces (e.g., spheres).
    """

    def __init__(self):
        self.surface = None

    def Sphere(self, radius=1.0, subdivisions=1, center=(0.0, 0.0, 0.0)):
        self.surface = self.Sphere(radius, subdivisions, center)
        return self.surface

    class Sphere:
        def __init__(self, radius=1.0, subdivisions=1, center=(0.0, 0.0, 0.0)):
            self.radius = radius
            self.subdivisions = subdivisions
            self.center = np.array(center)
            self.num_triangles = 20 * (4 ** subdivisions)
            self.triangles = np.zeros((self.num_triangles, 3, 3))
            self.centers = np.zeros((self.num_triangles, 3))
            self.areas = np.zeros(self.num_triangles)
            self.triangle_index = 0
            self.volume = None
            self.tessellate()

        def filter(self, condition):
            """
            Filter the sphere's centers, areas, normals, etc. by a string condition.
            Example: sphere.filter("z > 0")
            """
            x = self.centers[:, 0]
            y = self.centers[:, 1]
            z = self.centers[:, 2]
            mask = eval(condition)
            self.centers = self.centers[mask]
            self.areas = self.areas[mask]
            if hasattr(self, "normals"):
                self.normals = self.normals[mask]
            if hasattr(self, "volume"):
                self.volume = self.volume[mask]

        @staticmethod
        def normalize(v):
            return v / np.linalg.norm(v)

        def subdivide_triangle(self, v1, v2, v3, depth):
            if depth == 0:
                self.triangles[self.triangle_index] = [v1 + self.center, v2 + self.center, v3 + self.center]
                self.triangle_index += 1
                return
            v12 = self.normalize((v1 + v2) / 2) * self.radius
            v23 = self.normalize((v2 + v3) / 2) * self.radius
            v31 = self.normalize((v3 + v1) / 2) * self.radius
            self.subdivide_triangle(v1, v12, v31, depth - 1)
            self.subdivide_triangle(v12, v2, v23, depth - 1)
            self.subdivide_triangle(v31, v23, v3, depth - 1)
            self.subdivide_triangle(v12, v23, v31, depth - 1)

        def generate_icosphere(self):
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            patterns = [
                (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
                (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
                (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
            ]
            vertices = np.array([self.normalize(np.array(p)) * self.radius for p in patterns])
            faces = [
                (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
                (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
                (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
                (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
            ]
            for face in faces:
                v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
                self.subdivide_triangle(v1, v2, v3, self.subdivisions)

        def calculate_polygon_centers(self):
            self.centers = np.mean(self.triangles, axis=1)

        @staticmethod
        def calculate_triangle_area(v1, v2, v3):
            side1 = v2 - v1
            side2 = v3 - v1
            cross_product = np.cross(side1, side2)
            area = np.linalg.norm(cross_product) / 2
            return area

        def calculate_all_triangle_areas(self):
            for i, (v1, v2, v3) in enumerate(self.triangles):
                self.areas[i] = self.calculate_triangle_area(v1, v2, v3)

        def calculate_normals(self):
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
            self.generate_icosphere()
            self.calculate_polygon_centers()
            self.calculate_all_triangle_areas()
            self.calculate_normals()
            self.volume = self.areas * (self.radius / 3)

        def generate_dataframe(self):
            data = []
            for i, (triangle, center, area) in enumerate(zip(self.triangles, self.centers, self.areas)):
                data.append({
                    "Triangle": triangle.tolist(),
                    "Center": center.tolist(),
                    "Area": area
                })
            df = pd.DataFrame(data)
            return df



class Analyzer:
    def __init__(self, simulation, surface=None, slice=None, fields=None, snapshots=(1, 10), interpolator='griddata', method='linear', interp_kwargs=None):
        """
        General class for performing calculations on 3D surfaces or 2D planes.

        :param simulation: The simulation object (e.g., fp.Simulation).
        :param surface: The 3D surface object (e.g., Sphere) for 3D calculations.
        :param plane: The 2D plane ('XY', 'XZ', etc.) for 2D calculations.
        :param angle: The angle for slicing the 2D plane (e.g., 'phi=0').
        :param fields: List of fields to load (e.g., ['gasdens', 'gasv']).
        :param snapshots: Tuple indicating the range of snapshots to load (e.g., (1, 10)).
        :param interpolator: Interpolation algorithm ('griddata', 'rbf', etc.).
        :param method: Interpolation method ('linear', 'cubic', etc.).
        :param interp_kwargs: Dict of extra kwargs for the interpolator.
        """
        self.sim = simulation
        self.surface = surface
        self.slice = slice
        self.fields = fields
        self.snapshots = snapshots
        self.interpolator = interpolator
        self.method = method
        self.interp_kwargs = interp_kwargs or {}
        self.time = None
        self.interpolated_fields = None

        # Load fields with interpolation
        self.load_fields()

    def load_fields(self):
        """
        Loads and interpolates the fields based on the provided configuration.
        Ensures self.interpolated_fields is always a list, even for a single field.
        """
        if self.surface is not None:  # 3D case
            self.interpolated_fields = self.sim.load_field(
                fields=self.fields,
                snapshot=self.snapshots,
                interpolate=True
            )
            # Ensure it's always a list
            if not isinstance(self.interpolated_fields, (list, tuple)):
                self.interpolated_fields = [self.interpolated_fields]
        elif self.slice is not None:  # 2D case
            self.interpolated_fields = self.sim.load_field(
                fields=self.fields,
                slice=self.slice,
                snapshot=self.snapshots,
                interpolate=True
            )
            if not isinstance(self.interpolated_fields, (list, tuple)):
                self.interpolated_fields = [self.interpolated_fields]
        else:
            raise ValueError("Either a surface (3D) or a slice (2D) must be specified.")


    def evaluate_fields(
        self, time, coordinates,
        griddata_kwargs=None, rbf_kwargs=None, idw_kwargs=None, linearnd_kwargs=None
    ):
        """
        Evaluate interpolated fields at a given time and coordinates, allowing specific kwargs for each interpolator.

        :param time: The time at which to evaluate.
        :param coordinates: The coordinates (x, y, z) or (x, z).
        :param griddata_kwargs: Optional kwargs for griddata.
        :param rbf_kwargs: Optional kwargs for RBF.
        :param idw_kwargs: Optional kwargs for IDW.
        :param linearnd_kwargs: Optional kwargs for LinearND.
        :return: Dictionary with the field values.
        """
        results = {}
        for field, interp in zip(self.fields, self.interpolated_fields):
            # Prepare kwargs in the same format as FieldInterpolator.evaluate
            eval_kwargs = {}
            if griddata_kwargs is not None:
                eval_kwargs["griddata_kwargs"] = griddata_kwargs
            if rbf_kwargs is not None:
                eval_kwargs["rbf_kwargs"] = rbf_kwargs
            if idw_kwargs is not None:
                eval_kwargs["idw_kwargs"] = idw_kwargs
            if linearnd_kwargs is not None:
                eval_kwargs["linearnd_kwargs"] = linearnd_kwargs

            field_values = interp.evaluate(
                time=time,
                var1=coordinates[0],
                var2=coordinates[1],
                var3=coordinates[2] if len(coordinates) > 2 else None,
                interpolator=self.interpolator,
                method=self.method,
                **eval_kwargs
            )

            if field == 'gasv':
                results[field] = np.array(field_values).T
            else:
                results[field] = field_values

        return results
    

    def hill_radius(self, planet_index=0):
        """
        Calculates the Hill radius of the selected planet using simulation parameters.
        Returns the Hill radius in cm and AU.
        """
        # Conversion constants
        AU_to_cm = 1.495978707e13
        Mjup_to_g = 1.898e30
        Msun_to_g = 1.989e33

        # Check planet data
        if not hasattr(self.sim, "planets") or not self.sim.planets:
            raise ValueError("No planet data found. Run sim.load_planet_summary() first.")

        # Check stellar mass in macros
        if not hasattr(self.sim, "simulation_macros") or 'MSTAR' not in self.sim.simulation_macros:
            raise ValueError("Stellar mass (MSTAR) not found. Run sim.load_macros() first.")

        planet = self.sim.planets[planet_index]
        a_au = planet['distance']  # AU
        m_jup = planet['mass']     # Mjup

        # Get stellar mass in Msun and convert to grams
        mstar_msun = self.sim.simulation_macros['MSTAR']
        mstar_g = float(mstar_msun) * Msun_to_g

        # Convert planet mass to grams and distance to cm
        a_cm = a_au * AU_to_cm
        m_p = m_jup * Mjup_to_g

        # Hill radius formula
        r_hill_cm = a_cm * (m_p / (3 * mstar_g))**(1/3)
        r_hill_au = r_hill_cm / AU_to_cm

        return r_hill_cm, r_hill_au

    def calculate_integral(self, integrand, time_steps, dtype):
        """
        Calculates an integral based on the provided integrand and integration type.

        :param integrand: A callable function defining the integrand.
        :param time_steps: Number of time steps for the calculation.
        :param type: 'line', 'area', or 'volume' (default: 'area').
        :return: Array of results for each time step.
        """
        self.time = np.linspace(0, 1, time_steps)
        results = np.zeros(len(self.time))

        if self.surface is not None:  # 3D case
            xc, yc, zc = self.surface.centers[:, 0], self.surface.centers[:, 1], self.surface.centers[:, 2]
            # Select weights according to the integration type
            if dtype == 'volume':
                weights = self.surface.volume
            elif dtype == 'area':
                weights = self.surface.areas
            else:
                raise ValueError("For 3D, dtype must be 'area' or 'volume'.")
            for i, t in enumerate(tqdm(self.time, desc="Calculating integral")):
                field_values = self.evaluate_fields(t, (xc, yc, zc))
                integrand_values = integrand(**field_values)
                results[i] = np.sum(integrand_values * weights)

        elif self.slice is not None:  # 2D case
            n_points = len(self.surface.centers)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            x = self.surface.center[0] + self.surface.radius * np.cos(angles)
            y = self.surface.center[1] + self.surface.radius * np.sin(angles)
            # Select weights according to the integration type
            if dtype == 'line':
                dl = 2 * np.pi * self.surface.radius / n_points
                weights = dl
            elif dtype == 'area':
                weights = np.ones(n_points)  # You can define area elements if needed
            else:
                raise ValueError("For 2D, dtype must be 'line' or 'area'.")
            for i, t in enumerate(tqdm(self.time, desc="Calculating integral")):
                field_values = self.evaluate_fields(t, (x, y))
                integrand_values = integrand(**field_values)
                results[i] = np.sum(integrand_values * weights)

        else:
            raise ValueError("Either a surface (3D) or a slice (2D) must be specified.")

        return results
    



class FluxAnalyzer3D:

    def __init__(self, output_dir, sphere_center=(0.0, 0.0, 0.0), radius=1.0, subdivisions=1, snapi=110, snapf=210):
        """
        Initializes the class with the simulation and sphere parameters.
        """
        self.sim = fp.Simulation(output_dir=output_dir)
        self.radius = radius
        self.data_handler = fargopy.DataHandler(self.sim)
        self.data_handler.load_data(snapi=snapi, snapf=snapf)  # Load 3D data using the unified method
        self.sphere = Sphere(radius=radius, subdivisions=subdivisions, center=sphere_center)
        self.sphere.tessellate()
        self.sphere_center = np.array(sphere_center)
        self.time = None
        self.snapi = snapi
        self.snapf = snapf
        self.velocities = None
        self.densities = None
        self.normals = None
        self.flows = None

    def interpolate(self, time_steps):
        """Interpolates velocity and density fields at the sphere's points."""
        self.time = np.linspace(0, 1, time_steps)
        xc, yc, zc = self.sphere.centers[:, 0], self.sphere.centers[:, 1], self.sphere.centers[:, 2]

        self.velocities = np.zeros((time_steps, len(xc), 3))
        self.densities = np.zeros((time_steps, len(xc)))

        valid_triangles = None  # To store valid triangles across all time steps

        for i, t in enumerate(tqdm(self.time, desc="Interpolating fields")):
            # Interpolate velocity
            velx, vely, velz = self.data_handler.interpolate_velocity(t, xc, yc, zc)
            self.velocities[i, :, 0] = velx
            self.velocities[i, :, 1] = vely
            self.velocities[i, :, 2] = velz

            # Interpolate density
            rho = self.data_handler.interpolate_density(t, xc, yc, zc)
            self.densities[i] = rho

            # Filter triangles where density is greater than 0
            valid_mask = rho > 0
            if valid_triangles is None:
                valid_triangles = valid_mask  # Initialize valid triangles
            else:
                valid_triangles &= valid_mask  # Keep only triangles valid across all time steps

        # Update sphere centers and areas to include only valid triangles
        self.valid_centers = self.sphere.centers[valid_triangles]
        self.valid_areas = self.sphere.areas[valid_triangles]
        self.valid_normals = None  # Normals will be recalculated for valid triangles

        return self.velocities, self.densities

    def calculate_normals(self):
        """Calculates the normal vectors of the valid triangles."""
        if self.valid_normals is not None:
            return self.valid_normals  # Use cached normals if already calculated

        valid_triangles = self.sphere.triangles[self.sphere.areas > 0]  # Use valid triangles
        self.valid_normals = np.zeros((len(valid_triangles), 3))

        for i, tri in enumerate(valid_triangles):
            AB = tri[1] - tri[0]
            AC = tri[2] - tri[0]
            normal = np.cross(AB, AC)
            normal /= np.linalg.norm(normal)
            centroid = np.mean(tri, axis=0)
            to_centroid = centroid - self.sphere_center
            if np.dot(normal, to_centroid) < 0:
                normal = -normal
            self.valid_normals[i] = normal

        return self.valid_normals

    def calculate_fluxes(self):
        """Calculates the total flux at each time step."""
        if self.valid_normals is None:
            self.calculate_normals()

        self.flows = np.zeros(len(self.time))

        for i in range(len(self.time)):
            total_flux = np.sum(
                self.densities[i][self.sphere.areas > 0] *  # Use valid densities
                np.einsum('ij,ij->i', self.velocities[i][self.sphere.areas > 0], self.valid_normals) *
                self.valid_areas
            )
            self.flows[i] = (total_flux * self.sim.URHO * self.sim.UL**2 * self.sim.UV) * 1e-3 * 1.587e-23  # en Msun_yr

        return self.flows
    
    def calculate_accretion(self):
        """
        Calculates the accretion rate (dM/dt) and the total accreted mass inside the sphere.

        :return: A tuple containing:
            - accretion_rate: Array of accretion rates at each time step in Msun/yr.
            - total_accreted_mass: Total accreted mass over the simulation time in Msun.
        """
        # Ensure densities have been interpolated
        if self.densities is None:
            raise ValueError("Densities have not been interpolated. Call interpolate() first.")

        # Convert density to kg/m³
        rho_conv = self.sim.URHO * 1e3  # g/cm³ to kg/m³

        # Convert radius to meters
        r_m = self.radius * self.sim.UL * 1e-2  # cm to m

        # Convert areas to m² and calculate volume elements
        area_m2 = (self.sim.UL * 1e-2) ** 2  # cm² to m²
        
        vol_elem = self.sphere.areas * area_m2 * (r_m / 3)  # m³

        # Calculate the total mass inside the sphere at each time step
        total_mass = np.array([
            np.sum(self.densities[i] * rho_conv * vol_elem)  # Mass in kg
            for i in range(len(self.time))
        ])

        # Calculate the time step in physical units (seconds)
        dt = (self.time[1] - self.time[0]) * self.sim.UT  # Time step in seconds

        # Calculate the accretion rate as the time derivative of the total mass
        acc_rate = np.gradient(total_mass, dt)  # dM/dt in kg/s

        # Convert accretion rate to Msun/yr
        acc_rate_msun_yr = acc_rate * (1 / 1.989e30) * fp.YEAR  # Convert kg/s to Msun/yr

        # Calculate the total accreted mass (in Msun)
        total_mass_msun = np.sum(acc_rate_msun_yr * dt / fp.YEAR)  # Convert Msun/yr to Msun

        return acc_rate_msun_yr, float(total_mass_msun)


    def plot_fluxes(self):
        """Plots the total flux as a function of time."""
        if self.flows is None:
            raise ValueError("Flows have not been calculated. Call calculate_flows() first.")

        #times
        duration=(self.snapf - self.snapi + 1) * self.sim.UT / fp.YEAR
        times= self.time * duration
        
        start_time = self.snapi * self.sim.UT / fp.YEAR
        times += start_time
        
        average_flux = np.mean(self.flows)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.time,
            y=self.flows,
            mode='lines',
            name='Flux',
            line=dict(color='dodgerblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=self.time,
            y=[average_flux]* len(self.time),
            mode='lines',
            name=f'Avg: {average_flux:.2e} Msun/yr',
            line=dict(color='orangered', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f"Matter Flux over Planet-Centered Sphere (R={self.radius*self.sim.UL/fp.AU:.3f} [AU])",
            xaxis_title="Normalized Time",
            yaxis_title="Flux [Msun/yr]",
            template="plotly_white",
            font=dict(size=14),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True,exponentformat="e"),
        )
        fig.show()

    def planet_sphere(self, snapshot=1):
        """
        Plots the density map in both the XZ  and XY planes for a given snapshot,
        along with the circle representing the tessellation sphere.

        Parameters:
            snapshot (int): The snapshot to visualize (default is 1).
        """
        # Load the density field for the snapshot
        gasdens = self.sim.load_field("gasdens", snapshot=snapshot, type="scalar")

        # Get the density slice and coordinates for the XZ plane
        density_slice_xz, mesh_xz = gasdens.meshslice(slice="phi=0")
        x_xz, z_xz = mesh_xz.x, mesh_xz.z

        # Get the density slice and coordinates for the XY plane
        density_slice_xy, mesh_xy = gasdens.meshslice(slice="theta=1.56")
        x_xy, y_xy = mesh_xy.x, mesh_xy.y

        # Extract sphere center and radius
        sphere_center_x, sphere_center_y, sphere_center_z = self.sphere.center
        sphere_radius = self.sphere.radius * self.sim.UL / fp.AU  # Convert radius to AU

        # Create the figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Plot the density map for the XZ plane
        c1 = axes[0].pcolormesh(
            x_xz,
            z_xz,
            np.log10(density_slice_xz * self.sim.URHO ),
            cmap="Spectral_r",
            shading="auto"
        )
        fig.colorbar(c1, ax=axes[0], label=r"$\log_{10}(\rho)$ [g/cm³]")
        circle_xz = plt.Circle(
            (sphere_center_x, sphere_center_z),  # Sphere center in XZ plane
            sphere_radius,  # Sphere radius
            color="red",
            fill=False,
            linestyle="--",
            linewidth=3,
            label="Tessellation Sphere"
        )
        axes[0].add_artist(circle_xz)
        axes[0].set_xlabel("X [AU]")
        axes[0].set_ylabel("Z [AU]")
        axes[0].set_xlim(x_xz.min(), x_xz.max())
        axes[0].set_ylim(z_xz.min(), z_xz.max())
        axes[0].legend()

        # Plot the density map for the XY plane
        c2 = axes[1].pcolormesh(
            x_xy,
            y_xy,
            np.log10(density_slice_xy * self.sim.URHO),
            cmap="Spectral_r",
            shading="auto"
        )
        fig.colorbar(c2, ax=axes[1], label=r"$\log_{10}(\rho)$ [g/cm³]")
        circle_xy = plt.Circle(
            (sphere_center_x, sphere_center_y),  # Sphere center in XY plane
            sphere_radius,  # Sphere radius
            color="red",
            fill=False,
            linestyle="--",
            linewidth=3,
            label="Tessellation Sphere"
        )
        axes[1].add_artist(circle_xy)
        axes[1].set_xlabel("X [AU]")
        axes[1].set_ylabel("Y [AU]")
        axes[1].set_xlim(x_xy.min(), x_xy.max())
        axes[1].set_ylim(y_xy.min(), y_xy.max())
        axes[1].legend()



class FluxAnalyzer2D:
    def __init__(self, output_dir, plane="XY",angle="theta=1.56",snapi=110, snapf=210, center=(0,0),radius=1,subdivisions=10):
        """
        Initializes the class for 2D flux analysis.

        :param output_dir: Directory containing simulation data.
        :param plane: Plane to analyze ("XY" or "XZ").
        :param snapi: Initial snapshot index.
        :param snapf: Final snapshot index.
        """
        self.sim = fp.Simulation(output_dir=output_dir)
        self.data_handler = fargopy.DataHandler(self.sim)
        self.data_handler.load_data(plane=plane,angle=angle, snapi=snapi, snapf=snapf)  # Load 2D data
        self.plane = plane
        self.subdivisions = subdivisions
        self.center = center
        self.radius = radius
        self.angle = angle
        self.snapi = snapi
        self.snapf = snapf
        self.time = None
        self.velocities = None
        self.densities = None
        self.flows = None

            
    def interpolate(self, time_steps):
        """
        Interpolates velocity and density fields at the circle's perimeter points.

        :param time_steps: Number of time steps for interpolation.
        """
        self.time = np.linspace(0, 1, time_steps)
        angles = np.linspace(0, 2 * np.pi, self.subdivisions, endpoint=False)

        if self.plane == "XY":
            x = self.center[0] + self.radius * np.cos(angles)
            y = self.center[1] + self.radius * np.sin(angles)
            z = np.zeros_like(x)  # z = 0 for the XY plane
        elif self.plane == "XZ":
            x = self.center[0] + self.radius * np.cos(angles)
            z = self.center[1] + self.radius * np.sin(angles)
            y = np.zeros_like(x)  # y = 0 for the XZ plane

        self.velocities = np.zeros((time_steps, self.subdivisions, 2))
        self.densities = np.zeros((time_steps, self.subdivisions))

        valid_points = None  # To store valid points for all time steps

        for i, t in enumerate(tqdm(self.time, desc="Interpolating fields")):
            if self.plane == "XY":
                vx, vy = self.data_handler.interpolate_velocity(t, x, y)
                rho = self.data_handler.interpolate_density(t, x, y)
            elif self.plane == "XZ":
                vx, vz = self.data_handler.interpolate_velocity(t, x, z)
                rho = self.data_handler.interpolate_density(t, x, z)

            # Filter points where density is not zero
            valid_mask = rho > 0
            if valid_points is None:
                valid_points = valid_mask  # Initialize valid points
            else:
                valid_points &= valid_mask  # Keep only points valid across all time steps

            # Store interpolated values for valid points
            self.velocities[i, valid_mask, 0] = vx[valid_mask]
            self.velocities[i, valid_mask, 1] = vy[valid_mask] if self.plane == "XY" else vz[valid_mask]
            self.densities[i, valid_mask] = rho[valid_mask]

        # Update angles and coordinates to only include valid points
        self.valid_angles = angles[valid_points]
        self.valid_x = x[valid_points]
        self.valid_y = y[valid_points]
        self.valid_z = z[valid_points] if self.plane == "XZ" else np.zeros_like(self.valid_x)

        return self.velocities, self.densities

    def calculate_fluxes(self):
        """
        Calculates the total flux at each time step.
        """
        if self.velocities is None or self.densities is None:
            raise ValueError("Fields have not been interpolated. Call interpolate() first.")

        normals = np.stack((np.cos(self.valid_angles), np.sin(self.valid_angles)), axis=1)
        dl = 2 * np.pi * self.radius / self.subdivisions  # Differential length

        self.flows = np.zeros(len(self.time))

        for i in range(len(self.time)):
            velocity_dot_normal = np.einsum('ij,ij->i', self.velocities[i, :len(self.valid_angles)], normals)
            total_flux = np.sum(self.densities[i, :len(self.valid_angles)] * velocity_dot_normal * dl)
            self.flows[i] = (total_flux * self.sim.URHO * self.sim.UL**2 * self.sim.UV)* 1e-3 * 1.587e-23   # Convert to physical units

        return self.flows
    
    def calculate_accretion(self):
        """
        Calculates the accretion rate (dM/dt) and the total accreted mass in the 2D plane.

        :return: A tuple containing:
            - accretion_rate: Array of accretion rates at each time step in Msun/yr.
            - total_accreted_mass: Total accreted mass over the simulation time in Msun.
        """
        # Ensure densities have been interpolated
        if self.densities is None:
            raise ValueError("Densities have not been interpolated. Call interpolate() first.")

        # Differential area for each subdivision
        dA = (np.pi * self.radius**2) / self.subdivisions  # Area of each segment in AU²

        # Convert density to kg/m² (2D case)
        rho_conv = self.sim.UM/self.sim.UL**2 *10  # g/cm2 to kg/m2

        # Convert dA to m²
        dA_m2 = dA * (self.sim.UL * 1e-2)**2  # Convert from cm² to m²

        # Calculate the total mass in the 2D plane at each time step
        total_mass = np.array([
            np.sum(self.densities[i] * rho_conv * dA_m2)  # Mass in kg
            for i in range(len(self.time))
        ])

        # Calculate the time step in physical units (seconds)
        dt = (self.time[1] - self.time[0]) * self.sim.UT  # Time step in seconds

        # Calculate the accretion rate as the time derivative of the total mass
        acc_rate = np.gradient(total_mass, dt)  # dM/dt in kg/s

        # Convert accretion rate to Msun/yr
        acc_rate_msun_yr = acc_rate * (1 / 1.989e30) * fp.YEAR  # Convert kg/s to Msun/yr

        # Calculate the total accreted mass (in Msun)
        total_mass_msun = np.sum(acc_rate_msun_yr * dt / fp.YEAR)  # Convert Msun/yr to Msun

        return acc_rate_msun_yr, float(total_mass_msun)

    def plot_fluxes(self):
        """
        Plots the total flux as a function of time.
        """
        if self.flows is None:
            raise ValueError("Flows have not been calculated. Call calculate_fluxes() first.")

        # Convert time to physical units
        duration = (self.snapf - self.snapi + 1) * self.sim.UT / fp.YEAR
        times = self.time * duration
        start_time = self.snapi * self.sim.UT / fp.YEAR
        times += start_time

        average_flux = np.mean(self.flows)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.time,
            y=self.flows,
            mode='lines',
            name='Flux',
            line=dict(color='dodgerblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=self.time,
            y=[average_flux] * len(times),
            mode='lines',
            name=f'Avg: {average_flux:.2e} [Msun/yr]',
            line=dict(color='orangered', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f"Total Flux over Region (R={self.radius:.3f} [AU])",
            xaxis_title="Normalized Time",
            yaxis_title="Flux [Msun/yr]",
            template="plotly_white",
            font=dict(size=14),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True, exponentformat="e"),
        )
        fig.show()

    def plot_region(self, snapshot=1):
        """
        Plots the density map in 2D with the valid circular perimeter overlaid.

        :param snapshot: Snapshot to visualize.
        """
        # Load the density field for the snapshot
        gasdens = self.sim.load_field("gasdens", snapshot=snapshot, type="scalar")

        # Get the density slice and coordinates for the selected plane
        if self.plane == "XY":
            density_slice, mesh = gasdens.meshslice(slice=self.angle)
            x, y = mesh.x, mesh.y
        elif self.plane == "XZ":
            density_slice, mesh = gasdens.meshslice(slice=self.angle)
            x, y = mesh.x, mesh.z
        else:
            raise ValueError("Invalid plane. Choose 'XY' or 'XZ'.")

        # Plot the density map
        fig, ax = plt.subplots(figsize=(6, 6))
        c = ax.pcolormesh(
            x, y, np.log10(density_slice * self.sim.URHO),
            cmap="Spectral_r", shading="auto"
        )
        fig.colorbar(c, ax=ax, label=r"$\log_{10}(\rho)$ $[g/cm^3]$")

        # Add the circular perimeter
        circle = plt.Circle(
            self.center,  # Sphere center in the selected plane
            self.radius,  # Sphere radius
            color="red",
            fill=False,
            linestyle="--",
            linewidth=2
        )
        ax.add_artist(circle)  # Add the circle to the plot

        # Set plot labels and limits
        ax.set_xlabel(f"{self.plane[0]} [AU]")
        ax.set_ylabel(f"{self.plane[1]} [AU]")
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.legend()