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


class SphereTessellation:
    def __init__(self, radius=1.0, subdivisions=0, center=(0.0, 0.0, 0.0)):
        """
        Initializes the sphere tessellation.

        :radius: Radius of the sphere.
        :subdivisions: Number of subdivisions for the tessellation.
        :center: Coordinates of the sphere's center (x, y, z).
        """
        self.radius = radius
        self.subdivisions = subdivisions
        self.center = np.array(center)  # Sphere center
        self.num_triangles = 20 * (4 ** subdivisions)  # Total number of triangles
        self.triangles = np.zeros((self.num_triangles, 3, 3))  # Array for triangles
        self.centers = np.zeros((self.num_triangles, 3))  # Array for centers
        self.areas = np.zeros(self.num_triangles)  # Array for areas
        self.triangle_index = 0  # Index to fill the triangle array

    @staticmethod
    def normalize(v):
        """Normalizes a vector to have magnitude 1."""
        return v / np.linalg.norm(v)

    def subdivide_triangle(self, v1, v2, v3, depth):
        """Subdivides a triangle into 4 smaller triangles."""
        if depth == 0:
            # Shift the vertices according to the sphere's center
            self.triangles[self.triangle_index] = [v1 + self.center, v2 + self.center, v3 + self.center]
            self.triangle_index += 1
            return

        # Calculate the midpoints of the triangle's edges
        v12 = self.normalize((v1 + v2) / 2) * self.radius
        v23 = self.normalize((v2 + v3) / 2) * self.radius
        v31 = self.normalize((v3 + v1) / 2) * self.radius

        # Recursively subdivide
        self.subdivide_triangle(v1, v12, v31, depth - 1)
        self.subdivide_triangle(v12, v2, v23, depth - 1)
        self.subdivide_triangle(v31, v23, v3, depth - 1)
        self.subdivide_triangle(v12, v23, v31, depth - 1)

    def generate_icosphere(self):
        """Generates a tessellated sphere based on an icosahedron."""
        phi = (1.0 + np.sqrt(5.0)) / 2.0

        # Base patterns for vertex coordinates
        patterns = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ]

        # Generate and normalize vertices
        vertices = np.array([self.normalize(np.array(p)) * self.radius for p in patterns])

        # Define the faces of the icosahedron
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]

        # Subdivide each triangle
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            self.subdivide_triangle(v1, v2, v3, self.subdivisions)

    def calculate_polygon_centers(self):
        """Calculates the center of each triangle."""
        self.centers = np.mean(self.triangles, axis=1)

    @staticmethod
    def calculate_triangle_area(v1, v2, v3):
        """Calculates the area of a triangle given three vertices in 3D."""
        # Calculate the vectors of the triangle's edges
        side1 = v2 - v1
        side2 = v3 - v1

        # Calculate the cross product of the edges
        cross_product = np.cross(side1, side2)

        # The area is half the magnitude of the cross product
        area = np.linalg.norm(cross_product) / 2
        return area

    def calculate_all_triangle_areas(self):
        """Calculates the area of all triangles in the tessellation."""
        for i, (v1, v2, v3) in enumerate(self.triangles):
            self.areas[i] = self.calculate_triangle_area(v1, v2, v3)

    def tessellate(self):
        """Performs the complete tessellation of the sphere."""
        self.generate_icosphere()
        self.calculate_polygon_centers()
        self.calculate_all_triangle_areas()

    def plot_icosphere(self):
        """
        Visualizes the tessellated sphere and the centers of the triangles using Plotly.
        """
        x, y, z = [], [], []
        i, j, k = [], [], []

        # Add vertices and faces
        vertex_index = 0
        for v1, v2, v3 in self.triangles:
            x.extend([v1[0], v2[0], v3[0]])
            y.extend([v1[1], v2[1], v3[1]])
            z.extend([v1[2], v2[2], v3[2]])
            i.append(vertex_index)
            j.append(vertex_index + 1)
            k.append(vertex_index + 2)
            vertex_index += 3

        # Create the triangular mesh
        mesh = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color='lightblue',
            opacity=0.3
        )

        # Add centers as points
        scatter = go.Scatter3d(
            x=self.centers[:, 0],
            y=self.centers[:, 1],
            z=self.centers[:, 2],
            mode='markers',
            marker=dict(size=1.0, color='red'),
            name='Centers'
        )

        # Create the figure
        fig = go.Figure(data=[mesh, scatter])
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=True),
                yaxis=dict(visible=True),
                zaxis=dict(visible=True),
            ),
            template="plotly_white"
        )
        fig.show()

    def generate_dataframe(self):
        """
        Generates a DataFrame with the coordinates of the triangles, their centers, and their areas.

        :return: DataFrame with columns for triangles, centers, and areas.
        """
        data = []
        for i, (triangle, center, area) in enumerate(zip(self.triangles, self.centers, self.areas)):
            data.append({
                "Triangle": triangle.tolist(),  # Convert to list for DataFrame compatibility
                "Center": center.tolist(),
                "Area": area
            })

        # Create the DataFrame
        df = pd.DataFrame(data)
        return df


class FluxAnalyzer3D:

    def __init__(self, output_dir, sphere_center=(0.0, 0.0, 0.0), radius=1.0, subdivisions=1, snapi=110, snapf=210):
        """
        Initializes the class with the simulation and sphere parameters.
        """
        self.sim = fp.Simulation(output_dir=output_dir)
        self.radius = radius
        self.data_handler = fargopy.DataHandler(self.sim)
        self.data_handler.load_data(snapi=snapi, snapf=snapf)  # Load 3D data using the unified method
        self.sphere = SphereTessellation(radius=radius, subdivisions=subdivisions, center=sphere_center)
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