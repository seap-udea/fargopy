import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tqdm import tqdm
import fargopy as fp
from fargopy.Fsimulation import DataHandler


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


class FluxAnalyzer:
    def __init__(self, output_dir, sphere_center=(0.0, 0.0, 0.0), radius=1.0, subdivisions=1, snapi=110, snapf=210):
        """
        Initializes the class with the simulation and sphere parameters.
        """
        self.sim = fp.Simulation(output_dir=output_dir)
        self.radius = radius
        self.data_handler = DataHandler(self.sim)
        self.data_handler.load_data(snapi=snapi, snapf=snapf)  # Load 3D data using the unified method
        self.sphere = SphereTessellation(radius=radius, subdivisions=subdivisions, center=sphere_center)
        self.sphere.tessellate()
        self.sphere_center = np.array(sphere_center)
        self.time = None
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

        for i, t in enumerate(tqdm(self.time, desc="Interpolating fields")):
            # Use interpolate_velocity for velocities
            velx, vely, velz = self.data_handler.interpolate_velocity(t, xc, yc, zc)
            self.velocities[i, :, 0] = velx
            self.velocities[i, :, 1] = vely
            self.velocities[i, :, 2] = velz

            # Interpolate density
            self.densities[i] = self.data_handler.interpolate_density(t, xc, yc, zc)

        return self.velocities, self.densities

    def calculate_normals(self):
        """Calculates the normal vectors of the sphere's triangles."""
        vertices = self.sphere.triangles
        self.normals = np.zeros((len(vertices), 3))

        for i, tri in enumerate(vertices):
            AB = tri[1] - tri[0]
            AC = tri[2] - tri[0]
            normal = np.cross(AB, AC)
            normal /= np.linalg.norm(normal)
            centroid = np.mean(tri, axis=0)
            to_centroid = centroid - self.sphere_center
            if np.dot(normal, to_centroid) < 0:
                normal = -normal
            self.normals[i] = normal

    def calculate_fluxes(self):
        """Calculates the total flux at each time step."""
        if self.normals is None:
            self.calculate_normals()

        self.flows = np.zeros(len(self.time))

        for i in range(len(self.time)):
            total_flux = np.sum(
                self.densities[i] * np.einsum('ij,ij->i', self.velocities[i], self.normals) * self.sphere.areas
            )
            self.flows[i] = (total_flux * self.sim.URHO * self.sim.UL**2 * self.sim.UV)*1e-3  # en Kg/s
        
        return(self.flows)
    
    def plot_fluxes(self):
        """Plots the total flux as a function of time."""
        if self.flows is None:
            raise ValueError("Flows have not been calculated. Call calculate_flows() first.")

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
            y=[average_flux] * len(self.time),
            mode='lines',
            name=f'Avg: {average_flux:.2e} kg/s',
            line=dict(color='orangered', width=2, dash='dash')
        ))
        fig.update_layout(
            title=f"Matter Flux over Planet-Centered Sphere (R={self.radius*self.sim.UL/fp.AU:.3f} [AU])",
            xaxis_title="Normalized Time",
            yaxis_title="Flux [kg/s]",
            template="plotly_white",
            font=dict(size=14),
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
        )
        fig.show()

    def planet_sphere(self, snapshot=1):
        """
        Plots the density map in both the XZ (phi=0) and XY (theta=1.56) planes for a given snapshot,
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
            np.log10(density_slice_xz * self.sim.URHO * 1e3),
            cmap="Spectral_r",
            shading="auto"
        )
        fig.colorbar(c1, ax=axes[0], label=r"$\log_{10}(\rho)$ [kg/m³]")
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
        axes[0].set_title("XZ Plane (phi=0)")
        axes[0].set_xlabel("X [AU]")
        axes[0].set_ylabel("Z [AU]")
        axes[0].set_xlim(x_xz.min(), x_xz.max())
        axes[0].set_ylim(z_xz.min(), z_xz.max())
        axes[0].legend()

        # Plot the density map for the XY plane
        c2 = axes[1].pcolormesh(
            x_xy,
            y_xy,
            np.log10(density_slice_xy * self.sim.URHO * 1e3),
            cmap="Spectral_r",
            shading="auto"
        )
        fig.colorbar(c2, ax=axes[1], label=r"$\log_{10}(\rho)$ [kg/m³]")
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
        axes[1].set_title("XY Plane (theta=1.56)")
        axes[1].set_xlabel("X [AU]")
        axes[1].set_ylabel("Y [AU]")
        axes[1].set_xlim(x_xy.min(), x_xy.max())
        axes[1].set_ylim(y_xy.min(), y_xy.max())
        axes[1].legend()

        
        plt.tight_layout()
        plt.show()