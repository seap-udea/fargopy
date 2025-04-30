import pandas as pd
import numpy as np
import fargopy as fp

import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter

from ipywidgets import interact, FloatSlider, IntSlider
from celluloid import Camera
from IPython.display import HTML, Video

from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from tqdm import tqdm


class DataHandler:
    def __init__(self, sim):
        self.sim = sim
        self.df = None
        self.plane = None
        self.angle = None


    def load_data(self, plane=None, angle=None,snapi=1, snapf=50):
        """
        Loads data in 2D or 3D depending on the provided parameters.

        Parameters:
            plane (str, optional): Plane for 2D data ('XZ', 'XY', 'YZ'). Required for 2D.
            angle (float, optional): Angle for the 2D slice. Required for 2D.
            num_snapshots (int, optional): Number of snapshots for 2D. Required for 2D.
            snapi (int, optional): Initial snapshot for 3D. Required for 3D.
            snapf (int, optional): Final snapshot for 3D. Required for 3D.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        if plane and angle:  # Load 2D data
            print(f"Loading 2D data for plane {plane} at angle {angle} rad.")
            
            snapshots = np.arange(snapi, snapf + 1)
            num_snapshots = len(snapshots)
            time_values = np.linspace(0, 1, num_snapshots)

            df_snapshots = pd.DataFrame(columns=["snapshot", "time", "vel1", "vel2", "gasdens", "coord1", "coord2"])

            for i, snap in enumerate(snapshots):
                gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                gasvx, gasvy, gasvz = gasv.to_cartesian()
                gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')

                if plane == 'XZ':
                    vel1_slice, mesh = gasvx.meshslice(slice=angle)
                    vel2_slice, _ = gasvz.meshslice(slice=angle)
                    coord1, coord2 = mesh.x, mesh.z
                elif plane == 'XY':
                    vel1_slice, mesh = gasvx.meshslice(slice=angle)
                    vel2_slice, _ = gasvy.meshslice(slice=angle)
                    coord1, coord2 = mesh.x, mesh.y
                elif plane == 'YZ':
                    vel1_slice, mesh = gasvy.meshslice(slice='r=1.0')
                    vel2_slice, _ = gasvz.meshslice(slice='r=1.0')
                    coord1, coord2 = mesh.y, mesh.z

                gasd_slice, _ = gasd.meshslice(slice=angle)
                df_snapshots.loc[i] = [snap, time_values[i], vel1_slice, vel2_slice, gasd_slice, coord1, coord2]

            self.df = df_snapshots
            return df_snapshots

        elif plane==None and angle==None:  # Load 3D data
            print("Loading 3D data.")
            snapshots = np.arange(snapi, snapf + 1)
            num_snapshots = len(snapshots)
            time_values = np.linspace(0, 1, num_snapshots)

            df_snapshots = pd.DataFrame(columns=["snapshot", "time", "vel_x", "vel_y", "vel_z", "gasdens", "coord_x", "coord_y", "coord_z"])
            theta, r, phi = np.meshgrid(self.sim.domains.theta, self.sim.domains.r, self.sim.domains.phi, indexing='ij')
            x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)

            mesh = fp.Dictobj(dict=dict(r=r, phi=phi, theta=theta, x=x, y=y, z=z))

            for i, snap in enumerate(snapshots):
                gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                gasvx, gasvy, gasvz = gasv.to_cartesian()
                gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')

                # Get coordinates and field values
                coord_x, coord_y, coord_z = mesh.x, mesh.y, mesh.z
                vel_x, vel_y, vel_z = gasvx.data, gasvy.data, gasvz.data
                gasdens = gasd.data

                # Flatten arrays to store them in the DataFrame
                df_snapshots.loc[i] = [
                    snap,
                    time_values[i],
                    vel_x.ravel(),
                    vel_y.ravel(),
                    vel_z.ravel(),
                    gasdens.ravel(),
                    coord_x.ravel(),
                    coord_y.ravel(),
                    coord_z.ravel()
                ]

            self.df = df_snapshots
            return df_snapshots

        else:
            raise ValueError("Insufficient parameters to load data. Provide the necessary arguments for 2D or 3D.")

    def interpolate_field(self, time, var1, var2, var3=None, field_name=None):
        """
        Interpolates a field in 2D or 3D depending on the provided parameters.

        Parameters:
            time (float): Time at which to interpolate.
            var1, var2 (numpy.ndarray or float): Spatial coordinates in 2D.
            var3 (numpy.ndarray or float, optional): Additional coordinate for 3D. If None, 2D is assumed.
            field_name (str): Name of the field to interpolate (e.g., 'vel_x', 'gasdens').

        Returns:
            numpy.ndarray: Interpolated field values at the given coordinates.
        """
        df_sorted = self.df.sort_values("time")
        idx = df_sorted["time"].searchsorted(time) - 1
        if idx == -1:
            idx = 0
        idx_after = min(idx + 1, len(df_sorted) - 1)

        t0, t1 = df_sorted.iloc[idx]["time"], df_sorted.iloc[idx_after]["time"]
        factor = (time - t0) / (t1 - t0) if t1 > t0 else 0
        if factor < 0:
            factor = 0

        def interp(idx):
            if var3 is not None:  # 3D interpolation
                coord_x = df_sorted.iloc[idx]["coord_x"]
                coord_y = df_sorted.iloc[idx]["coord_y"]
                coord_z = df_sorted.iloc[idx]["coord_z"]
                points = np.column_stack((coord_x.ravel(), coord_y.ravel(), coord_z.ravel()))
                data = df_sorted.iloc[idx][field_name].ravel()
                return griddata(points, data, (var1, var2, var3), method='nearest', fill_value=0.0)
            else:  # 2D interpolation
                coord1 = df_sorted.iloc[idx]["coord1"]
                coord2 = df_sorted.iloc[idx]["coord2"]
                points = np.column_stack((coord1.ravel(), coord2.ravel()))
                data = df_sorted.iloc[idx][field_name].ravel()
                return griddata(points, data, (var1, var2), method='linear', fill_value=0.0)

        result = (1 - factor) * interp(idx) + factor * interp(idx_after)
        return result

    def interpolate_velocity(self, time, var1, var2, var3=None):
        """
        Interpolates the velocity field in 2D or 3D depending on the provided parameters.

        Parameters:
            time (float): Time at which to interpolate.
            var1, var2 (numpy.ndarray or float): Spatial coordinates in 2D.
            var3 (numpy.ndarray or float, optional): Additional coordinate for 3D. If None, 2D is assumed.

        Returns:
            tuple: Interpolated velocities (velx, vely[, velz]).
        """
        if var3 is not None:  # 3D interpolation
            velx = self.interpolate_field(time, var1, var2, var3, "vel_x")
            vely = self.interpolate_field(time, var1, var2, var3, "vel_y")
            velz = self.interpolate_field(time, var1, var2, var3, "vel_z")
            return velx, vely, velz
        else:  # 2D interpolation
            v1 = self.interpolate_field(time, var1, var2, field_name="vel1")
            v2 = self.interpolate_field(time, var1, var2, field_name="vel2")
            return v1, v2

    def interpolate_density(self, time, var1, var2, var3=None):
        """
        Interpolates the density field in 2D or 3D depending on the provided parameters.

        Parameters:
            time (float): Time at which to interpolate.
            var1, var2 (numpy.ndarray or float): Spatial coordinates in 2D.
            var3 (numpy.ndarray or float, optional): Additional coordinate for 3D. If None, 2D is assumed.

        Returns:
            numpy.ndarray: Interpolated density.
        """
        return self.interpolate_field(time, var1, var2, var3, "gasdens")
        

class VisualizationSimulation:
    def __init__(self, plane, angle, num_snapshots, dir_path):
        """
        Initializes the Simulation class.

        Parameters:
            plane (str): The plane for 2D data ('XZ', 'XY', 'YZ').
            angle (float): The angle for the 2D slice.
            num_snapshots (int): Number of snapshots for 2D data.
            dir_path (str): Directory path where the simulation data is stored.
        """
        self.sim = fp.Simulation(output_dir=dir_path)
        self.data_handler = DataHandler(self.sim)
        self.data_handler.load_data(plane, angle, num_snapshots)

    def velocity_field(self, t, y):
        """
        Computes the velocity field at a given time and position.

        Parameters:
            t (float): Time at which to compute the velocity field.
            y (array-like): Position [var1, var2] where the velocity is computed.

        Returns:
            numpy.ndarray: Velocity vector [v1, v2] at the given position and time.
        """
        var1, var2 = y
        v1, v2 = self.data_handler.interpolate_velocity(t, np.array([var1]), np.array([var2]))
        return np.array([v1[0], v2[0]])

    def integrate_particles(self, particle_pos, time, dt=0.01):
        """
        Integrates all particles using an explicit Euler step.

        Parameters:
            particle_pos (numpy.ndarray): Array of particle positions (shape: [n_particles, 2]).
            time (float): Current time of the simulation.
            dt (float): Time step for integration.

        Returns:
            numpy.ndarray: Updated particle positions after integration.
        """
        if len(particle_pos) == 0:
            return np.array([])

        v1, v2 = self.data_handler.interpolate_velocity(time, particle_pos[:, 0], particle_pos[:, 1])

        # Euler step: x_{n+1} = x_n + v * dt
        particle_pos[:, 0] += v1 * dt
        particle_pos[:, 1] += v2 * dt

        return particle_pos

    def generate_uniform_particles(self, var1_min, var1_max, var2_min, var2_max, num_particles):
        """
        Generates uniformly distributed particles within a specified region.

        Parameters:
            var1_min, var1_max (float): Range for the first coordinate.
            var2_min, var2_max (float): Range for the second coordinate.
            num_particles (int): Number of particles to generate.

        Returns:
            numpy.ndarray: Array of particle positions (shape: [num_particles, 2]).
        """
        grid_size = int(np.sqrt(num_particles))
        var1_candidates = np.linspace(var1_min + 0.01, var1_max - 0.01, grid_size)
        var2_candidates = np.linspace(var2_min + 0.001, var2_max - 0.001, grid_size)
        VAR1_grid, VAR2_grid = np.meshgrid(var1_candidates, var2_candidates, indexing='ij')

        density_values = self.data_handler.interpolate_density(0, VAR1_grid, VAR2_grid)
        valid_mask = density_values > 0

        valid_var1 = VAR1_grid[valid_mask]
        valid_var2 = VAR2_grid[valid_mask]

        if len(valid_var1) == 0:
            return []

        num_valid_points = min(num_particles, len(valid_var1))
        new_particles = np.column_stack((valid_var1[:num_valid_points], valid_var2[:num_valid_points]))

        return new_particles

    def run_simulation(self, res, var1_min, var1_max, var2_min, var2_max, ts, npi, max_lifetime, generation_interval):
        """
        Runs the particle simulation and generates an animation.

        Parameters:
            res (int): Resolution of the grid for density interpolation.
            var1_min, var1_max (float): Range for the first coordinate.
            var2_min, var2_max (float): Range for the second coordinate.
            ts (int): Number of time steps for the simulation.
            npi (int): Number of particles to generate at each interval.
            max_lifetime (int): Maximum lifetime of particles.
            generation_interval (int): Interval for generating new particles.

        Returns:
            IPython.display.Video: Animation of the particle simulation.
        """
        var1_reg, var2_reg = np.linspace(var1_min, var1_max, res), np.linspace(var2_min, var2_max, res)
        VAR1_reg, VAR2_reg = np.meshgrid(var1_reg, var2_reg, indexing='ij')

        t_span = (0, 1)
        t_eval = np.linspace(t_span[0], t_span[1], ts)

        particle_pos = np.empty((0, 2))
        lifetimes = np.empty(0)
        new_particles = self.generate_uniform_particles(var1_min, var1_max, var2_min, var2_max, npi)

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        fig, ax = plt.subplots(figsize=(8, 8))
        camera = Camera(fig)

        with tqdm(total=len(t_eval), desc="Generating animation", unit="frame") as pbar:
            for frame in range(len(t_eval)):
                time = t_eval[frame]
                if frame % generation_interval == 0:
                    particle_pos = np.vstack([particle_pos, new_particles])
                    lifetimes = np.concatenate([lifetimes, np.full(len(new_particles), max_lifetime)])

                updated_pos = self.integrate_particles(particle_pos, time, dt=0.01)
                updated_pos = np.array([pos for pos in updated_pos if pos is not None])
                updated_lifetimes = lifetimes - 1

                valid_indices = updated_lifetimes > 0
                particle_pos = updated_pos[valid_indices]
                lifetimes = updated_lifetimes[valid_indices]

                lifetimes_normalized = lifetimes / max_lifetime

                # Add density interpolation as background
                gasd_interpolated = self.data_handler.interpolate_density(time, VAR1_reg, VAR2_reg)
                c = ax.pcolormesh(VAR1_reg, VAR2_reg, np.log10(gasd_interpolated * self.sim.URHO * 1e3),
                                  cmap="viridis", shading='auto')

                # Plot particles
                if len(particle_pos) > 0:
                    ax.scatter(particle_pos[:, 0], particle_pos[:, 1], c='lightgray', alpha=lifetimes_normalized, s=1.0)

                ax.set_xlim(var1_min, var1_max)
                ax.set_ylim(var2_min, var2_max)
                ax.set_xlabel(r"$r \ [AU]$", size=12)
                ax.set_ylabel(y_label, size=12)
                camera.snap()

                pbar.update(1)

        # Add color bar
        fig.colorbar(c, ax=ax, label=r'$\log_{10}(\rho)$ [kg/m$^3$]')
        plt.close(fig)
        animation = camera.animate()
        video_filename = 'particles.mp4'
        animation.save(video_filename, writer=FFMpegWriter(fps=10, codec='libx264', bitrate=5000))

        # Display the video in the interactive environment
        return Video(video_filename, embed=True)
    

class Visualize:
    def __init__(self, data_handler):
        """
        Initializes the Visualize class with an instance of DataHandler.
        """
        self.data_handler = data_handler

    def density(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Plots a contour map of interpolated density at a given time.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolate density at the given time
        density = self.data_handler.interpolate_density(time, VAR1, VAR2).T

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Create the contour plot
        fig = go.Figure(
            data=go.Contour(
                z=np.log10(density * self.data_handler.sim.URHO * 1e3),
                x=var1,
                y=var2,
                colorscale="Spectral_r",
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="log10(ρ) [kg/m³]")
            )
        )

        # Configure the plot layout
        fig.update_layout(
            title=f"Density Contour Map (t = {time:.2f})",
            xaxis_title="r [AU]",
            yaxis_title=y_label,
            width=600,
            height=600
        )

        # Show the plot
        fig.show()

    def animate_density(self, var1_min, var1_max, var2_min, var2_max, res, time_array):
        """
        Creates an animation of the density field evolution over time.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Precompute the density field for all times
        precalculated_density = [
            self.data_handler.interpolate_density(time, VAR1, VAR2).T
            for time in time_array
        ]

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Create the initial figure
        initial_density = precalculated_density[0]
        fig = go.Figure(
            data=go.Contour(
                z=np.log10(initial_density * self.data_handler.sim.URHO * 1e3),
                x=var1,
                y=var2,
                colorscale="Spectral_r",
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="log10(ρ) [kg/m³]")
            )
        )

        # Create frames for the animation
        frames = []
        for i, time in enumerate(time_array):
            density = precalculated_density[i]
            frames.append(go.Frame(
                data=go.Contour(
                    z=np.log10(density * self.data_handler.sim.URHO * 1e3),
                    x=var1,
                    y=var2,
                    colorscale="Spectral_r",
                    contours=dict(coloring="heatmap")
                ),
                name=f"{time:.2f}"
            ))

        fig.frames = frames

        # Configure the slider
        sliders = [
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f"{time:.2f}"], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
                        label=f"{time:.2f}"
                    )
                    for time in time_array
                ],
                transition=dict(duration=0),
                currentvalue=dict(font=dict(size=16), prefix="Time: ", visible=True),
                len=0.9
            )
        ]

        # Configure play and pause buttons
        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )
        ]

        # Update the figure layout
        fig.update_layout(
            title="Density Field Evolution",
            xaxis_title='r [AU]',
            yaxis_title=y_label,
            width=600,
            height=600,
            sliders=sliders,
            updatemenus=updatemenus
        )

        # Show the animation
        fig.show()

    def velocity(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Plots a map of interpolated velocity magnitude at a given time.
        Areas where the interpolation gives 0 will not be colored.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolate the velocity field at the given time
        velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)

        # Calculate the velocity magnitude
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convert to km/s

        # Set areas where velocity_x and velocity_y are 0 to None
        velocity_magnitude[velocity_magnitude == 0] = None

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Create the contour plot
        fig = go.Figure(
            data=go.Contour(
                z=velocity_magnitude,
                x=var1,
                y=var2,
                colorscale="Viridis",
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="|v| [km/s]"),
                zmin=np.nanmin(velocity_magnitude),
                zmax=np.nanmax(velocity_magnitude),
                showscale=True
            )
        )

        # Configure the plot layout
        fig.update_layout(
            title=f"Velocity Map (t = {time:.2f})",
            xaxis_title="r [AU]",
            yaxis_title=y_label,
            width=600,
            height=600,
            plot_bgcolor="white"  # White background to make uncolored areas visible
        )

        # Show the plot
        fig.show()


    def animate_velocity(self, var1_min, var1_max, var2_min, var2_max, res, time_array):
        """
        Creates an animation of the interpolated velocity magnitude map over time.
        Areas where the velocity magnitude is 0 will not be colored.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Precompute the velocity magnitude field for all times
        precalculated_velocity_magnitude = []
        for time in time_array:
            velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convert to km/s
            velocity_magnitude[velocity_magnitude == 0] = None  # Replace areas with magnitude 0 with None
            precalculated_velocity_magnitude.append(velocity_magnitude)

        # Create the initial figure
        initial_velocity_magnitude = precalculated_velocity_magnitude[0]
        fig = go.Figure(
            data=go.Contour(
                z=initial_velocity_magnitude,
                x=var1,
                y=var2,
                colorscale="Viridis",
                contours=dict(coloring="heatmap"),
                colorbar=dict(title="|v| [km/s]")
            )
        )

        # Create frames for the animation
        frames = []
        for i, time in enumerate(time_array):
            velocity_magnitude = precalculated_velocity_magnitude[i]
            frames.append(go.Frame(
                data=go.Contour(
                    z=velocity_magnitude,
                    x=var1,
                    y=var2,
                    colorscale="Viridis",
                    contours=dict(coloring="heatmap")
                ),
                name=f"{time:.2f}"
            ))

        fig.frames = frames

        # Configure the slider
        sliders = [
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[f"{time:.2f}"], dict(mode="immediate", frame=dict(duration=150, redraw=True), transition=dict(duration=0))],
                        label=f"{time:.2f}"
                    )
                    for time in time_array
                ],
                transition=dict(duration=0),
                currentvalue=dict(font=dict(size=16), prefix="Time: ", visible=True),
                len=0.9
            )
        ]

        # Configure play and pause buttons
        updatemenus = [
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
                ]
            )
        ]

        # Update the figure layout
        fig.update_layout(
            title="Velocity Field Evolution",
            xaxis_title='r [AU]',
            yaxis_title=y_label,
            width=600,
            height=600,
            sliders=sliders,
            updatemenus=updatemenus
        )

        # Show the animation
        fig.show()

    def vel_streamlines(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Plots the streamlines of the velocity field with density as the background for a given time.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolate the density and velocity fields at the given time
        density = self.data_handler.interpolate_density(time, VAR1, VAR2)
        velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)

        # Calculate the velocity magnitude
        v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convert to km/s

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Create the plot
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
        plt.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0, cmap='viridis')
        plt.colorbar(label="|v| [km/s]")
        plt.title(f"Streamlines (t = {time:.2f})")
        plt.xlabel("r [AU]")
        plt.ylabel(y_label)
        plt.show()  

    def vel_streamlines_slide(self, var1_min, var1_max, var2_min, var2_max, res, time_array):
        """
        Creates an interactive slider to visualize the streamlines of the velocity field
        with density as the background for different time steps.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Precompute density and velocity fields for all time steps
        precalculated_data = [
            {
                "density": self.data_handler.interpolate_density(time, VAR1, VAR2),
                "velocity": self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            }
            for time in time_array
        ]

        def update_plot(time_index):
            """
            Updates the plot for a given time index.
            """
            data = precalculated_data[time_index]
            density = data["density"]
            velocity_x = data["velocity"][0]
            velocity_y = data["velocity"][1]
            v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convert to km/s

            # Create the plot
            plt.figure(figsize=(6, 6))
            plt.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
            plt.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0, cmap='viridis')
            plt.colorbar(label="|v| [km/s]")
            plt.title(f"Streamlines (t = {time_array[time_index]:.2f})")
            plt.xlabel("r [AU]")
            plt.ylabel(y_label)
            plt.show()

        # Create an interactive slider for time (with integer indices)
        interact(update_plot, time_index=IntSlider(value=0, min=0, max=len(time_array) - 1, step=1))

    def vel_streamlines_vid(self, var1_min, var1_max, var2_min, var2_max, res, time_array, output_file="streamlines_animation.mp4"):
        """
        Creates a video animation of the streamlines of the velocity field
        with density as the background for different time steps.

        Parameters:
            var1_min, var1_max (float): Range for the first coordinate.
            var2_min, var2_max (float): Range for the second coordinate.
            res (int): Resolution of the grid.
            time_array (list): Array of time steps for the animation.
            output_file (str): Name of the output video file.
        """
        # Create a regular grid for the coordinates
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Determine the y-axis label based on the plane
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else "Y [AU]"

        # Configure the figure and camera for the animation
        fig, ax = plt.subplots(figsize=(6, 6))
        camera = Camera(fig)

        # Generate frames for the animation
        for time in time_array:
            # Interpolate the density and velocity fields
            density = self.data_handler.interpolate_density(time, VAR1, VAR2)
            velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convert to km/s

            # Create the plot for the current frame
            c = ax.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
            strm = ax.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0, cmap='viridis')
            ax.set_xlabel("r [AU]")
            ax.set_ylabel(y_label)
            camera.snap()

        # Create the animation
        fig.colorbar(strm.lines, ax=ax, label="|v| [km/s]")
        plt.close(fig)
        animation = camera.animate()

        # Save the animation to a file
        animation.save(output_file, writer="ffmpeg", fps=10)

        plt.close(fig)
        return Video(output_file, embed=True)
