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


class FieldInterpolate:
    def __init__(self, sim):
        self.sim = sim
        self.df = None
        self.plane = None
        self.angle = None

    def load_data(self, field=None, plane=None, angle=None, snapshots=None):
        """
        Loads data in 2D or 3D depending on the provided parameters.

        Parameters:
            field (list of str, optional): List of fields to load (e.g., ["gasdens", "gasv"]).
            plane (str, optional): Plane for 2D data ('XZ', 'XY', 'YZ'). Required for 2D.
            angle (float, optional): Angle for the 2D slice. Required for 2D.
            snapshots (list or int, optional): List of snapshot indices or a single snapshot to load. Required for both 2D and 3D.
        Returns:
            pd.DataFrame: DataFrame containing the loaded data.
        """
        if field is None:
            raise ValueError("You must specify at least one field to load using the 'fields' parameter.")

        if (plane and not angle) or (angle and not plane):
            raise ValueError("Both 'plane' and 'angle' must be provided for 2D data.")

        if angle and not isinstance(angle, str):
            raise ValueError("'angle' must be a str example: angle='theta=1.5' [rad]")

        if not isinstance(snapshots, (int, list, tuple)):
            raise ValueError("'snapshots' must be an integer, a list, or a tuple.")

        if isinstance(snapshots, (list, tuple)) and len(snapshots) == 2:
            if snapshots[0] > snapshots[1]:
                raise ValueError("The range in 'snapshots' is invalid. The first value must be less than or equal to the second.")

        if not hasattr(self.sim, "domains") or self.sim.domains is None:
            raise ValueError("Simulation domains are not loaded. Ensure the simulation data is properly initialized.")

        # Convert a single snapshot to a list
        if isinstance(snapshots, int):
            snapshots = [snapshots]

        # Handle the case where snapshots is a single value or a list with one value
        if len(snapshots) == 1:
            snaps = snapshots
            time_values = [0]  # Single snapshot corresponds to a single time value
        else:
            snaps = np.arange(snapshots[0], snapshots[1] + 1)
            time_values = np.linspace(0, 1, len(snaps))

        if plane and angle:  # Load 2D data
        
            # Map plane to coordinate names
            plane_map = {
                "XY": ("x", "y", "vx", "vy"),
                "XZ": ("x", "z", "vx", "vz"),
                "YZ": ("y", "z", "vy", "vz")
            }

            if plane not in plane_map:
                raise ValueError(f"Invalid plane '{plane}'. Valid options are 'XY', 'XZ', 'YZ'.")

            coord1, coord2, vel1, vel2 = plane_map[plane]

            # Dynamically create DataFrame columns based on the fields
            columns = ["snapshot", "time", coord1, coord2]
            if field=="gasdens":
                print(f"Loading 2D density data for plane {plane} at angle {angle} rad.")
                columns.append("gasdens")
            if field=="gasv":
                columns.extend([vel1, vel2])
                print(f"Loading 2D gas velocity data for plane {plane} at angle {angle} rad.")
            
            if field=="gasenergy":
                columns.append("gasenergy")
                print(f"Loading 2D gas energy data for plane {plane} at angle {angle} rad.")
            df_snapshots = pd.DataFrame(columns=columns)

            for i, snap in enumerate(snaps):
                row = {"snapshot": snap, "time": time_values[i]}

                # Assign coordinates for all fields
                gasv = self.sim.load_field('gasv', snapshot=snap, type='vector')
                _, mesh = gasv.meshslice(slice=angle)
                coord1_vals, coord2_vals = getattr(mesh, coord1), getattr(mesh, coord2)
                row[coord1] = coord1_vals
                row[coord2] = coord2_vals

                if field=="gasdens" :
                    gasd = self.sim.load_field('gasdens', snapshot=snap, type='scalar')
                    gasd_slice, _ = gasd.meshslice(slice=angle)
                    row["gasdens"] = gasd_slice

                if field=="gasv":
                    gasvx, gasvy, gasvz = gasv.to_cartesian()
                    vel1_slice, _ = getattr(gasvx, f"meshslice")(slice=angle)
                    vel2_slice, _ = getattr(gasvy, f"meshslice")(slice=angle)
                    row[vel1] = vel1_slice
                    row[vel2] = vel2_slice

                if field=="gasenergy":
                    gasenergy = self.sim.load_field('gasenergy', snapshot=snap, type='scalar')
                    gasenergy_slice, _ = gasenergy.meshslice(slice=angle)
                    row["gasenergy"] = gasenergy_slice

                # Convert the row to a DataFrame and concatenate it
                row_df = pd.DataFrame([row])
                df_snapshots = pd.concat([df_snapshots, row_df], ignore_index=True)

            self.df = df_snapshots
            return df_snapshots
        
        elif plane is None and angle is None:  # Load 3D data
            print("Loading 3D data.")


            # Generate 3D mesh
            theta, r, phi = np.meshgrid(self.sim.domains.theta, self.sim.domains.r, self.sim.domains.phi, indexing='ij')
            x, y, z = r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi), r * np.cos(theta)

        
            # Create an empty DataFrame for the current field
            columns = ["snapshot", "time", "x", "y", "z"]
            if field == "gasv":
                columns.extend(["vx", "vy", "vz"])
            else:
                columns.append(field)

            df_snapshots = pd.DataFrame(columns=columns)

            for i, snap in enumerate(snaps):
                row = {"snapshot": snap, "time": time_values[i], "x": x.ravel(), "y": y.ravel(), "z": z.ravel()}

                if field == "gasdens":
                    gasd = self.sim.load_field("gasdens", snapshot=snap, type="scalar")
                    row[field] = gasd.data.ravel()

                elif field == "gasv":
                    gasv = self.sim.load_field("gasv", snapshot=snap, type="vector")
                    gasvx, gasvy, gasvz = gasv.to_cartesian()
                    row["vx"] = gasvx.data.ravel()
                    row["vy"] = gasvy.data.ravel()
                    row["vz"] = gasvz.data.ravel()

                elif field == "gasenergy":
                    gasenergy = self.sim.load_field("gasenergy", snapshot=snap, type="scalar")
                    row[field] = gasenergy.data.ravel()

                # Append the row to the DataFrame
                df_snapshots = pd.concat([df_snapshots, pd.DataFrame([row])], ignore_index=True)
            
            self.df = df_snapshots
            # Return the single DataFrame
            return df_snapshots
    

    def evaluate(self, time, var1, var2, var3=None):
        """
        Interpolates a field in 2D or 3D depending on the provided parameters.

        Parameters:
            time (float): Time at which to interpolate.
            var1, var2 (numpy.ndarray or float): Spatial coordinates in 2D.
            var3 (numpy.ndarray or float, optional): Additional coordinate for 3D. If None, 2D is assumed.

        Returns:
            numpy.ndarray or float: Interpolated field values at the given coordinates.
                                    If velocity fields are present, returns a tuple (vx, vy, vz) or (vx, vy).
        """
        # Automatically determine the field to interpolate
        if "gasdens" in self.df.columns:
            field_name = "gasdens"
        elif "gasenergy" in self.df.columns:
            field_name = "gasenergy"
        elif {"vx", "vy", "vz"}.issubset(self.df.columns):  # 3D velocity
            field_name = ["vx", "vy", "vz"]
        elif {"vx", "vy"}.issubset(self.df.columns):  # 2D velocity (vx, vy)
            field_name = ["vx", "vy"]
        elif {"vx", "vz"}.issubset(self.df.columns):  # 2D velocity (vx, vz)
            field_name = ["vx", "vz"]
        elif {"vy", "vz"}.issubset(self.df.columns):  # 2D velocity (vy, vz)
            field_name = ["vy", "vz"]
        else:
            raise ValueError("No valid field found in the DataFrame for interpolation.")

        # Sort the DataFrame by time
        df_sorted = self.df.sort_values("time")
        idx = df_sorted["time"].searchsorted(time) - 1
        if idx == -1:
            idx = 0
        idx_after = min(idx + 1, len(df_sorted) - 1)

        t0, t1 = df_sorted.iloc[idx]["time"], df_sorted.iloc[idx_after]["time"]
        factor = (time - t0) / (t1 - t0) if t1 > t0 else 0
        if factor < 0:
            factor = 0

        # Check if the input is a single point or a mesh
        is_scalar = np.isscalar(var1) and np.isscalar(var2) and (var3 is None or np.isscalar(var3))
        if is_scalar:
            result_shape = ()
        else:
            result_shape = var1.shape  # Preserve the shape of the input mesh

        def interp(idx, field):
            if var3 is not None:  # 3D interpolation
                coord_x = np.array(df_sorted.iloc[idx]["x"])
                coord_y = np.array(df_sorted.iloc[idx]["y"])
                coord_z = np.array(df_sorted.iloc[idx]["z"])
                points = np.column_stack((coord_x.ravel(), coord_y.ravel(), coord_z.ravel()))
                data = np.array(df_sorted.iloc[idx][field]).ravel()
                return griddata(points, data, (var1, var2, var3), method='nearest', fill_value=0.0)
            else:  # 2D interpolation
                if 'x' in self.df.columns and 'y' in self.df.columns:
                    coord1 = np.array(df_sorted.iloc[idx]["x"])
                    coord2 = np.array(df_sorted.iloc[idx]["y"])
                elif 'x' in self.df.columns and 'z' in self.df.columns:
                    coord1 = np.array(df_sorted.iloc[idx]["x"])
                    coord2 = np.array(df_sorted.iloc[idx]["z"])
                elif 'y' in self.df.columns and 'z' in self.df.columns:
                    coord1 = np.array(df_sorted.iloc[idx]["y"])
                    coord2 = np.array(df_sorted.iloc[idx]["z"])
                else:
                    raise ValueError("Insufficient spatial coordinates for interpolation.")
                points = np.column_stack((coord1.ravel(), coord2.ravel()))
                data = np.array(df_sorted.iloc[idx][field]).ravel()
                return griddata(points, data, (var1, var2), method='linear', fill_value=0.0)

        # Preallocate arrays for results
        if isinstance(field_name, list):  # Velocity (multiple fields)
            results = []
            for field in field_name:
                interpolated = (1 - factor) * interp(idx, field) + factor * interp(idx_after, field)
                if is_scalar:
                    results.append(interpolated.item())  # Extract scalar value
                else:
                    results.append(interpolated)
            return results
        else:  # Scalar field (gasdens or gasenergy)
            interpolated = (1 - factor) * interp(idx, field_name) + factor * interp(idx_after, field_name)
            if is_scalar:
                return interpolated.item()  # Extract scalar value
            else:
                return interpolated


        

class fluidmotion:
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
    

