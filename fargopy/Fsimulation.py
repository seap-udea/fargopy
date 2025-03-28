import pandas as pd
import numpy as np
import fargopy as fp


import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from matplotlib.animation import FFMpegWriter

from ipywidgets import interact, FloatSlider,IntSlider 
from celluloid import Camera
from IPython.display import HTML
from IPython.display import Video

from scipy.interpolate import griddata
from scipy.integrate import solve_ivp
from tqdm import tqdm




class DataHandler:
    def __init__(self, sim):
        self.sim = sim
        self.df = None
        self.plane = None

    def load_data(self, plane, angle, num_snapshots):
        self.plane = plane 
        snapshots = np.arange(1, num_snapshots + 1)
        time_values = snapshots / num_snapshots

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

    def interpolate_field(self, time, var1, var2, field_name):
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
            coord1, coord2 = df_sorted.iloc[idx]["coord1"], df_sorted.iloc[idx]["coord2"]
            points = np.column_stack((coord1.ravel(), coord2.ravel()))
            data = df_sorted.iloc[idx][field_name].ravel()
            return griddata(points, data, (var1, var2), method='linear', fill_value=0.0)

        result = (1 - factor) * interp(idx) + factor * interp(idx_after)
        return result

    def interpolate_velocity(self, time, var1, var2):
        v1 = self.interpolate_field(time, var1, var2, "vel1")
        v2 = self.interpolate_field(time, var1, var2, "vel2")
        return v1, v2

    def interpolate_density(self, time, var1, var2):
        return self.interpolate_field(time, var1, var2, "gasdens")
        

class Simulation:
    def __init__(self, plane, angle, num_snapshots, dir_path):
        self.sim = fp.Simulation(output_dir=dir_path)
        self.data_handler = DataHandler(self.sim)
        self.data_handler.load_data(plane, angle, num_snapshots)

    def velocity_field(self, t, y):
        var1, var2 = y
        v1, v2 = self.data_handler.interpolate_velocity(t, np.array([var1]), np.array([var2]))
        return np.array([v1[0], v2[0]])

    def integrate_particles(self, particle_pos, time, dt=0.01):
        """ Integra todas las partículas con un paso explícito de Euler."""
        if len(particle_pos) == 0:
            return np.array([])

        v1, v2 = self.data_handler.interpolate_velocity(time, particle_pos[:, 0], particle_pos[:, 1])

        # Paso de Euler: x_{n+1} = x_n + v * dt
        particle_pos[:, 0] += v1 * dt
        particle_pos[:, 1] += v2 * dt

        return particle_pos

    def generate_uniform_particles(self, var1_min, var1_max, var2_min, var2_max, num_particles):
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
        var1_reg, var2_reg = np.linspace(var1_min, var1_max, res), np.linspace(var2_min, var2_max, res)
        VAR1_reg, VAR2_reg = np.meshgrid(var1_reg, var2_reg, indexing='ij')

        t_span = (0, 1)
        t_eval = np.linspace(t_span[0], t_span[1], ts)

        particle_pos = np.empty((0, 2))
        lifetimes = np.empty(0)
        new_particles = self.generate_uniform_particles(var1_min, var1_max, var2_min, var2_max, npi)

            # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        fig, ax = plt.subplots(figsize=(8, 8))
        camera = Camera(fig)

        with tqdm(total=len(t_eval), desc="Generando animación", unit="frame") as pbar:
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

                # Agregar interpolación de densidad como fondo
                gasd_interpolated = self.data_handler.interpolate_density(time, VAR1_reg, VAR2_reg)
                c = ax.pcolormesh(VAR1_reg, VAR2_reg, np.log10(gasd_interpolated * self.sim.URHO * 1e3),
                                cmap="viridis", shading='auto')

                # Graficar partículas
                if len(particle_pos) > 0:
                    ax.scatter(particle_pos[:, 0], particle_pos[:, 1], c='lightgray', alpha=lifetimes_normalized, s=1.0)

                ax.set_xlim(var1_min, var1_max)
                ax.set_ylim(var2_min, var2_max)
                ax.set_xlabel(r"$r \ [AU]$",size=12)
                ax.set_ylabel(y_label,size=12)
                camera.snap()

                pbar.update(1)

        # Agregar barra de color
        fig.colorbar(c, ax=ax, label=r'$\log_{10}(\rho)$ [kg/m$^3$]')
        plt.close(fig)
        animation = camera.animate()
        video_filename = 'figures/streaklines.mp4'
        animation.save(video_filename, writer=FFMpegWriter(fps=10, codec='libx264', bitrate=5000))

        # Mostrar el video en el entorno interactivo
        return Video(video_filename, embed=True)
    

class Visualize:
    def __init__(self, data_handler):
        """
        Inicializa la clase Visualize con una instancia de DataHandler.
        """
        self.data_handler = data_handler

    def density(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Grafica un mapa de contornos de densidad interpolada en un tiempo dado.
        """
        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolar la densidad en el tiempo dado
        density = self.data_handler.interpolate_density(time, VAR1, VAR2).T

        # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Crear el gráfico de contorno
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

        # Configurar el diseño del gráfico
        fig.update_layout(
            title=f"Mapa de Contornos de Densidad (t = {time:.2f})",
            xaxis_title="r [AU]",
            yaxis_title=y_label,
            width=600,
            height=600
        )

        # Mostrar el gráfico
        fig.show()

        


    def animate_density(self, var1_min, var1_max, var2_min, var2_max, res, time_array):

        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Precalcular el campo de densidad para todos los tiempos
        precalculated_density = [
            self.data_handler.interpolate_density(time, VAR1, VAR2).T
            for time in time_array
        ]
            # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Crear la figura inicial
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

        # Crear los frames para la animación
        frames = []
        for i, time in enumerate(time_array):
            density = precalculated_density[i]
            frames.append(go.Frame(
                data=go.Contour(
                    z=np.log10(density * self.data_handler.sim.URHO * 1e3),
                    x=var1,
                    y=var2,
                    colorscale="Spectral_r",  # Este colormap será actualizado dinámicamente
                    contours=dict(coloring="heatmap")
                ),
                name=f"{time:.2f}"
            ))

        fig.frames = frames

        # Configurar el slider
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
                currentvalue=dict(font=dict(size=16), prefix="Tiempo: ", visible=True),
                len=0.9
            )
        ]

        # Configurar los botones de reproducción y pausa
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

        # Actualizar el diseño de la figura
        fig.update_layout(
            title="Evolución del Campo de Densidad",
            xaxis_title='r [AU]',
            yaxis_title=y_label,
            width=600,
            height=600,
            sliders=sliders,
            updatemenus=updatemenus
        )

        # Mostrar la animación
        fig.show()

    def velocity(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Grafica un mapa de magnitud de velocidad interpolada en un tiempo dado.
        Las zonas donde la interpolación da 0 no serán coloreadas.
        """
        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolar el campo de velocidad en el tiempo dado
        velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)

        # Calcular la magnitud de la velocidad
        velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convertir a km/s

        # Establecer las zonas donde velocity_x y velocity_y son 0 como None
        velocity_magnitude[velocity_magnitude==0] = None

        # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Crear el gráfico de contorno
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

        # Configurar el diseño del gráfico
        fig.update_layout(
            title=f"Mapa de Velocidad (t = {time:.2f})",
            xaxis_title="r [AU]",
            yaxis_title=y_label,
            width=600,
            height=600,
            plot_bgcolor="white"  # Fondo blanco para que las zonas no coloreadas sean visibles
        )

        # Mostrar el gráfico
        fig.show()

    def animate_velocity(self, var1_min, var1_max, var2_min, var2_max, res, time_array):
        """
        Crea una animación del mapa de magnitud de velocidad interpolada en función del tiempo.
        Las zonas donde la magnitud de la velocidad es 0 no serán coloreadas.
        """
        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Precalcular la magnitud del campo de velocidad para todos los tiempos
        precalculated_velocity_magnitude = []
        for time in time_array:
            velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            velocity_magnitude = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convertir a km/s
            velocity_magnitude[velocity_magnitude == 0] = None  # Reemplazar las zonas con magnitud 0 por None
            precalculated_velocity_magnitude.append(velocity_magnitude)

        # Crear la figura inicial
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

        # Crear los frames para la animación
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

        # Configurar el slider
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
                currentvalue=dict(font=dict(size=16), prefix="Tiempo: ", visible=True),
                len=0.9
            )
        ]

        # Configurar los botones de reproducción y pausa
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

        # Actualizar el diseño de la figura
        fig.update_layout(
            title="Evolución del Campo de Velocidad",
            xaxis_title='r [AU]',
            yaxis_title=y_label,
            width=600,
            height=600,
            sliders=sliders,
            updatemenus=updatemenus
        )

        # Mostrar la animación
        fig.show()

    def vel_streamlines(self, var1_min, var1_max, var2_min, var2_max, res, time):
        """
        Grafica las streamlines del campo de velocidad con la densidad de fondo para un tiempo dado.
        """
        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')

        # Interpolar el campo de densidad y velocidad en el tiempo dado
        density = self.data_handler.interpolate_density(time, VAR1, VAR2)
        velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)

        # Calcular la magnitud de la velocidad
        v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T * self.data_handler.sim.UV / (1e5)  # Convertir a km/s

        # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Crear el gráfico
        plt.figure(figsize=(6, 6))
        plt.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
        plt.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0, cmap='viridis')
        plt.colorbar(label="|v| [km/s]")
        plt.title(f"Streamlines (t = {time:.2f})")
        plt.xlabel("r [AU]")
        plt.ylabel(y_label)
        plt.show()          

    def vel_streamlines_slide(self, var1_min, var1_max, var2_min, var2_max, res, time_array):

        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')
            # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Precalcular densidad y velocidades para todos los tiempos
        precalculated_data = [
            {
                "density": self.data_handler.interpolate_density(time, VAR1, VAR2),
                "velocity": self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            }
            for time in time_array
        ]

        def update_plot(time_index):
            """
            Actualiza el gráfico para un índice de tiempo dado.
            """
            data = precalculated_data[time_index]
            density = data["density"]
            velocity_x = data["velocity"][0]
            velocity_y = data["velocity"][1]
            v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T*self.data_handler.sim.UV/(1e5)  
            # Crear el gráfico
            plt.figure(figsize=(6, 6))
            plt.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
            plt.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0,cmap='viridis')
            plt.colorbar(label="|v| [km/s]")
            plt.title(f"Streamlines (t = {time_array[time_index]:.2f})")
            plt.xlabel("r [AU]")
            plt.ylabel(y_label)
            plt.show()

        # Crear un slider interactivo para el tiempo (con índices enteros)
        interact(update_plot, time_index=IntSlider(value=0, min=0, max=len(time_array) - 1, step=1))

        
    def vel_streamlines_vid(self, var1_min, var1_max, var2_min, var2_max, res, time_array,output_file="streamlines_animation.mp4"):

        # Crear una cuadrícula regular para las coordenadas
        var1 = np.linspace(var1_min, var1_max, res)
        var2 = np.linspace(var2_min, var2_max, res)
        VAR1, VAR2 = np.meshgrid(var1, var2, indexing='ij')
            # Determinar el label del eje y en función del plano
        plane = self.data_handler.plane
        y_label = "Z [AU]" if plane == "XZ" else r"$\phi$ [rad]"

        # Configurar la figura y la cámara para la animación
        fig, ax = plt.subplots(figsize=(6, 6))
        camera = Camera(fig)

        # Generar los frames de la animación
        for time in time_array:
            # Interpolar el campo de densidad y velocidad
            density = self.data_handler.interpolate_density(time, VAR1, VAR2)
            velocity_x, velocity_y = self.data_handler.interpolate_velocity(time, VAR1, VAR2)
            v_mag = np.sqrt(velocity_x**2 + velocity_y**2).T*self.data_handler.sim.UV/(1e5) 
            
            # Crear el gráfico para el frame actual
            c = ax.pcolormesh(var1, var2, np.log10(density * self.data_handler.sim.URHO * 1e3).T, cmap="Spectral_r", shading='auto')
            strm=ax.streamplot(var1, var2, velocity_x.T, velocity_y.T, color=v_mag, linewidth=0.7, density=3.0,cmap='viridis')
            ax.set_xlabel("r [AU]")
            ax.set_ylabel(y_label)
            camera.snap()
            

        # Crear la animación
        fig.colorbar(strm.lines, ax=ax, label="|v| [km/s]")
        plt.close(fig)
        animation = camera.animate()


        # Guardar la animación en un archivo
        animation.save(output_file, writer="ffmpeg", fps=10)

        plt.close(fig)
        return Video(output_file, embed=True)
    
