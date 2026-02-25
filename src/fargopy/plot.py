###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import matplotlib.pyplot as plt
import numpy as np

###############################################################
# Constants
###############################################################


###############################################################
# Classes
###############################################################
class Plot(object):
    """Plotting utilities and visualization helpers for FARGO3D data.

    The ``Plot`` class encapsulates static methods for common plotting tasks,
    such as adding watermarks to figures and creating standardized heatmaps
    for simulation fields.
    """

    @staticmethod
    def fargopy_mark(ax, frac=1/6, alpha=0.5):
        """Add a watermark to a 2D or 3D plot.

        Places a rotated "FARGOpy {version}" watermark in the top-right corner
        of the specified axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object where the watermark will be added.

        Returns
        -------
        matplotlib.text.Text
            The created text object.

        Examples
        --------
        Add watermark to a plot:

        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3], [1, 2, 3])
        >>> fp.Plot.fargopy_mark(ax)
        """
        # Get the height of axe
        axh = (
            ax.get_window_extent()
            .transformed(ax.get_figure().dpi_scale_trans.inverted())
            .height
        )
        fig_factor = frac * axh

        # Options of the water mark
        args = dict(
            rotation=270,
            ha="left",
            va="top",
            transform=ax.transAxes,
            color="pink",
            alpha=alpha,
            fontsize=10 * fig_factor,
            zorder=100,
        )

        # Text of the water mark
        mark = f"FARGOpy {fargopy.__version__}"

        # Choose the according to the fact it is a 2d or 3d plot
        try:
            ax.add_collection3d
            plt_text = ax.text2D
        except:
            plt_text = ax.text

        text = plt_text(1, 1, mark, **args)
        return text

    # @staticmethod
    # def plot_heatmap(
    #     data, x=None, y=None, title="Heatmap", xlabel="X", ylabel="Y", contour_levels=10
    # ):
    #     """Plot a 2D heatmap with pcolormesh and contours.

    #     Creates a figure displaying the provided 2D data as a heatmap using a
    #     reversed Spectral colormap, overlaid with black contour lines.

    #     Parameters
    #     ----------
    #     data : np.ndarray
    #         2D array of data to plot.
    #     x : np.ndarray, optional
    #         1D array of X-axis coordinates.
    #     y : np.ndarray, optional
    #         1D array of Y-axis coordinates.
    #     title : str, optional
    #         Plot title (default: "Heatmap").
    #     xlabel : str, optional
    #         X-axis label (default: "X").
    #     ylabel : str, optional
    #         Y-axis label (default: "Y").
    #     contour_levels : int or list, optional
    #         Number of contour levels or specific level values (default: 10).

    #     Examples
    #     --------
    #     Plot a random heatmap:

    #     >>> data = np.random.rand(10, 10)
    #     >>> fp.Plot.plot_heatmap(data, title="Random Field")
    #     """
    #     plt.figure(figsize=(8, 6))

    #     if x is not None and y is not None:
    #         extent = [x.min(), x.max(), y.min(), y.max()]
    #         X, Y = np.meshgrid(x, y)
    #         # Plot the heatmap with pcolormesh
    #         mesh = plt.pcolormesh(X, Y, data, shading="auto", cmap="Spectral_r")
    #         # Add contour lines
    #         contours = plt.contour(
    #             X, Y, data, levels=contour_levels, colors="black", linewidths=0.5
    #         )
    #         plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
    #     else:
    #         # Plot the heatmap with pcolormesh
    #         mesh = plt.pcolormesh(data, shading="auto", cmap="Spectral_r")
    #         # Add contour lines
    #         contours = plt.contour(
    #             data, levels=contour_levels, colors="black", linewidths=0.5
    #         )
    #         plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")

    #     plt.colorbar(mesh, label="Value")
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.show()

    # @staticmethod
    # def interactive(sim):
    #     """
    #     Interactive plot for the simulation using ipywidgets.

    #     Allows selection of density, energy, or velocity (and component) for the colormap.
    #     Provides controls for slice, resolution, interpolation, streamlines, and Hill radius overlay.

    #     Parameters
    #     ----------
    #     sim : Simulation
    #         The simulation object to interact with.

    #     Examples
    #     --------
    #     >>> fp.Plot.interactive(sim)
    #     """
    #     import ipywidgets as widgets
    #     import matplotlib.pyplot as plt
    #     from IPython.display import display, clear_output

    #     # --- Widgets ---
    #     time_slider = widgets.IntSlider(
    #         min=0, max=sim._get_nsnaps() - 1, step=1, value=1, description="Snapshot"
    #     )
    #     slice_text = widgets.Text(value="theta=1.568", description="Slice")
    #     res_slider = widgets.IntSlider(
    #         min=50, max=1000, step=10, value=500, description="Res"
    #     )
    #     interp_toggle = widgets.ToggleButton(
    #         value=False, description="Interpolate", icon="check"
    #     )
    #     progress = widgets.Label(value="")
    #     streamlines_toggle = widgets.ToggleButton(
    #         value=False, description="Streamlines", icon="random"
    #     )
    #     density_slider = widgets.FloatSlider(
    #         min=1, max=10, step=0.5, value=3, description="Stream density"
    #     )
    #     hill_frac_slider = widgets.FloatSlider(
    #         min=0.1, max=2.0, step=0.05, value=1.0, description="Hill frac"
    #     )
    #     show_circle_toggle = widgets.ToggleButton(
    #         value=False, description="Show Hill", icon="circle"
    #     )
    #     cmap_options = [
    #         "Spectral_r",
    #         "viridis",
    #         "plasma",
    #         "inferno",
    #         "magma",
    #         "cividis",
    #         "YlGnBu",
    #         "cubehelix",
    #         "twilight",
    #         "turbo",
    #     ]
    #     cmap_dropdown = widgets.Dropdown(
    #         options=cmap_options, value="Spectral_r", description="Colormap"
    #     )
    #     update_button = widgets.Button(description="Update", icon="refresh")
    #     map_options = ["Densidad", "Energia", "Velocidad"]
    #     map_dropdown = widgets.Dropdown(
    #         options=map_options, value="Densidad", description="Mapa"
    #     )
    #     vel_components = ["vx", "vy", "vz"]
    #     vel_dropdown = widgets.Dropdown(
    #         options=vel_components, value="vx", description="Componente v"
    #     )
    #     vel_dropdown.layout.display = "none"  # Ocultar por defecto

    #     def is_fixed(var, slice_str):
    #         import re

    #         match = re.search(rf"{var}=([^\[\],]+)", slice_str.replace(" ", ""))
    #         return match is not None

    #     # show or hide velocity component dropdown based on map selection
    #     def on_map_change(change):
    #         if change["new"] == "Velocidad":
    #             vel_dropdown.layout.display = ""
    #         else:
    #             vel_dropdown.layout.display = "none"

    #     map_dropdown.observe(on_map_change, names="value")

    #     def plot_density(change=None):
    #         clear_output(wait=True)
    #         display(
    #             time_slider,
    #             slice_text,
    #             res_slider,
    #             interp_toggle,
    #             streamlines_toggle,
    #             density_slider,
    #             hill_frac_slider,
    #             show_circle_toggle,
    #             cmap_dropdown,
    #             map_dropdown,
    #             vel_dropdown,
    #             progress,
    #             update_button,
    #         )
    #         import numpy as np
    #         import re

    #         slice_str = slice_text.value
    #         res = res_slider.value
    #         interpolate = interp_toggle.value
    #         show_streamlines = streamlines_toggle.value
    #         stream_density = density_slider.value
    #         hill_frac = hill_frac_slider.value
    #         show_circle = show_circle_toggle.value
    #         cmap = cmap_dropdown.value
    #         map_type = map_dropdown.value
    #         vel_comp = vel_dropdown.value

    #         # --- Ejes y nombres de malla ---
    #         # Detect coordinate system
    #         coords = sim.vars.COORDINATES if hasattr(sim, 'vars') else 'spherical'
            
    #         if coords == 'spherical':
    #             # Spherical coordinate logic
    #             if is_fixed("theta", slice_str):
    #                 xlabel, ylabel = "X", "Y"
    #                 mesh_x_name = "var1_mesh"
    #                 mesh_y_name = "var2_mesh"
    #             elif is_fixed("phi", slice_str):
    #                 xlabel, ylabel = "X", "Z"
    #                 mesh_x_name = "var1_mesh"
    #                 mesh_y_name = "var3_mesh"
    #             else:
    #                 print(
    #                     "Warning: Please fix either theta or phi for a valid 2D slice (XY or XZ plane)."
    #                 )
    #                 return
    #         elif coords == 'cylindrical':
    #             # Cylindrical coordinate logic
    #             if is_fixed("z", slice_str):
    #                 xlabel, ylabel = "X", "Y"
    #                 mesh_x_name = "var1_mesh"
    #                 mesh_y_name = "var2_mesh"
    #             elif is_fixed("phi", slice_str):
    #                 xlabel, ylabel = "R", "Z"
    #                 mesh_x_name = "var1_mesh"
    #                 mesh_y_name = "var3_mesh"
    #             else:
    #                 print(
    #                     "Warning: Please fix either z or phi for a valid 2D slice (XY or RZ plane)."
    #                 )
    #                 return
    #         else:
    #             print(f"Warning: Unsupported coordinate system: {coords}")
    #             return

    #         n = time_slider.value

    #         if mesh_y_name == "var2_mesh":
    #             vel_dropdown.options = ["vx", "vy"]
    #             if vel_dropdown.value not in vel_dropdown.options:
    #                 vel_dropdown.value = "vx"
    #         elif mesh_y_name == "var3_mesh":
    #             vel_dropdown.options = ["vx", "vz"]
    #             if vel_dropdown.value not in vel_dropdown.options:
    #                 vel_dropdown.value = "vx"

    #         # --- Carga de datos según selección ---
    #         if map_type == "Densidad":
    #             loader = sim.load_field(
    #                 fields=["gasdens", "gasv"],
    #                 slice=slice_str,
    #                 snapshot=[n],
    #                 interpolate=interpolate,
    #             )
    #             if interpolate:
    #                 gasdens = loader
    #                 gasv = loader
    #             else:
    #                 gasdens, gasv = loader
    #         elif map_type == "Energia":
    #             gasenergy_loader = sim.load_field(
    #                 fields="gasenergy",
    #                 slice=slice_str,
    #                 snapshot=[n],
    #                 interpolate=interpolate,
    #             )
    #             if interpolate:
    #                 gasenergy = gasenergy_loader
    #             else:
    #                 gasenergy = gasenergy_loader
    #             gasv_loader = sim.load_field(
    #                 fields="gasv",
    #                 slice=slice_str,
    #                 snapshot=[n],
    #                 interpolate=interpolate,
    #             )
    #             if interpolate:
    #                 gasv = gasv_loader
    #             else:
    #                 gasv = gasv_loader
    #         elif map_type == "Velocidad":
    #             gasv_loader = sim.load_field(
    #                 fields="gasv",
    #                 slice=slice_str,
    #                 snapshot=[n],
    #                 interpolate=interpolate,
    #             )
    #             if interpolate:
    #                 gasv = gasv_loader
    #             else:
    #                 gasv = gasv_loader

    #         # --- Interpolación y selección de variable a graficar ---
    #         if not interpolate:
    #             if map_type == "Densidad":
    #                 X = getattr(gasdens, mesh_x_name)[0]
    #                 Y = getattr(gasdens, mesh_y_name)[0]
    #                 data_map = np.log10(gasdens.gasdens_mesh[0] * sim.URHO)
    #             elif map_type == "Energia":
    #                 X = getattr(gasenergy, mesh_x_name)[0]
    #                 Y = getattr(gasenergy, mesh_y_name)[0]
    #                 data_map = np.log10(gasenergy.gasenergy_mesh[0])
    #             elif map_type == "Velocidad":
    #                 X = getattr(gasv, mesh_x_name)[0]
    #                 Y = getattr(gasv, mesh_y_name)[0]
    #                 idx = {"vx": 0, "vy": 1, "vz": 2}[vel_comp]
    #                 data_map = gasv.gasv_mesh[0][idx]
    #             vx = vy = vmag = None
    #         else:
    #             progress.value = "Interpolando..."
    #             if mesh_y_name == "var2_mesh":
    #                 xmin, xmax = (
    #                     getattr(gasv, mesh_x_name)[0].min(),
    #                     getattr(gasv, mesh_x_name)[0].max(),
    #                 )
    #                 ymin, ymax = (
    #                     getattr(gasv, mesh_y_name)[0].min(),
    #                     getattr(gasv, mesh_y_name)[0].max(),
    #                 )
    #                 xs = np.linspace(xmin, xmax, res)
    #                 ys = np.linspace(ymin, ymax, res)
    #                 X, Y = np.meshgrid(xs, ys)
    #                 if map_type == "Densidad":
    #                     data_map = gasdens.evaluate(
    #                         time=n, var1=X, var2=Y, field="gasdens"
    #                     )
    #                     data_map = np.log10(data_map * sim.URHO)
    #                     vel = gasv.evaluate(time=n, var1=X, var2=Y, field="gasv")
    #                     vx = vel[0]
    #                     vy = vel[1]
    #                     vmag = np.sqrt(vx**2 + vy**2)
    #                 elif map_type == "Energia":
    #                     data_map = gasenergy.evaluate(
    #                         time=n, var1=X, var2=Y, field="gasenergy"
    #                     )
    #                     # data_map = np.log10(data_map)
    #                     vel = gasv.evaluate(time=n, var1=X, var2=Y, field="gasv")
    #                     vx = vel[0]
    #                     vy = vel[1]
    #                     vmag = np.sqrt(vx**2 + vy**2)
    #                 elif map_type == "Velocidad":
    #                     vel = gasv.evaluate(time=n, var1=X, var2=Y, field="gasv")
    #                     idx = {"vx": 0, "vy": 1, "vz": 2}[vel_comp]
    #                     data_map = vel[idx]
    #                     vx = vel[0]
    #                     vy = vel[1]
    #                     vmag = np.sqrt(vx**2 + vy**2)
    #             else:
    #                 xmin, xmax = (
    #                     getattr(gasv, mesh_x_name)[0].min(),
    #                     getattr(gasv, mesh_x_name)[0].max(),
    #                 )
    #                 zmin, zmax = (
    #                     getattr(gasv, mesh_y_name)[0].min(),
    #                     getattr(gasv, mesh_y_name)[0].max(),
    #                 )
    #                 xs = np.linspace(xmin, xmax, res)
    #                 zs = np.linspace(zmin, zmax, res)
    #                 X, Y = np.meshgrid(xs, zs)
    #                 if map_type == "Densidad":
    #                     data_map = gasdens.evaluate(
    #                         time=n, var1=X, var3=Y, field="gasdens"
    #                     )
    #                     data_map = np.log10(data_map * sim.URHO)
    #                     vel = gasv.evaluate(time=n, var1=X, var3=Y, field="gasv")
    #                     vx = vel[0]
    #                     vy = vel[2]
    #                     vmag = np.sqrt(vx**2 + vy**2)
    #                 elif map_type == "Energia":
    #                     data_map = gasenergy.evaluate(
    #                         time=n, var1=X, var3=Y, field="gasenergy"
    #                     )
    #                     # data_map = np.log10(data_map)
    #                     vel = gasv.evaluate(time=n, var1=X, var3=Y, field="gasv")
    #                     vx = vel[0]
    #                     vy = vel[2]
    #                     vmag = np.sqrt(vx**2 + vy**2)
    #                 elif map_type == "Velocidad":
    #                     vel = gasv.evaluate(time=n, var1=X, var3=Y, field="gasv")

    #                     idx = {"vx": 0, "vy": 1, "vz": 2}[vel_comp]
    #                     data_map = vel[idx]
    #                     vx = vel[0]
    #                     vy = vel[2]
    #                     vmag = np.sqrt(vx**2 + vy**2)

    #         # --- Máscara por rango r (igual que antes) ---
    #         r = np.sqrt(X**2 + Y**2)
    #         r_match = re.search(
    #             r"r=\[([0-9\.]+),([0-9\.]+)\]", slice_str.replace(" ", "")
    #         )
    #         if r_match:
    #             r_min = float(r_match.group(1))
    #             r_max = float(r_match.group(2))
    #         else:
    #             r_min = None
    #             r_max = None

    #         if r_min is not None and r_max is not None:
    #             mask = (r >= r_min) & (r <= r_max)
    #             data_map = np.where(mask, data_map, np.nan)
    #             if (
    #                 show_streamlines
    #                 and vx is not None
    #                 and vy is not None
    #                 and vmag is not None
    #             ):
    #                 vx = np.where(mask, vx, np.nan)
    #                 vy = np.where(mask, vy, np.nan)
    #                 vmag = np.where(mask, vmag, np.nan)

    #         # --- Plot ---
    #         fig, ax = plt.subplots(figsize=(7, 5))
    #         pcm = ax.pcolormesh(
    #             X * sim.UL / sim.AU,
    #             Y * sim.UL / sim.AU,
    #             data_map,
    #             shading="auto",
    #             cmap=cmap,
    #         )

    #         # Mostrar streamlines para cualquier tipo de mapa si están disponibles
    #         stream_obj = None
    #         if interpolate and show_streamlines and vx is not None and vy is not None:
    #             stream_obj = ax.streamplot(
    #                 X * sim.UL / sim.AU,
    #                 Y * sim.UL / sim.AU,
    #                 vx,
    #                 vy,
    #                 color=vmag * sim.UL / sim.UT * 1e-5 if vmag is not None else None,
    #                 linewidth=0.5,
    #                 density=stream_density,
    #                 cmap="viridis",
    #                 arrowsize=1,
    #             )

    #         # --- Hill radius
    #         planets = sim.load_planets(snapshot=n)
    #         if planets:
    #             center_x = planets[0].pos.x
    #             center_y = planets[0].pos.y
    #             radius = hill_frac * planets[0].hill_radius
    #         else:
    #             center_x = 0
    #             center_y = 0
    #             radius = 0

    #         if show_circle:
    #             if is_fixed("theta", slice_str):
    #                 circle = plt.Circle(
    #                     (center_x * sim.UL / sim.AU, center_y * sim.UL / sim.AU),
    #                     radius * sim.UL / sim.AU,
    #                     color="black",
    #                     fill=False,
    #                     linestyle="--",
    #                     linewidth=1,
    #                 )
    #                 ax.add_patch(circle)
    #             elif is_fixed("phi", slice_str):
    #                 theta = np.linspace(0, np.pi, 100)
    #                 x = center_x + radius * np.cos(theta)
    #                 y = center_y + radius * np.sin(theta)
    #                 ax.plot(
    #                     x * sim.UL / sim.AU,
    #                     y * sim.UL / sim.AU,
    #                     color="black",
    #                     linewidth=2,
    #                 )

    #         ax.set_xlabel(xlabel + " [AU]")
    #         ax.set_ylabel(ylabel + " [AU]")
    #         # ax.axis('equal')

    #         if (
    #             interpolate
    #             and show_streamlines
    #             and stream_obj is not None
    #             and vmag is not None
    #         ):
    #             # Colorbar for velocity magnitude (streamlines)
    #             cbar = fig.colorbar(stream_obj.lines, ax=ax, label=r"$|v|$ [km/s]")
    #         else:
    #             # Colorbar for main map
    #             if map_type == "Densidad":
    #                 cbar_label = r"$\log_{10}(\rho) [g/cm^3]$"
    #             elif map_type == "Energia":
    #                 cbar_label = r"$\log_{10}(\mathrm{energy})$"
    #             else:
    #                 cbar_label = f"{vel_comp} [AU]"
    #             fig.colorbar(pcm, ax=ax, label=cbar_label)

    #         fargopy.Plot.fargopy_mark(ax)
    #         plt.show()

    #     # --- Events ---
    #     update_button.on_click(plot_density)
    #     slice_text.on_submit(plot_density)
    #     show_circle_toggle.observe(plot_density, names="value")
    #     interp_toggle.observe(plot_density, names="value")
    #     streamlines_toggle.observe(plot_density, names="value")
    #     cmap_dropdown.observe(plot_density, names="value")
    #     map_dropdown.observe(plot_density, names="value")
    #     vel_dropdown.observe(plot_density, names="value")

    #     # --- Display inicial ---
    #     display(
    #         time_slider,
    #         slice_text,
    #         res_slider,
    #         interp_toggle,
    #         streamlines_toggle,
    #         density_slider,
    #         hill_frac_slider,
    #         show_circle_toggle,
    #         cmap_dropdown,
    #         map_dropdown,
    #         vel_dropdown,
    #         progress,
    #         update_button,
    #     )
    #     plot_density()

    @staticmethod
    def mesh(
        sim,
        snapshot=0,
        slice="theta=1.56",
        planet=0,
        draw_hill=True,
        hill_frac=1.0,
        figsize=(8, 8),
        point_size=1,
        line_alpha=0.5,
        cmap="viridis",
        show=True,
    ):
        """
        Plot the simulation mesh in the XY plane and (optionally) the planet Hill circle.

        Parameters
        ----------
        sim : Simulation
            The simulation object.
        snapshot : int, optional
            Snapshot to plot, by default 0.
        slice : str, optional
            Slice definition, by default 'theta=1.56'.
        planet : int or str, optional
            Planet index or name to focus, by default 0.
        draw_hill : bool, optional
            Whether to draw the Hill sphere, by default True.
        hill_frac : float, optional
            Fraction of Hill radius to draw, by default 1.0.
        figsize : tuple, optional
            Figure size, by default (8,8).
        point_size : int, optional
            Size of mesh points, by default 1.
        line_alpha : float, optional
            Alpha transparency of mesh lines, by default 0.5.
        cmap : str, optional
            Colormap for points, by default 'viridis'.
        show : bool, optional
            Whether to show the plot, by default True.

        Returns
        -------
        tuple
            (fig, ax, nr_celdas_radial, nr_celdas_azimutal, n_inside)
            Matplotlib figure and axes, max contiguous radial cells, max contiguous azimuthal cells,
            and the number of mesh cells inside the hill_frac * Hill radius.

        Examples
        --------
        >>> fp.Plot.mesh(sim, snapshot=0)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        # Load a 2D interpolated field (keeps same interface used elsewhere)
        gasdens = sim.load_field(
            fields=["gasdens"], snapshot=snapshot, slice=slice
        )

        # Expect interpolator result with var1_mesh / var2_mesh (as used in plot_interactive)
        try:
            X = gasdens.var1_mesh[0]
            Y = gasdens.var2_mesh[0]
        except Exception:
            # Fallback: if a raw Field-like object is returned with mesh names var1_mesh/var2_mesh attributes
            X = getattr(gasdens, "var1_mesh", None)
            Y = getattr(gasdens, "var2_mesh", None)
            if X is None or Y is None:
                raise RuntimeError(
                    "Could not obtain var1_mesh/var2_mesh from loaded field. Use a valid slice."
                )

        # Prepare figure
        plt.close("all")
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points (convert to AU for axis if simulation units defined)
        scale = getattr(sim, "UL", 1.0) / getattr(sim, "AU", 1.0)
        ax.scatter(
            (X * scale).ravel(),
            (Y * scale).ravel(),
            s=point_size,
            c=(X * 0 + 0.5).ravel(),
            cmap=cmap,
            marker=".",
            linewidths=0,
        )

        # If mesh is 2D arrays, draw grid lines
        if X.ndim == 2 and Y.ndim == 2:
            # rows
            for i in range(X.shape[0]):
                ax.plot(
                    X[i, :] * scale,
                    Y[i, :] * scale,
                    color="gray",
                    linewidth=0.5,
                    alpha=line_alpha,
                )
            # columns
            for j in range(X.shape[1]):
                ax.plot(
                    X[:, j] * scale,
                    Y[:, j] * scale,
                    color="gray",
                    linewidth=0.5,
                    alpha=line_alpha,
                )

        # Planet selection
        planets = sim.load_planets(snapshot=snapshot)
        center_x = center_y = None
        radius = 0.0
        if planets:
            sel = None
            if isinstance(planet, int):
                try:
                    sel = planets[planet]
                except Exception:
                    sel = planets[0]
            else:
                # name lookup
                for p in planets:
                    if getattr(p, "name", None) == planet:
                        sel = p
                        break
                if sel is None:
                    sel = planets[0]
            # planet object expected to have pos.x / pos.y and hill_radius property
            center_x = sel.pos.x
            center_y = sel.pos.y
            if draw_hill:
                radius = hill_frac * getattr(sel, "hill_radius", 0.0)

        # Draw Hill circle if requested and compute counts
        nr_celdas_radial = 0
        nr_celdas_azimutal = 0
        n_inside = 0
        if draw_hill and center_x is not None and center_y is not None and radius > 0:
            circle = patches.Circle(
                (center_x * scale, center_y * scale),
                radius * scale,
                edgecolor="red",
                facecolor="lightblue",
                linestyle="-",
                linewidth=1.5,
            )
            ax.add_patch(circle)

            # Count mesh cells (points) inside the requested fraction of Hill radius
            try:
                # X,Y are in simulation length units (same as center_x, center_y)
                mask_inside = ((X - center_x) ** 2 + (Y - center_y) ** 2) <= (radius**2)
                n_inside = int(np.count_nonzero(mask_inside))

                # If mesh is structured 2D array, compute contiguous runs:
                if X.ndim == 2 and Y.ndim == 2:
                    # Helper to get max contiguous True length in a 1D boolean array
                    def max_contiguous_true(arr1d):
                        idx = np.flatnonzero(arr1d)
                        if idx.size == 0:
                            return 0
                        splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
                        lengths = [s.size for s in splits]
                        return max(lengths) if lengths else 0

                    # Azimutal: along rows (axis 1) -> for each row find longest contiguous True segment
                    max_az = 0
                    for i in range(mask_inside.shape[0]):
                        l = max_contiguous_true(mask_inside[i, :])
                        if l > max_az:
                            max_az = l
                    nr_celdas_azimutal = max_az

                    # Radial: along cols (axis 0) -> for each col find longest contiguous True segment
                    max_rad = 0
                    for j in range(mask_inside.shape[1]):
                        l = max_contiguous_true(mask_inside[:, j])
                        if l > max_rad:
                            max_rad = l
                    nr_celdas_radial = max_rad

            except Exception as e:
                print(f"Warning computing counts: {e}")

        fargopy.Plot.fargopy_mark(ax)
        ax.set_aspect("equal")
        ax.set_xlabel("x [AU]")
        ax.set_ylabel("y [AU]")

        if show:
            plt.show()

        return fig, ax, nr_celdas_radial, nr_celdas_azimutal, n_inside
