<p></p>
<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/fargopy/main/docs/fargopy_logo.webp" alt="FARGOpy Logo" width="600"/>
</div>
<p></p>

<h2 align="center">A FARGO3D wrapper and more</h2>

<!-- This are visual tags that you may add to your package at the beginning with useful information on your package -->
[![version](https://img.shields.io/pypi/v/fargopy?color=blue)](https://pypi.org/project/fargopy/) 
[![downloads](https://img.shields.io/pypi/dw/fargopy)](https://pypi.org/project/fargopy/) 
[![License](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://github.com/seap-udea/fargopy/blob/master/LICENSE) 
[![python](https://img.shields.io/badge/python-3-grey)](https://pypi.org/project/fargopy/) 
[![Powered by FARGO3D](https://img.shields.io/badge/Powered%20by-FARGO3D-blue)](https://fargo3d.bitbucket.io/)
<!--[![arXiv](https://img.shields.io/badge/arXiv-0000.00000-orange.svg?style=flat)](https://arxiv.org/abs/0000.00000)-->
<a target="_blank" href="https://colab.research.google.com/github/seap-udea/fargopy/blob/main/README.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/fargopy/main/gallery/pds-70c-disk_densiy-200_orbits_high_resolution.gif" alt="PDS 70c disk density" width="60%"/>
  <!--<img src="https://raw.githubusercontent.com/seap-udea/fargopy/main/gallery/pds-70c-disk_densiy_vertical-200_orbits_high_resolution.gif" alt="PDS 70c disk density (vertical)" width="45%"/>-->
</div>

## Introducing FARGOpy

`FARGOpy` is a Python wrapper and post-processing tool designed for `FARGO3D`, a widely used hydrodynamical code for simulating planet-disk interactions.

With `FARGOpy`, you can easily:

- Analyze and visualize simulation outputs.
- Control and run `FARGO3D` simulations directly from Python (optional).
- Generate complex initial conditions and diverse setups with minimal effort.

It streamlines the workflow for researchers, allowing them to focus on the physics rather than the technicalities of setting up and processing simulations. 

For instance, the animations above show the gas density of the circumstellar disk around the planet **PDS-70c** coming from a `FARGO3D` high resolution simulation. The reading of the simulation output and the generation of the animations, the interpolation of the fields, and the creation of the animations   with just a few lines of code. 

For the code used to generate these animations, see the tutorial notebook [basics with FARGOpy](https://github.com/seap-udea/fargopy/blob/main/examples/fargopy-tutorial-basics.ipynb).

## Resources

A complete list of resources and further information about the package and the science relate to it can be found in the following links:

- **Technical report**: [FARGOpy: A Python Package for Post-processing and Analyzing FARGO3D Hydrodynamical Simulations](https://github.com/seap-udea/fargopy/blob/main/science/introducing-fargopy/MurilloZuluagaMontesinos2026-IntroducingFARGOpy.pdf)
- **GitHub Repository**: [https://github.com/seap-udea/fargopy](https://github.com/seap-udea/fargopy)
- **Documentation**: [https://fargopy.readthedocs.io](https://fargopy.readthedocs.io)
- **PyPI Page**: [https://pypi.org/project/fargopy/](https://pypi.org/project/fargopy/)

## Authors and Licensing

This project is developed by members the **Solar, Earth and Planetary Physics Group (SEAP)** at Universidad de Antioquia, Medellín, Colombia and the the **Department of Physics** of the Universidad Técnica Federico Santa María, Valdivia, Chile. The main developers are:

- **Jorge I. Zuluaga** (SEAP/FACom/UdeA) - jorge.zuluaga@udea.edu.co
- **Alejandro Murillo-González** (SEAP/FACom/UdeA) - alejandro.murillo1@udea.edu.co
- **Matías Montesinos** (Physics/USM) - matias.montesinosa@usm.cl

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## What's New

For a detailed list of changes and new features in each version, please see the [WHATSNEW.md](https://github.com/seap-udea/fargopy/blob/main/WHATSNEW.md) file.

----

## Installation

### From PyPI

`FARGOpy` is available at the `Python` package index and can be installed using:

```bash
$ pip install fargopy
```
or if you want the latest (and possibly unstable) version use:

```bash
$ pip install git+https://github.com/seap-udea/fargopy
```

as usual this command will install all dependencies (excluding `FARGO3D` which must be installed indepently as explained before) and download some useful data, scripts and constants. Installation includes two system-level commands, `ifargopy` used mainly for running the package in the `IPython` environmente (see below) and `vfargopy` which is the command to open the graphical interface (see below).

### From sources

You can also install from the [GitHub repository](https://github.com/seap-udea/fargopy):

```bash
git clone https://github.com/seap-udea/fargopy
cd fargopy
pip install .
```

For development, use an editable installation:

```bash
cd fargopy
pip install -e .
```

### In Google Colab

Since `FARGOpy` is a python wrap for `FARGO3D` the ideal environment to work with the package is `IPython`/`Jupyter`. It works really fine in `Google Colab` ensuing training and demonstration purposes. This README, for instance, can be ran in `Google Colab`:

<a target="_blank" href="https://colab.research.google.com/github/seap-udea/fargopy/blob/main/README.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This code only works in Colab and it is intended to install the latest version of `FARGOpy`

```python
try:
    from google.colab import drive
    %pip install -Uq git+https://github.com/seap-udea/fargopy
except ImportError:
    print("Not running in Colab, skipping installation")
%mkdir -p ./gallery/
```

    Not running in Colab, skipping installation

### Running in `IPython`

If you are working on a remote Linux server, it is better to run the package using `IPython`. For this purpose, after installation, `FARGOpy` provides a special initialization command:

```bash
$ ifargopy
```

The first time you run this script, it will create a configuration directory `~/.fargopy` (with `~` the abbreviation for the home directory). This directory contains a set of basic configuration variables which are stored in the file `~/.fargopy/fargopyrc`. You may change this file if you want to customize the installation. The configuration directory also contains the `IPython` initialization script `~/.fargopy/ifargopy.py`.

You may also use the commando `ifargopy` to run several interesting commands:

- Verify the installation:

    ```bash
    $ ifargopy --verify
    ```
    ```
    Running FARGOpy version X.Y.Z
    fargopy X.Y.Z is successfully installed.
    Location: /usr/local/lib/pythonX.X/site-packages/fargopy
    ```

- Run a battery of tests:

    ```bash
    $ ifargopy --test
    ```

-----

<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/fargopy/main/gallery/pds-70c-disk_densiy_vertical-200_orbits_high_resolution.gif" alt="PDS 70c disk density (vertical)" width="45%"/>
</div>

## Quickstart

Here is a quick example of how to use FARGOpy. For more examples, see the [examples](https://fargopy.readthedocs.io/en/latest/examples.html) directory in the documentation.

Import the package:

```python
import fargopy as fp
```

### Density map

Download a precomputed simulation to test the package:

```python
fp.Simulation.download_precomputed('fargo')
```

    Downloading fargo.tgz from cloud (compressed size around 55 MB) into /tmp

    Downloading...
    From: https://docs.google.com/uc?export=download&id=1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3
    To: /tmp/fargo.tgz
    100%|██████████| 54.7M/54.7M [00:01<00:00, 46.5MB/s]

    Uncompressing fargo.tgz into /tmp/fargo
    Done.

    '/tmp/fargo'

Connect to the simulation output directory:

```python
sim = fp.Simulation(output_dir='/tmp/fargo')
```

    Your simulation is now connected with '/Users/jzuluaga/fargo3d/'
    Now you are connected with output directory '/tmp/fargo'
    Found a variables.par file in '/tmp/fargo', loading properties
    Loading variables
    84 variables loaded
    Simulation in 2 dimensions
    Loading domain in cylindrical coordinates:
    	Variable phi: 384 [[0, np.float64(-3.1334114227210694)], [-1, np.float64(3.1334114227210694)]]
    	Variable r: 128 [[0, np.float64(0.408203125)], [-1, np.float64(2.491796875)]]
    	Variable z: 1 [[0, np.float64(0.0)], [-1, np.float64(0.0)]]
    Number of snapshots in output directory: 51
    Planets found in summary.dat:
      Name: Jupiter, Initial pos: [1.0, 0.001, 0.0], Mass: 0.001

Load a field (e.g., gas density) from a specific snapshot:

```python
gasdens = sim.load_field('gasdens', snapshot=20, interpolate=False)
```

Crate a 2D slice of a 3D field at $z=0$, 

```python
gasdens_plane, mesh = gasdens.meshslice(slice='z=0')
```

Plot the fields of the FARGO simulation using a `colormesh` plot:

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6,6))
ax.pcolormesh(mesh.x, mesh.y, gasdens_plane, cmap='prism')
ax.axis('equal')
ax.set_xlabel('x [au]')
ax.set_ylabel('y [au]')

fp.Plot.fargopy_mark(ax)
plt.savefig('gallery/readme-gasdens.png')
plt.show()
```

<img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/gallery/readme-gasdens.png" alt="png">

### Streamlines

We can visualize the streamlines of the fluid to observe the flow direction at a specific snapshot by interpolating the velocity field. This visualization provides a detailed representation of the velocity field, enabling us to analyze the fluid dynamics and identify key patterns such as vortices, flow separations, or other phenomena of interest. By combining this with contour plots of other fields, such as density or energy, we can gain deeper insights into the interactions and behavior of the simulated system.

We will use the 3D simulation of a disk with a Jovian planet:

```python
fp.Simulation.download_precomputed('p3disoj')
sim = fp.Simulation(output_dir='/tmp/p3disoj')
```

    Precomputed output directory '/tmp/p3disoj' already exist
    Your simulation is now connected with '/Users/jzuluaga/fargo3d/'
    Now you are connected with output directory '/tmp/p3disoj'
    Found a variables.par file in '/tmp/p3disoj', loading properties
    Loading variables
    85 variables loaded
    Simulation in 3 dimensions
    Loading domain in spherical coordinates:
    	Variable phi: 128 [[0, np.float64(-3.117048960983623)], [-1, np.float64(3.117048960983623)]]
    	Variable r: 64 [[0, np.float64(0.5078125)], [-1, np.float64(1.4921875)]]
    	Variable theta: 32 [[0, np.float64(1.4231400767948967)], [-1, np.float64(1.5684525767948965)]]
    Number of snapshots in output directory: 11
    Planets found in summary.dat:
      Name: Jupiter, Initial pos: [1.0, 0.001, 0.0], Mass: 0.001

First load the density and velocity fields at snapshot 6:

```python
data = sim.load_field(
    fields=['gasv'],
    slice="phi=0",
    snapshot=(0,10)
    )
```

Now is necesary create a regular mesh where the value of the field is interpolated:

```python
import numpy as np

# This is the snapshot number where the streamplot will be computed
time = 6

# Get the minimum and maximum values of var1 and var3 (x and z coordinates)
xmin=data.var1_mesh[time].min()
xmax=data.var1_mesh[time].max()
zmin=data.var3_mesh[time].min()
zmax=data.var3_mesh[time].max()

# Create a grid of points
resolution = 120
xs = np.linspace(xmin, xmax, resolution)
zs = np.linspace(zmin, zmax, resolution)
X, Z = np.meshgrid(xs, zs, indexing='ij')
```

Using the mesh we can evaluate the field at any point in space:

```python
vxs, vys, vzs = data.evaluate(field='gasv', time=6, var1=X, var3=Z)

# Velocity magnitude at each mesh point
v_mag = np.sqrt(vxs**2 + vzs**2).T *sim.UV / 1e5 
```

Now we can plot the streamlines of the gas velocity:

```python
fig,axs = plt.subplots(1,1,figsize=(6,6))

cmap = 'Spectral_r'

strm = axs.streamplot(X.T, Z.T, vxs.T, vzs.T,
                       color=v_mag, linewidth=0.7, density=3.0, cmap='viridis')

fig.colorbar(strm.lines, ax=axs, label="|v| [km/s]")

axs.set_xlim(xmin, xmax)
axs.set_ylim(zmin, zmax)
axs.set_xlabel('$x$ [au]')
axs.set_ylabel('$z$ [au]')

fp.Plot.fargopy_mark(axs)
plt.savefig('gallery/readme-streamlines.png') # Save figure
```

<img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/gallery/readme-streamlines.png" alt="png">

### Accretion rate (mass flux)

We compute the mass flux through a closed surface S around the planet. The mass flux (accretion rate) is:\n
$$\dot{M}=\int_S\rho(\mathbf{v}\cdot\hat{n})\,dS$$
where $\rho$ is the density, $\mathbf{v}$ the velocity vector and $\hat{n}$ the outward normal to the surface. We evaluate this integral by interpolating fields at triangle centers and summing $\rho (\mathbf{v}dotat{n}) A_{i}$ over tessellation triangles.

### Define Planet and Surface for Accretion Calculation

Set up the planet and surface objects for the accretion rate calculation.

For this example we will calculate the total flux going through a sphere with a radius equal to the planet's hill radius. For this purpose we need the location and the value of Hill radius of the planet at the snapchot we are interested in:

```python
snap = 10
planet = sim.load_planets(snapshot=snap)[0]
r_hill = planet.hill_radius
print(f"Jupiter Hill Radius: {r_hill * sim.UL / fp.AU:.3f} AU")
```

    Jupiter Hill Radius: 0.068 AU

Now we define the surface over which you want to compute the fluxes:

```python
sphere = fp.Surface(
    type='sphere',
    radius=r_hill,
    subdivisions=6,
)
```

The computation is performed with `follow_planet=True`, ensuring that the integration surface co-moves with the planet. This choice is required because the Hill radius evolves in time as a consequence of planetary migration, and therefore the associated control volume must be updated consistently to remain centered on the planet.

```python
acc_rate = sphere.mass_flux(sim=sim, snapshot=[0,snap], follow_planet=True) 
```

    Calculating mass flux: 100%|██████████| 11/11 [00:37<00:00,  3.44s/it]

And we can plot it:

```python
snaps = np.linspace(0, snap, len(acc_rate))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(snaps, acc_rate * sim.UM / sim.UT, color='dodgerblue')
ax.set_xlabel('Planet Orbits', size=15)
ax.set_ylabel(r'$\dot{M}$ [kg/s]', size=15)
ax.legend(fontsize=12)
fp.Plot.fargopy_mark(ax)
plt.savefig('gallery/readme-accretion.png')

```

<img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/gallery/readme-accretion.png" alt="png">

## Graphical interface for FARGOpy

In order to ease the manipulation of the FARGO3D data FARGOpy provides a graphical interface written in Python using PyQt5. Below you can find a snapshot of the current version of the interface.

<div align="center">
    <img src="https://raw.githubusercontent.com/seap-udea/fargopy/main/gallery/vfargopy.png" alt="FARGOpy Interface" width="600">
</div>

In order to run the graphical interface use:

```bash
$ vfargopy
```

-----

## Contributing

We welcome contributions! If you're interested in contributing to MultiNEAs, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

## Citation

If you use `FARGOpy` in your research, please cite:

```bibtex
@software{FARGOpy2026,
  author = {Zuluaga, Jorge I., Murillo-González, Alejandro and Montesinos, Matías},
  title = {FARGOpy: A python wrapper for FARGO3D and more},
  year = {2026},
  url = {https://github.com/seap-udea/fargopy}
}

@misc{MurilloGonzalez2026,
    author       = {Murillo-Gonzalez, Alejandro and Zuluaga, Jorge I. and Montesinos, Matias},
    title        = {{FARGOpy}: A {Python} Package for Post-processing and Analyzing {FARGO3D} Hydrodynamical Simulations},
    year         = {2026},
    howpublished = {\url{https://github.com/seap-udea/fargopy/blob/main/science/introducing-fargopy/MurilloZuluagaMontesinos2026-IntroducingFARGOpy.pdf}},
    note         = {Manuscript hosted on GitHub},
}
```

<!-- You may also cite our first science paper using the package:

```bibtex
@article{Zuluaga2026,
  author = {Murillo-González, Alejandro, Zuluaga, Jorge I. and Montesinos, Matías},
  title = {Characterizing the PDS 70c Circumplanetary Disk: High-Resolution 3D Simulations and Analysis with FARGO3D and FARGOpy},
  year = {2026},
  journal = {In preparation},
  url = {https://github.com/seap-udea/fargopy}
}
``` -->

---
*Powered by fargopy*. For more examples see [fargopy GitHub repo](https://github.com/seap-udea/fargopy/tree/main/examples). 

Jorge I. Zuluaga, Alejandro Murillo-González and Matías Montesinos © 2023-present

