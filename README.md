<p></p>
<div align="center">
  <img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/docs/fargopy_logo.webp" alt="FARGOpy Logo" width="600"/>
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
  <img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/gallery/pds-70c-disk_densiy-200_orbits_high_resolution.gif" alt="PDS 70c disk density" width="60%"/>
  <!--<img src="https://raw.githubusercontent.com/seap-udea/fargopy/refactor/gallery/pds-70c-disk_densiy_vertical-200_orbits_high_resolution.gif" alt="PDS 70c disk density (vertical)" width="45%"/>-->
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

- **GitHub Repository**: [https://github.com/seap-udea/fargopy](https://github.com/seap-udea/fargopy)
- **Documentation**: [https://fargopy.readthedocs.io](https://fargopy.readthedocs.io)
- **PyPI Page**: [https://pypi.org/project/fargopy/](https://pypi.org/project/fargopy/)

## Installation

### From PyPI

`FARGOpy` is available at the `Python` package index and can be installed using:

```bash
$ pip install fargopy
```
as usual this command will install all dependencies (excluding `FARGO3D` which must be installed indepently as explained before) and download some useful data, scripts and constants.

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

## Quickstart

Here is a quick example of how to use FARGOpy. For more examples, see the [examples](file:///Users/jzuluaga/dev/fargopy/docs/_build/html/examples.html) directory in the documentation.

Import the package:

```python
import fargopy as fp
```

    Running FARGOpy version X.Y.Z

Download a precomputed simulation to test the package:

```python
fp.Simulation.download_precomputed('fargo')
```

    Downloading fargo.tgz from cloud (compressed size around 55 MB) into /tmp

    Downloading...
    From: https://docs.google.com/uc?export=download&id=1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3
    To: /tmp/fargo.tgz
    100%|██████████| 54.7M/54.7M [00:01<00:00, 35.2MB/s]

    Uncompressing fargo.tgz into /tmp/fargo
    Done.

    '/tmp/fargo'

Connect to the simulation output directory:

```python
sim = fp.Simulation(output_dir='/tmp/fargo')
```

    Your simulation is now connected with '/local_directory/fargo3d/'
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

## What's New

For a detailed list of changes and new features in each version, please see the [WHATSNEW.md](https://github.com/seap-udea/fargopy/blob/main/WHATSNEW.md) file.

## Authors and Licensing

This project is developed by the Solar, Earth and Planetary Physics Group (SEAP) at Universidad de Antioquia, Medellín, Colombia. The main developers are:

- **Jorge I. Zuluaga** - jorge.zuluaga@udea.edu.co
- **Alejandro Murillo-González** - alejandro.murillo1@udea.edu.co
- **Matías Montesinos** - matias.montesinosa@usm.cl

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! If you're interested in contributing to MultiNEAs, please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Please read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.

