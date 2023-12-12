# FARGOpy
## Wrapping FRAGO3D

<!-- This are visual tags that you may add to your package at the beginning with useful information on your package --> 
[![version](https://img.shields.io/pypi/v/fargopy?color=blue)](https://pypi.org/project/fargopy/)
[![downloads](https://img.shields.io/pypi/dw/fargopy)](https://pypi.org/project/fargopy/)

`FARGOpy` is a python wrapping for [`FARGO3D`](https://fargo3d.bitbucket.io/intro.html)., the well-knwon hydrodynamics and magnetohydrodynamics parallel code. This wrapping is intended to ensue the interaction with FARGO3D especially for those starting using the code, for instance for teaching and training purposes, but also provide functionalities for most advanced users in tasks related to the postprocessing of output files and plotting.

This is an example of what can be done using `FARGOpy`:

<p align="center"><img src="https://github.com/seap-udea/fargopy/blob/main/gallery/fargo-animation.gif?raw=true" alt="Animation""/></p>

## Install `FARGOpy` 

To install the package run:

```bash
$ pip3 install fargopy
```

Since `FARGOpy` is a python wrap for `FARGO3D` the ideal environment to work with the package is `IPython` or other similar environments such as `jupyter-lab`, `Google Colab`, etc. 

If you are working on a Linux server it is better to run the package for the first time using the `IPython` initialization command:

```bash
$ ifargopy
```

The first time you run this command, it will create the configuration directory `$HOME/.fargopy`. This directory contains a set of basic configuration variables `$HOME/.fargopy/fargopyrc`, that can be customized, and the `IPython` initialization script `$HOME/.fargopy/ifargopy.py`.

If you are working in `Jupyter` or in `Google Colab`, the configuration directory and its content will be crated the first time you import the package:


```python
import fargopy as fp
%load_ext autoreload
%autoreload 2
```

    Running FARGOpy version 0.2.1


## Download and install FARGO3D

`FARGOpy` can be used either to *control* `FARGO3D`, ie. download, compile and run it, and/or read and manipulate the output of simulations.  In either case it would be convenient to obtain a copy of the package and all its prerequisites. For a detailed guide please see the [FARGO documentation](https://fargo3d.bitbucket.io/index.html) or the [project repo at bitbucket](https://bitbucket.org/fargo3d/public/src/ae0fcdc67bb7c83aed85fc9a4d4a2d5061324597/?at=release%2Fpublic). 

> **NOTE**: It is important to understand that `FARGO3D` works especially well on Linux plaforms (including `MacOS`). The same condition applies for `FARGOpy`. Because of that, most internal as well as public features of the packages are designed to work in a `Linux` environment. For working in another operating systems, for instance for teaching or training purposes, please consider to use virtual machines.

`FARGOpy` provides a simple way to get the latest version of `FARGO3D`. In the terminal run:

 ```shell
$ ifargopy download
```

A copy of `FARGO3D` will be download in the HOME directory `~/fargo3d/`.

If you are working in a `Jupyter` environment you may run:


```python
fp.initialize('download')
```

    Downloading FARGOpy...


    Cloning into 'fargo3d'...


    	FARGO3D downloaded to /home/jzuluaga/fargo3d/
    Header file for FARGO3D is in the fargo directory /home/jzuluaga/fargo3d/


> **NOTE**: As you may see the default directory where FARGO3D is download is `~/` and the name of the directory is `fargo3d`. You may change this by setting the configuration variables `FP_FARGO3D_BASEDIR` or `FP_FARGO3D_PACKDIR` brefore downloading the package.

## Quickstart

Here we will illustrate the minimal commands you may run to test the package. A more detailed set of examples can be found in [this file](EXAMPLES.md). 

For this example we will assume that you already have a set of FARGO3D simulation results. You may download a precomputed set of results prepared by the developers of `FARGOpy` using the command: 


```python
fp.Util.download_precomputed(setup='fargo')
```

    Downloading fargo.tgz from cloud (compressed size around 55 MB) into /tmp
    Uncompressing fargo.tgz into /tmp/fargo
    Done.





    '/tmp/fargo'



Create a simulation object:


```python
sim = fp.Simulation()
```

Set the output directory (in this case, the directory where the precomputed simulation has been stored):


```python
sim.set_output_dir('/tmp/fargo')
```

    Now you are connected with output directory '/tmp/fargo'


Load the properties of the simulation:


```python
sim.load_properties()
```

    Loading variables
    84 variables loaded
    Simulation in 2 dimensions
    Loading domain in cylindrical coordinates:
    	Variable phi: 384 [[0, -3.1334114227210694], [-1, 3.1334114227210694]]
    	Variable r: 128 [[0, 0.408203125], [-1, 2.491796875]]
    	Variable z: 1 [[0, 0.0], [-1, 0.0]]
    Number of snapshots in output directory: 51
    Configuration variables and domains load into the object. See e.g. <sim>.vars


Load gas density from a given snapshot:


```python
gasdens = sim.load_field('gasdens',snapshot=20)
```

Create a `meshslice` of the field:


```python
gasdens_r, mesh = gasdens.meshslice(slice='z=0,phi=0')
```

And plot the slice:


```python
import matplotlib.pyplot as plt
plt.ioff() # Drop this out of this tutorial
fig,ax = plt.subplots()
ax.semilogy(mesh.r,gasdens_r)
ax.set_xlabel(r"$r$ [cu]")
ax.set_ylabel(r"$\rho$ [cu]")
fp.Util.fargopy_mark(ax)
fig.savefig('gallery/example-dens_r.png') # Drop this out of this tutorial
```

<p align="center"><img src="https://github.com/seap-udea/fargopy/blob/main/gallery/example-dens_r.png?raw=true" alt="Animation""/></p>

You may also create a 2-dimensional slice:


```python
gasdens_plane, mesh = gasdens.meshslice(slice='z=0')
```

And plot it:


```python
plt.ioff() # Drop this out of this tutorial
fig,axs = plt.subplots(1,2,figsize=(12,6))

ax = axs[0]
ax.pcolormesh(mesh.phi,mesh.r,gasdens_plane,cmap='prism')
ax.set_xlabel('$\phi$ [rad]')
ax.set_ylabel('$r$ [UL]')
fp.Util.fargopy_mark(ax)

ax = axs[1]
ax.pcolormesh(mesh.x,mesh.y,gasdens_plane,cmap='prism')
ax.set_xlabel('$x$ [UL]')
ax.set_ylabel('$y$ [UL]')
fp.Util.fargopy_mark(ax)
ax.axis('equal')
fig.savefig('gallery/example-dens_disk.png') # Drop this out of this tutorial
```

<p align="center"><img src="https://github.com/seap-udea/fargopy/blob/main/gallery/example-dens_disk.png?raw=true" alt="Animation""/></p>

## What's new


Version 0.3.*:

- Refactoring of initializing routines.
- Improvements in documentation of basic classes in `__init__.py`.

Version 0.2.*:

- First real applications tested with FARGOpy.
- All basic routines for reading output created.
- Major refactoring. 

Version 0.1.*:

- Package is now provided with a script 'ifargopy' to run 'ipython' with fargopy initialized.
- A new 'progress' mode has been added to status method.
- All the dynamics of loading/compiling/running/stoppìng/resuming FARGO3D has been developed.

Version 0.0.*:

- First classes created.
- The project is started!

------------

This package has been designed and written mostly by Jorge I. Zuluaga with advising and contributions by Matías Montesinos (C) 2023

