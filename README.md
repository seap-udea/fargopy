# FARGOpy
## Wrapping FRAGO3D

<!-- This are visual tags that you may add to your package at the beginning with useful information on your package --> 
[![version](https://img.shields.io/pypi/v/fargopy?color=blue)](https://pypi.org/project/fargopy/)
[![downloads](https://img.shields.io/pypi/dw/fargopy)](https://pypi.org/project/fargopy/)

`FARGOpy` is a python wrapping for [`FARGO3D`](https://fargo3d.bitbucket.io/intro.html)., the well-knwon hydrodynamics and magnetohydrodynamics parallel code. This wrapping is intended to ensue the interaction with FARGO3D especially for those starting using the code, for instance for teaching and training purposes, but also provide functionalities for most advanced users in tasks related to the postprocessing of output files and plotting.

## Download and install

For using `FARGOpy` you first need to download and install `FARGO3D` and all its prerequisites. For a detailed guide please see the [FARGO documentation](https://fargo3d.bitbucket.io/index.html) or the [project repo at bitbucket](https://bitbucket.org/fargo3d/public/src/ae0fcdc67bb7c83aed85fc9a4d4a2d5061324597/?at=release%2Fpublic). Still, `FARGOpy` provides some useful commands and tools to test the platform on which you are working and check if it is prepared to use the whole functionalities of the packages or part of them.

> **NOTE**: It is important to understand that `FARGO3D` works especially well on Linux plaforms (including `MacOS`). The same condition applies for `FARGOpy`. Because of that, most internal as well as public features of the packages are designed to work in a `Linux` environment. For working in another operating systems, for instance for teaching or training purposes, please consider to use virtual machines.

For installing `FARGOpy` run:


```python
# Uncomment if you are in Google Colab
# !pip install -q fargopy
```

Once installed import it:


```python
import fargopy as fp

# For developing purpose. Remove during production
%load_ext autoreload
%autoreload 2
```

    Running FARGOpy version 0.0.1


## Quickstart

The first thing you need to run `FARGOpy` is to create a simulation:


```python
sim = fp.Simulation()
```

### Install FARGO3D

For obvious reasons `FARGO3D` is not provided with `FARGOpy`. If you already have a copy of the package in your computer you just need to configure your copy setting the path directory where the copy is:


```python
fp.Conf.update_fargo3d_dir('/tmp','public')
```

If you don't have a copy you may want to get one using:


```python
sim.set_fargo3d(basedir='.')
```

    Running FARGOpy version 0.0.1
    > Checking for FARGO3D directroy:
    FARGO3D source code is not available at './public/'
    	Getting FARGO3D public repo...
    	✓Package downloaded to './public/'
    > Checking for FARGO3D normal binary:
    FARGO3D binary with options '' not compiled at './public/'
    	Compiling FARGO3D (it may take a while)...
    	✓Binary in normal mode compiling correctly
    > Checking for FARGO3D parallel binary:
    FARGO3D binary with options 'PARALLEL=1' not compiled at './public/'
    	Compiling FARGO3D in parallel (it may take a while)...
    	✓Binary in parallel mode compiling correctly
    > Checking for FARGO3D GPU binary:
    FARGO3D binary with options 'GPU=1' not compiled at './public/'
    	Compiling FARGO3D with GPU (it may take a while)...
    	No GPU available


## What's new


Version 0.0.*:

- First classes created.
- The project is started!

------------

This package has been designed and written by Jorge I. Zuluaga and Matías Montesinos (C) 2023
