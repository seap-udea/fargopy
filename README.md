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

# Set in True to see all messages
fp.Conf.VERBOSE = False

# For developing purpose. Remove during production
%load_ext autoreload
%autoreload 2
```

    Running FARGOpy version 0.1.0


## Install FARGO3D

For obvious reasons `FARGO3D` is not provided with `FARGOpy`. Get and set the package using:


```python
fp.Conf.set_fargo3d()
```

    > Checking for FARGO3D directroy:
    	✓FARGO3D source code is available in your system at './public/'
    > Checking for FARGO3D normal binary:
    	✓Binary in normal mode compiling correctly
    > Checking for FARGO3D parallel binary:
    	✓Binary in parallel mode compiling correctly
    > Checking for FARGO3D GPU binary:
    FARGO3D binary with options 'GPU=1' not compiled at './public/'
    	Compiling FARGO3D with GPU (it may take a while)...
    	No GPU available


Once set, check configuration:


```python
fp.Conf.show_fargo3d_configuration()
```

    Is FARGO3D installed:  True
    Is FARGO3D compiling:  True
    Is FARGO3D compiling in parallel:  False
    Is FARGO3D compiling in GPU:  False
    FARGO3D clone repositoty command:  git clone https://bitbucket.org/fargo3d/public.git
    FARGO3D directories: 
    	Base directory:  ./
    	Package directory:  public/
    	Basic package header:  src/fargo3d.h
    	Setups location:  setups/
    	Setups location:  ./public/
    Compile in parallel:  1
    Compile in GPU:  0


If you already have a copy of `FARGO3D` you just need to set the configuration variables telling `FARGOpy` where the source code is located:


```python
fp.Conf.configure_fargo3d(basedir='/tmp',packdir='public',parallel=1,gpu=0)

```

Once the variables has been set, you should set the package:


```python
fp.Conf.set_fargo3d()
fp.Conf.show_fargo3d_configuration()
```

    > Checking for FARGO3D directroy:
    	✓FARGO3D source code is available in your system at '/tmp/public/'
    > Checking for FARGO3D normal binary:
    	✓Binary in normal mode compiling correctly
    > Checking for FARGO3D parallel binary:
    	✓Binary in parallel mode compiling correctly
    > Checking for FARGO3D GPU binary:
    FARGO3D binary with options 'GPU=1' not compiled at '/tmp/public/'
    	Compiling FARGO3D with GPU (it may take a while)...
    	No GPU available
    Is FARGO3D installed:  True
    Is FARGO3D compiling:  True
    Is FARGO3D compiling in parallel:  False
    Is FARGO3D compiling in GPU:  False
    FARGO3D clone repositoty command:  git clone https://bitbucket.org/fargo3d/public.git
    FARGO3D directories: 
    	Base directory:  /tmp/
    	Package directory:  public/
    	Basic package header:  src/fargo3d.h
    	Setups location:  setups/
    	Setups location:  /tmp/public/
    Compile in parallel:  1
    Compile in GPU:  0


## Quickstart

The first thing you need to run `FARGOpy` is to create a simulation:


```python
sim = fp.Simulation()
```

`FARGO3D` comes along with several test simulations configured as `setups`. You may list which setups are available using:




```python
sim.list_setups()
```




    ['binary',
     'fargo',
     'fargo_multifluid',
     'fargo_nu',
     'mri',
     'otvortex',
     'p3diso',
     'p3disof',
     'sod1d']



Once you select the setup, you may load it into the simulation:


```python
sim.load_setup('fargo')
```

Alternatively, you may create the simulation from the beginning using the `setup` option:


```python
sim = fp.Simulation(setup='fargo')
```

Before start, it's a good idea to clean all outputs (if they exists):


```python
sim.clean_output()
```

    Cleaning simulation outputs...
    Done.


You may compile the simulation:


```python
sim.compile(force=True)
```

    Compiling FARGO3D with options 'SETUP=fargo PARALLEL=1 GPU=0 ' (it may take a while... go for a coffee)


Once you have loaded the setup we may proceed compiling and running the simulation:


```python
sim.run()
```

    Running asynchronously: mpirun -np 1 ./.fargo3d_SETUP-fargo_PARALLEL-1_GPU-0_ -m -t setups/fargo/fargo.par
    Command is running in background


You can check the status:


```python
sim.status(mode='isrunning')
```

    
    ################################################################################
    Running status of the process:
    	The process is running.


There are several status modes:

- Show progress of the simulation


```python
sim.status(mode='progress')
```

    OUTPUTS 0 at date t = 0.000000 OK [output pace = 0.1 secs]
    OUTPUTS 1 at date t = 6.283185 OK [output pace = 0.1 secs]
    OUTPUTS 2 at date t = 12.566371 OK [output pace = 1.5 secs]
    OUTPUTS 3 at date t = 18.849556 OK [output pace = 3.0 secs]
    OUTPUTS 4 at date t = 25.132741 OK [output pace = 3.0 secs]
    OUTPUTS 5 at date t = 31.415927 OK [output pace = 3.0 secs]


- Check the latest lines of the `logfile`: 


```python
sim.status(mode='logfile')
```

    
    ################################################################################
    Logfile content:
    The latest 10 lines of the logfile:
    
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ...

- Check (and return) the output files:


```python
sim.status(mode='outputs')
```

    
    ################################################################################
    Output content:
    
    58 available datafiles:
    
    bigplanet0.dat, dims.dat, domain_x.dat, domain_y.dat, domain_z.dat, gasdens0.dat, gasdens0_2d.dat, gasdens1.dat, gasdens2.dat, gasdens3.dat, 
    gasdens4.dat, gasdens5.dat, gasdens6.dat, gasdens7.dat, gasdens8.dat, gasenergy0.dat, gasenergy1.dat, gasenergy2.dat, gasenergy3.dat, gasenergy4.dat, 
    gasenergy5.dat, gasenergy6.dat, gasenergy7.dat, gasenergy8.dat, gasvx0.dat, gasvx0_2d.dat, gasvx1.dat, gasvx2.dat, gasvx3.dat, gasvx4.dat, 
    gasvx5.dat, gasvx6.dat, gasvx7.dat, gasvx8.dat, gasvy0.dat, gasvy0_2d.dat, gasvy1.dat, gasvy2.dat, gasvy3.dat, gasvy4.dat, 
    gasvy5.dat, gasvy6.dat, gasvy7.dat, gasvy8.dat, orbit0.dat, outputgas.dat, planet0.dat, summary0.dat, summary1.dat, summary2.dat, 
    summary3.dat, summary4.dat, summary5.dat, summary6.dat, summary7.dat, summary8.dat, tqwk0.dat, used_rad.dat, 


- Check the available snapshots:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 10
    	Latest resumable snapshot: 8


You may also combine them:


```python
sim.status(mode='isrunning snapshots')
```

    
    ################################################################################
    Running status of the process:
    	The process is running.
    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 11
    	Latest resumable snapshot: 9


Or ran all of them:


```python
sim.status(mode='all')
```

    
    ################################################################################
    Running status of the process:
    	The process is running.
    
    ################################################################################
    Logfile content:
    The latest 10 lines of the logfile:
    
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    ..............
    
    ################################################################################
    Output content:
    
    68 available datafiles:
    
    bigplanet0.dat, dims.dat, domain_x.dat, domain_y.dat, domain_z.dat, gasdens0.dat, gasdens0_2d.dat, gasdens1.dat, gasdens10.dat, gasdens2.dat, 
    gasdens3.dat, gasdens4.dat, gasdens5.dat, gasdens6.dat, gasdens7.dat, gasdens8.dat, gasdens9.dat, gasenergy0.dat, gasenergy1.dat, gasenergy10.dat, 
    gasenergy2.dat, gasenergy3.dat, gasenergy4.dat, gasenergy5.dat, gasenergy6.dat, gasenergy7.dat, gasenergy8.dat, gasenergy9.dat, gasvx0.dat, gasvx0_2d.dat, 
    gasvx1.dat, gasvx10.dat, gasvx2.dat, gasvx3.dat, gasvx4.dat, gasvx5.dat, gasvx6.dat, gasvx7.dat, gasvx8.dat, gasvx9.dat, 
    gasvy0.dat, gasvy0_2d.dat, gasvy1.dat, gasvy10.dat, gasvy2.dat, gasvy3.dat, gasvy4.dat, gasvy5.dat, gasvy6.dat, gasvy7.dat, 
    gasvy8.dat, gasvy9.dat, orbit0.dat, outputgas.dat, planet0.dat, summary0.dat, summary1.dat, summary10.dat, summary2.dat, summary3.dat, 
    summary4.dat, summary5.dat, summary6.dat, summary7.dat, summary8.dat, summary9.dat, tqwk0.dat, used_rad.dat, 
    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 11
    	Latest resumable snapshot: 9


You may stop the run at any time:


```python
sim.stop()
```

    The process is already running with pid '30287'
    Stopping FARGO3D process (pid = 30287)


Before resuming get the latest resumable snapshot:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 13
    	Latest resumable snapshot: 11


And resume since the latest resumable snapshot:


```python
sim.resume(since=sim.resumable_snapshot)
```

    Resuming from snapshot 11
    Running asynchronously: mpirun -np 1 ./.fargo3d_SETUP-fargo_PARALLEL-1_GPU-0_ -m -t -S 11 -t setups/fargo/fargo.par
    Command is running in background


Check the status again:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 16
    	Latest resumable snapshot: 12


Once the running is finished you will get:


```python
sim.status(mode='isrunning')
```

    
    ################################################################################
    Running status of the process:
    	The process has ended with termination code 0.


## What's new


Version 0.1.*:

- Package is now provided with a script 'ifargopy' to run 'ipython' with fargopy initialized.
- A new 'progress' mode has been added to status method.
- All the dynamics of loading/compiling/running/stoppìng/resuming FARGO3D has been developed.

Version 0.0.*:

- First classes created.
- The project is started!

------------

This package has been designed and written mostly by Jorge I. Zuluaga with advising and contributions by Matías Montesinos (C) 2023

