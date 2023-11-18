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

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


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
    Compiling FARGO3D with options 'GPU=1' (it may take a while... go for a coffee)
    	No GPU available


These are the critical configuration variables that must be setup before starting working with `FARGOpy`:


```python
print("FARGO3D is present: ",fp.Conf.FARGO3D_IS_HERE)
print("Is FARGO3D compiling: ",fp.Conf.FARGO3D_IS_COMPILING)
print("Is FARGO3D compiling in parallel: ",fp.Conf.FARGO3D_PARALLEL)
print("Is FARGO3D compiling in GPU: ",fp.Conf.FARGO3D_GPU)
```

    FARGO3D is present:  True
    Is FARGO3D compiling:  True
    Is FARGO3D compiling in parallel:  1
    Is FARGO3D compiling in GPU:  0


If you already have a copy of `FARGO3D` you just need to set the configuration variables telling `FARGOpy` where the source code is located:

```python
  fp.Conf.update_fargo3d_dir('/tmp','public')
  ```

once the variables has been set, you should run:

```python
  fp.Conf.set_fargo3d()
  ```

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
    Output directory is clean already.


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
    ...........

- Check (and return) the output files:


```python
sim.status(mode='outputs')
```

    
    ################################################################################
    Output content:
    
    118 available datafiles:
    
    bigplanet0.dat, dims.dat, domain_x.dat, domain_y.dat, domain_z.dat, gasdens0.dat, gasdens0_2d.dat, gasdens1.dat, gasdens10.dat, gasdens11.dat, 
    gasdens12.dat, gasdens13.dat, gasdens14.dat, gasdens15.dat, gasdens16.dat, gasdens17.dat, gasdens18.dat, gasdens19.dat, gasdens2.dat, gasdens20.dat, 
    gasdens3.dat, gasdens4.dat, gasdens5.dat, gasdens6.dat, gasdens7.dat, gasdens8.dat, gasdens9.dat, gasenergy0.dat, gasenergy1.dat, gasenergy10.dat, 
    gasenergy11.dat, gasenergy12.dat, gasenergy13.dat, gasenergy14.dat, gasenergy15.dat, gasenergy16.dat, gasenergy17.dat, gasenergy18.dat, gasenergy19.dat, gasenergy2.dat, 
    gasenergy20.dat, gasenergy3.dat, gasenergy4.dat, gasenergy5.dat, gasenergy6.dat, gasenergy7.dat, gasenergy8.dat, gasenergy9.dat, gasvx0.dat, gasvx0_2d.dat, 
    gasvx1.dat, gasvx10.dat, gasvx11.dat, gasvx12.dat, gasvx13.dat, gasvx14.dat, gasvx15.dat, gasvx16.dat, gasvx17.dat, gasvx18.dat, 
    gasvx19.dat, gasvx2.dat, gasvx20.dat, gasvx3.dat, gasvx4.dat, gasvx5.dat, gasvx6.dat, gasvx7.dat, gasvx8.dat, gasvx9.dat, 
    gasvy0.dat, gasvy0_2d.dat, gasvy1.dat, gasvy10.dat, gasvy11.dat, gasvy12.dat, gasvy13.dat, gasvy14.dat, gasvy15.dat, gasvy16.dat, 
    gasvy17.dat, gasvy18.dat, gasvy19.dat, gasvy2.dat, gasvy20.dat, gasvy3.dat, gasvy4.dat, gasvy5.dat, gasvy6.dat, gasvy7.dat, 
    gasvy8.dat, gasvy9.dat, orbit0.dat, outputgas.dat, planet0.dat, summary0.dat, summary1.dat, summary10.dat, summary11.dat, summary12.dat, 
    summary13.dat, summary14.dat, summary15.dat, summary16.dat, summary17.dat, summary18.dat, summary19.dat, summary2.dat, summary20.dat, summary3.dat, 
    summary4.dat, summary5.dat, summary6.dat, summary7.dat, summary8.dat, summary9.dat, tqwk0.dat, used_rad.dat, 


- Check the available snapshots:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 25
    	Latest resumable snapshot: 23


You may also combine them:


```python
sim.status(mode='isrunning snapshots')
```

    
    ################################################################################
    Running status of the process:
    	The process is running.
    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 30
    	Latest resumable snapshot: 28


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
    
    .............
    .............
    .............
    .............
    .............
    .............
    .............
    .............
    .............
    ....
    ################################################################################
    Output content:
    
    178 available datafiles:
    
    bigplanet0.dat, dims.dat, domain_x.dat, domain_y.dat, domain_z.dat, gasdens0.dat, gasdens0_2d.dat, gasdens1.dat, gasdens10.dat, gasdens11.dat, 
    gasdens12.dat, gasdens13.dat, gasdens14.dat, gasdens15.dat, gasdens16.dat, gasdens17.dat, gasdens18.dat, gasdens19.dat, gasdens2.dat, gasdens20.dat, 
    gasdens21.dat, gasdens22.dat, gasdens23.dat, gasdens24.dat, gasdens25.dat, gasdens26.dat, gasdens27.dat, gasdens28.dat, gasdens29.dat, gasdens3.dat, 
    gasdens30.dat, gasdens31.dat, gasdens32.dat, gasdens4.dat, gasdens5.dat, gasdens6.dat, gasdens7.dat, gasdens8.dat, gasdens9.dat, gasenergy0.dat, 
    gasenergy1.dat, gasenergy10.dat, gasenergy11.dat, gasenergy12.dat, gasenergy13.dat, gasenergy14.dat, gasenergy15.dat, gasenergy16.dat, gasenergy17.dat, gasenergy18.dat, 
    gasenergy19.dat, gasenergy2.dat, gasenergy20.dat, gasenergy21.dat, gasenergy22.dat, gasenergy23.dat, gasenergy24.dat, gasenergy25.dat, gasenergy26.dat, gasenergy27.dat, 
    gasenergy28.dat, gasenergy29.dat, gasenergy3.dat, gasenergy30.dat, gasenergy31.dat, gasenergy32.dat, gasenergy4.dat, gasenergy5.dat, gasenergy6.dat, gasenergy7.dat, 
    gasenergy8.dat, gasenergy9.dat, gasvx0.dat, gasvx0_2d.dat, gasvx1.dat, gasvx10.dat, gasvx11.dat, gasvx12.dat, gasvx13.dat, gasvx14.dat, 
    gasvx15.dat, gasvx16.dat, gasvx17.dat, gasvx18.dat, gasvx19.dat, gasvx2.dat, gasvx20.dat, gasvx21.dat, gasvx22.dat, gasvx23.dat, 
    gasvx24.dat, gasvx25.dat, gasvx26.dat, gasvx27.dat, gasvx28.dat, gasvx29.dat, gasvx3.dat, gasvx30.dat, gasvx31.dat, gasvx32.dat, 
    gasvx4.dat, gasvx5.dat, gasvx6.dat, gasvx7.dat, gasvx8.dat, gasvx9.dat, gasvy0.dat, gasvy0_2d.dat, gasvy1.dat, gasvy10.dat, 
    gasvy11.dat, gasvy12.dat, gasvy13.dat, gasvy14.dat, gasvy15.dat, gasvy16.dat, gasvy17.dat, gasvy18.dat, gasvy19.dat, gasvy2.dat, 
    gasvy20.dat, gasvy21.dat, gasvy22.dat, gasvy23.dat, gasvy24.dat, gasvy25.dat, gasvy26.dat, gasvy27.dat, gasvy28.dat, gasvy29.dat, 
    gasvy3.dat, gasvy30.dat, gasvy31.dat, gasvy32.dat, gasvy4.dat, gasvy5.dat, gasvy6.dat, gasvy7.dat, gasvy8.dat, gasvy9.dat, 
    orbit0.dat, outputgas.dat, planet0.dat, summary0.dat, summary1.dat, summary10.dat, summary11.dat, summary12.dat, summary13.dat, summary14.dat, 
    summary15.dat, summary16.dat, summary17.dat, summary18.dat, summary19.dat, summary2.dat, summary20.dat, summary21.dat, summary22.dat, summary23.dat, 
    summary24.dat, summary25.dat, summary26.dat, summary27.dat, summary28.dat, summary29.dat, summary3.dat, summary30.dat, summary31.dat, summary32.dat, 
    summary4.dat, summary5.dat, summary6.dat, summary7.dat, summary8.dat, summary9.dat, tqwk0.dat, used_rad.dat, 
    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 33
    	Latest resumable snapshot: 31


You may stop the run at any time:


```python
sim.stop()
```

    Stopping FARGO3D process (pid = 8084)


Before resuming get the latest resumable snapshot:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 37
    	Latest resumable snapshot: 35


And resume since the latest resumable snapshot:


```python
sim.resume(since=sim.resumable_snapshot)
```

    Resuming from snapshot 35
    Running asynchronously: mpirun -np 1 ./.fargo3d_SETUP-fargo_PARALLEL-1_GPU-0_ -m -t -S 35 -t setups/fargo/fargo.par
    Command is running in background


Check the status again:


```python
sim.status(mode='snapshots')
```

    
    ################################################################################
    Snapshots:
    	Number of available snapshots: 41
    	Latest resumable snapshot: 37


Once the running is finished you will get:


```python
sim.status(mode='isrunning')
```

    
    ################################################################################
    Running status of the process:
    	The process has ended with termination code 0.


## What's new


Version 0.1.*:

- All the dynamics of loading/compiling/running/stoppìng/resuming FARGO3D has been developed.

Version 0.0.*:

- First classes created.
- The project is started!

------------

This package has been designed and written mostly by Jorge I. Zuluaga with advising and contributions by Matía Montesinos (C) 2023

