# FARGOpy

## Tutorial

We will need the following external modules for this example:


```python
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML

import fargopy as fp
from fargopy import DEG, RAD
%load_ext autoreload
%autoreload 2
```

    Running FARGOpy version 0.2.1


There are two main functionalities of `FARGOpy`:

- **Output postprocessing**: `FARGOpy` allows you to load the outputs of a simulation with `FARGO3D` and postprocess the resulting data. You may to calculate derivative quantities, plot the fields or slices of them or any other computation.

- **FARGO3D Control**: `FARGOpy` allows you to download, compile, run and control `FARGO3D`. These functionalities could be very useful if you are starting using `FARGO3D`, you do not want to bother with Linux commands or you are using the package in `Google Colab`.

- **Simulation configuration**: `FARGOpy` is also intended to create or modify `FARGO3D` setups.

### Results postprocessing

For illustrating the usage of `FARGOpy` let's get some precomputed examples:


```python
fp.Simulation.download_precomputed(setup='fargo')
```

    Precomputed output directory '/tmp/fargo' already exist





    '/tmp/fargo'



Any operation in `fargopy` requires the creation of a simulation:


```python
sim = fp.Simulation()
```

    Your simulation is now connected with '/home/jzuluaga/fargo3d/'


Set the directory where the outputs are located: 


```python
sim.set_output_dir('/tmp/fargo')
```

    Now you are connected with output directory '/tmp/fargo'


List the files available in the output directory:


```python
outputs = sim.list_outputs()
```

    273 files in output directory
    IDL.var, bigplanet0.dat, dims.dat, domain_x.dat, domain_y.dat, domain_z.dat, fargo, gasdens0.dat, gasdens0_2d.dat, gasdens1.dat, 
    gasdens10.dat, gasdens11.dat, gasdens12.dat, gasdens13.dat, gasdens14.dat, gasdens15.dat, gasdens16.dat, gasdens17.dat, gasdens18.dat, gasdens19.dat, 
    gasdens2.dat, gasdens20.dat, gasdens21.dat, gasdens22.dat, gasdens23.dat, gasdens24.dat, gasdens25.dat, gasdens26.dat, gasdens27.dat, gasdens28.dat, 
    gasdens29.dat, gasdens3.dat, gasdens30.dat, gasdens31.dat, gasdens32.dat, gasdens33.dat, gasdens34.dat, gasdens35.dat, gasdens36.dat, gasdens37.dat, 
    gasdens38.dat, gasdens39.dat, gasdens4.dat, gasdens40.dat, gasdens41.dat, gasdens42.dat, gasdens43.dat, gasdens44.dat, gasdens45.dat, gasdens46.dat, 
    gasdens47.dat, gasdens48.dat, gasdens49.dat, gasdens5.dat, gasdens50.dat, gasdens6.dat, gasdens7.dat, gasdens8.dat, gasdens9.dat, gasenergy0.dat, 
    gasenergy1.dat, gasenergy10.dat, gasenergy11.dat, gasenergy12.dat, gasenergy13.dat, gasenergy14.dat, gasenergy15.dat, gasenergy16.dat, gasenergy17.dat, gasenergy18.dat, 
    gasenergy19.dat, gasenergy2.dat, gasenergy20.dat, gasenergy21.dat, gasenergy22.dat, gasenergy23.dat, gasenergy24.dat, gasenergy25.dat, gasenergy26.dat, gasenergy27.dat, 
    gasenergy28.dat, gasenergy29.dat, gasenergy3.dat, gasenergy30.dat, gasenergy31.dat, gasenergy32.dat, gasenergy33.dat, gasenergy34.dat, gasenergy35.dat, gasenergy36.dat, 
    gasenergy37.dat, gasenergy38.dat, gasenergy39.dat, gasenergy4.dat, gasenergy40.dat, gasenergy41.dat, gasenergy42.dat, gasenergy43.dat, gasenergy44.dat, gasenergy45.dat, 
    gasenergy46.dat, gasenergy47.dat, gasenergy48.dat, gasenergy49.dat, gasenergy5.dat, gasenergy50.dat, gasenergy6.dat, gasenergy7.dat, gasenergy8.dat, gasenergy9.dat, 
    gasvx0.dat, gasvx0_2d.dat, gasvx1.dat, gasvx10.dat, gasvx11.dat, gasvx12.dat, gasvx13.dat, gasvx14.dat, gasvx15.dat, gasvx16.dat, 
    gasvx17.dat, gasvx18.dat, gasvx19.dat, gasvx2.dat, gasvx20.dat, gasvx21.dat, gasvx22.dat, gasvx23.dat, gasvx24.dat, gasvx25.dat, 
    gasvx26.dat, gasvx27.dat, gasvx28.dat, gasvx29.dat, gasvx3.dat, gasvx30.dat, gasvx31.dat, gasvx32.dat, gasvx33.dat, gasvx34.dat, 
    gasvx35.dat, gasvx36.dat, gasvx37.dat, gasvx38.dat, gasvx39.dat, gasvx4.dat, gasvx40.dat, gasvx41.dat, gasvx42.dat, gasvx43.dat, 
    gasvx44.dat, gasvx45.dat, gasvx46.dat, gasvx47.dat, gasvx48.dat, gasvx49.dat, gasvx5.dat, gasvx50.dat, gasvx6.dat, gasvx7.dat, 
    gasvx8.dat, gasvx9.dat, gasvy0.dat, gasvy0_2d.dat, gasvy1.dat, gasvy10.dat, gasvy11.dat, gasvy12.dat, gasvy13.dat, gasvy14.dat, 
    gasvy15.dat, gasvy16.dat, gasvy17.dat, gasvy18.dat, gasvy19.dat, gasvy2.dat, gasvy20.dat, gasvy21.dat, gasvy22.dat, gasvy23.dat, 
    gasvy24.dat, gasvy25.dat, gasvy26.dat, gasvy27.dat, gasvy28.dat, gasvy29.dat, gasvy3.dat, gasvy30.dat, gasvy31.dat, gasvy32.dat, 
    gasvy33.dat, gasvy34.dat, gasvy35.dat, gasvy36.dat, gasvy37.dat, gasvy38.dat, gasvy39.dat, gasvy4.dat, gasvy40.dat, gasvy41.dat, 
    gasvy42.dat, gasvy43.dat, gasvy44.dat, gasvy45.dat, gasvy46.dat, gasvy47.dat, gasvy48.dat, gasvy49.dat, gasvy5.dat, gasvy50.dat, 
    gasvy6.dat, gasvy7.dat, gasvy8.dat, gasvy9.dat, monitor, orbit0.dat, outputgas.dat, planet0.dat, planets, summary0.dat, 
    summary1.dat, summary10.dat, summary11.dat, summary12.dat, summary13.dat, summary14.dat, summary15.dat, summary16.dat, summary17.dat, summary18.dat, 
    summary19.dat, summary2.dat, summary20.dat, summary21.dat, summary22.dat, summary23.dat, summary24.dat, summary25.dat, summary26.dat, summary27.dat, 
    summary28.dat, summary29.dat, summary3.dat, summary30.dat, summary31.dat, summary32.dat, summary33.dat, summary34.dat, summary35.dat, summary36.dat, 
    summary37.dat, summary38.dat, summary39.dat, summary4.dat, summary40.dat, summary41.dat, summary42.dat, summary43.dat, summary44.dat, summary45.dat, 
    summary46.dat, summary47.dat, summary48.dat, summary49.dat, summary5.dat, summary50.dat, summary6.dat, summary7.dat, summary8.dat, summary9.dat, 
    tqwk0.dat, used_rad.dat, variables.par, 


The files describing the basic properties of the simulations are `dims.dat`, `variables.par` and `domain_*.dat`. You may load the information in these files:


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


#### Load field data into memory

The outputs of a simulation are given as datafiles containing the value of different fields in the coordinate grid. You can load a single field:


```python
gasdens0 = sim.load_field('gasdens',snapshot=0)
gasdens0, gasdens0.data.shape
```




    ([[[0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]
       [0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]
       [0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]
       ...
       [0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]
       [0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]
       [0.00063662 0.00063662 0.00063662 ... 0.00063662 0.00063662 0.00063662]]],
     (1, 128, 384))



As you can see, fields are loaded as special `Field` objects (see `fp.Field?` for a list of attributes and methods), whose most important attribute is `data` which is the `numpy` array containing the values of the field in the coordinate domain.

Vectorial fields are special cases. In `FARGO3D` each component of the field is separated by file suffixes such as `x`, `y` and `z` (even if you are working in different coordinate systems). `fargopy` is able to load all components if a vector field using:


```python
vel = sim.load_field('gasv',snapshot=0,type='vector')
vel.data.shape
```




    (2, 1, 128, 384)



As you can see, the first index correspond to the component of the field (`x` and `y` in the `FARGO3D` convention, but actually `phi` and `r` in cylindrical coordinates). The second index is the `z` coordinate, the third the `y` coordinate (`r` in the cylindrical system of coordinates) and the fourh is the `z` coordinate (`phi` in the cylindrical system of coordinates).

Depending on the size of the outputs, you can also load all physical fields in the output associated to a given fluid. Use this method with caution:


```python
fields0 = sim.load_allfields('gas',snapshot=0)
fields0.keys(), fields0.size
```




    (['gasdens', 'gasenergy', 'gasvx', 'gasvy', 'size'], 1.5)



Size here is given in Megabytes.

If you want all the fields drop the `snapshot` option (or set in `None`):


```python
fields = sim.load_allfields('gas')
fields.print_keys()
fields.item('0').keys(), fields.size
```

    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 
    18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 
    27, 28, 29, 3, 30, 31, 32, 33, 34, 35, 
    36, 37, 38, 39, 4, 40, 41, 42, 43, 44, 
    45, 46, 47, 48, 49, 5, 50, 6, 7, 8, 
    9, snapshots, size





    (['gasdens', 'gasenergy', 'gasvx', 'gasvy'], 76.5)




```python
fields.item('0').gasdens.data.shape
```




    (1, 128, 384)



As you may see, the size of the fields start to be considerable large (36 MB in this case), so it is important to not abusing of this command.

#### Field slices

Once you have loaded a given field you may want to extract a slice for plotting. Let's for instance plot the density as a function of distance in the example simulation at a given snapshot:


```python
gasdens10 = sim.load_field('gasdens',snapshot=10)
gasdens10.data.shape
```




    (1, 128, 384)



Let's extract the density of the gas at `phi=0` and `z=0`:


```python
gasdens_r = gasdens10.slice(phi=0,z=0)
gasdens_r.shape
```




    (128,)



And plot:


```python
fig,ax = plt.subplots()
ax.semilogy(sim.domains.r,gasdens_r)
ax.set_xlabel(r"$r$ [cu]")
ax.set_ylabel(r"$\rho$ [cu]")
fp.Util.fargopy_mark(ax);
```


    
![png](TUTORIAL_files/TUTORIAL_36_0.png)
    


We can do this in a single step with `fargopy`:


```python
gasdens, mesh = gasdens10.meshslice(slice='z=0,phi=0')
```

The object `mesh` now contains matrices of the coordinates:


```python
mesh.keys()
```




    ['r', 'phi', 'x', 'y', 'z']



If you are plotting `r` vs. `gasdens` the plot will be:


```python
fig,ax = plt.subplots()
ax.semilogy(mesh.r,gasdens)
ax.set_xlabel(r"$r$ [cu]")
ax.set_ylabel(r"$\rho$ [cu]")
fp.Util.fargopy_mark(ax)
```




    Text(1, 1, 'FARGOpy 0.2.1')




    
![png](TUTORIAL_files/TUTORIAL_42_1.png)
    


This simple procedure reduce considerably the creation of more complex plots, for instance, a map of the density in different planes:


```python
gasdens, mesh = gasdens10.meshslice(slice='z=0')
```

And plot it:


```python
fig,axs = plt.subplots(1,2,figsize=(12,6))

ax = axs[0]
ax.pcolormesh(mesh.phi*RAD,mesh.r*sim.UL/fp.AU,gasdens,cmap='prism')
ax.set_xlabel('$\phi$ [deg]')
ax.set_ylabel('$r$ [au]')
fp.Util.fargopy_mark(ax)

ax = axs[1]
ax.pcolormesh(mesh.x*sim.UL/fp.AU,mesh.y*sim.UL/fp.AU,
              gasdens,cmap='prism')
ax.set_xlabel('$x$ [au]')
ax.set_ylabel('$y$ [au]')
fp.Util.fargopy_mark(ax)
ax.axis('equal')
```




    (-2.5002509788199982, 2.499916334793502, -2.499916334793502, 2.499916334793502)




    
![png](TUTORIAL_files/TUTORIAL_46_1.png)
    


Let's create an animation for illustrating how easy `FARGOpy` make life:


```python
from celluloid import Camera
from tqdm import tqdm

sim = fp.Simulation()
sim.set_output_dir('/tmp/fargo')
sim.load_properties()

gasdens_all = sim.load_allfields('gasdens')
fig,axs = plt.subplots(1,2,figsize=(12,6))

cmap = 'prism'
camera = Camera(fig)
for snapshot in tqdm(gasdens_all.snapshots):
    gasdens_snap = gasdens_all.item(str(snapshot)).gasdens
    gasdens,mesh = gasdens_snap.meshslice(slice='z=0')

    ax = axs[0]
    ax.pcolormesh(mesh.phi*RAD,mesh.r*sim.UL/fp.AU,gasdens,cmap=cmap)
    ax.set_xlabel('$\phi$ [deg]')
    ax.set_ylabel('$r$ [au]')

    ax = axs[1]
    ax.pcolormesh(mesh.x*sim.UL/fp.AU,mesh.y*sim.UL/fp.AU,gasdens,cmap=cmap)
    ax.set_xlabel('$x$ [au]')
    ax.set_ylabel('$y$ [au]')
    fp.Util.fargopy_mark(ax)

    camera.snap()

animation = camera.animate()
animation.save('gallery/fargo-animation.gif')
```

    Your simulation is now connected with '/home/jzuluaga/fargo3d/'
    Now you are connected with output directory '/tmp/fargo'
    Loading variables
    84 variables loaded
    Simulation in 2 dimensions
    Loading domain in cylindrical coordinates:
    	Variable phi: 384 [[0, -3.1334114227210694], [-1, 3.1334114227210694]]
    	Variable r: 128 [[0, 0.408203125], [-1, 2.491796875]]
    	Variable z: 1 [[0, 0.0], [-1, 0.0]]
    Number of snapshots in output directory: 51
    Configuration variables and domains load into the object. See e.g. <sim>.vars


      0%|          | 0/51 [00:00<?, ?it/s]100%|██████████| 51/51 [00:00<00:00, 60.30it/s]
    MovieWriter ffmpeg unavailable; using Pillow instead.



    
![png](TUTORIAL_files/TUTORIAL_48_2.png)
    


### Control

Before starting you need a local copy of `FARGO3D`. For getting a copy use:


```python
fp.initialize('download')
```

    Downloading FARGOpy...
    	FARGO3D directory already present in '/home/jzuluaga/fargo3d/'
    Header file for FARGO3D is in the fargo directory /home/jzuluaga/fargo3d/


Create the simulation object:


```python
sim = fp.Simulation()
```

    Your simulation is now connected with '/home/jzuluaga/fargo3d/'


> **NOTE**: If you already have a copy of `FARGO3D` you may configure the location using the method of the simulation object `sim.set_fargo3d_dir(<your_fargo3d_location>)`


```python
sim.set_fargo3d_dir('/home/jzuluaga/fargo3d')
```

    Your simulation is now connected with '/home/jzuluaga/fargo3d'


Once you have connected your simulation you must choose your setup:


```python
sim.set_setup('fargo')
```

    Now your simulation setup is at '/home/jzuluaga/fargo3d/setups/fargo'





    'fargo'



Now you can check the properties of the simulation:


```python
print(sim)
```

    Simulation information:
        FARGO3D directory: /home/jzuluaga/fargo3d
            Outputs: /home/jzuluaga/fargo3d/outputs
            Setups: /home/jzuluaga/fargo3d/setups
        Units:
            G = 1 UL^3/(UM UT^2)
            UL, UM, UT = 14959800000000.0 m, 1.9891e+33 kg, 5022421.439871778 s
            UE = 5271139.687948494 J/m^3
            UV = 2978603.086001067 m/s
            URHO = 5.941269533317109e-07 kg/m^3
            USIGMA = 8888020.39645173 kg/m^2
        Setup: fargo
        Setup directory: /home/jzuluaga/fargo3d/setups/fargo
        Output directory: None
    


You may abbreviate the previous commands using from the beginning:


```python
sim = fp.Simulation(fargo3d_dir='/home/jzuluaga/fargo3d',setup='fargo')
sim
```

    Your simulation is now connected with '/home/jzuluaga/fargo3d'
    Now your simulation setup is at '/home/jzuluaga/fargo3d/setups/fargo'





    Simulation information:
        FARGO3D directory: /home/jzuluaga/fargo3d
            Outputs: /home/jzuluaga/fargo3d/outputs
            Setups: /home/jzuluaga/fargo3d/setups
        Units:
            G = 1 UL^3/(UM UT^2)
            UL, UM, UT = 14959800000000.0 m, 1.9891e+33 kg, 5022421.439871778 s
            UE = 5271139.687948494 J/m^3
            UV = 2978603.086001067 m/s
            URHO = 5.941269533317109e-07 kg/m^3
            USIGMA = 8888020.39645173 kg/m^2
        Setup: fargo
        Setup directory: /home/jzuluaga/fargo3d/setups/fargo
        Output directory: None



Now you can compile `FARGO3D` using a given setup:


```python
sim.compile(setup='fargo',parallel=0,gpu=0,force=True)
```

    Now your simulation setup is at '/home/jzuluaga/fargo3d/setups/fargo'
    Cleaning FARGO3D directory /home/jzuluaga/fargo3d...
    Compiling fargo3d_SETUP-fargo_PARALLEL-0_GPU-0...
    Succesful compilation of FARGO3D binary fargo3d_SETUP-fargo_PARALLEL-0_GPU-0





    True



If successful, you may run the simulation:


```python
sim.run(cleanrun=True)
```

    Cleaning output directory /home/jzuluaga/fargo3d/outputs/fargo
    Running asynchronously (test = False):  ./fargo3d_SETUP-fargo_PARALLEL-0_GPU-0 -m -t setups/fargo/fargo.par
    Now you are connected with output directory '/home/jzuluaga/fargo3d/outputs/fargo'


Once running, check the status:


```python
sim.status()
```

    
    ################################################################################
    Running status of the process:
    	The process is running.


There are several ways to check the status. This way will allow you to check the status in real time:


```python
sim.status('progress')
```

    OUTPUTS 3 at date t = 18.849556 OK [output pace = 0.1 secs]
    OUTPUTS 4 at date t = 25.132741 OK [output pace = 0.1 secs]
    OUTPUTS 5 at date t = 31.415927 OK [output pace = 0.9 secs]
    OUTPUTS 6 at date t = 37.699112 OK [output pace = 1.9 secs]
    OUTPUTS 7 at date t = 43.982297 OK [output pace = 1.9 secs]
    OUTPUTS 8 at date t = 50.265482 OK [output pace = 1.9 secs]
    OUTPUTS 9 at date t = 56.548668 OK [output pace = 1.9 secs]
    OUTPUTS 10 at date t = 62.831853 OK [output pace = 1.9 secs]
    OUTPUTS 11 at date t = 69.115038 OK [output pace = 1.9 secs]
    OUTPUTS 12 at date t = 75.398224 OK [output pace = 1.9 secs]
    OUTPUTS 13 at date t = 81.681409 OK [output pace = 1.0 secs]
    OUTPUTS 14 at date t = 87.964594 OK [output pace = 2.0 secs]
    OUTPUTS 15 at date t = 94.247780 OK [output pace = 2.0 secs]
    OUTPUTS 16 at date t = 100.530965 OK [output pace = 2.0 secs]
    OUTPUTS 17 at date t = 106.814150 OK [output pace = 2.0 secs]
    OUTPUTS 18 at date t = 113.097336 OK [output pace = 2.0 secs]
    OUTPUTS 19 at date t = 119.380521 OK [output pace = 1.0 secs]
    OUTPUTS 20 at date t = 125.663706 OK [output pace = 1.5 secs]


You may stop the simulation:


```python
sim.stop()
```

    Stopping FARGO3D process (pid = 9809)


Once stopped you may resume it:


```python
sim.resume()
```

    The process has ended with termination code 0.



```python
sim.status('progress')
```

    The simulation is not running anymore


This will resume the simulation from the *latest resumable snapshot*. You may also resume from a different snapshot using: 
