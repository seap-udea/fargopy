###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import os
import json
import numpy as np
import re
import subprocess
import time
import gdown
import os
import signal
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from pathlib import Path

###############################################################
# Constants
###############################################################
KB = 1.380650424e-16  # Boltzmann constant: erg/K, erg = g cm^2 / s^2
MP = 1.672623099e-24 # Mass of the proton, g
GCONST = 6.67259e-8 # Gravitational constant, cm^3/g/s^2 
RGAS = 8.314472e7 # Gas constant, erg/K/mol
MSUN = 1.9891e33 # g
AU = 1.49598e13 # cm
YEAR = 31557600.0 # s

PRECOMPUTED_BASEURL = 'https://docs.google.com/uc?export=download&id='
PRECOMPUTED_SIMULATIONS = dict(
    # Download link: https://drive.google.com/file/d/1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3/view?usp=drive_link
    fargo=dict(
        id='1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3',
        description="""Protoplanetary disk with a Jovian planet [2D]""",
        size=55
    ),
    # Download link: https://drive.google.com/file/d/1KMp_82ylQn3ne_aNWEF1T9ElX2aWzYX6/view?usp=drive_link
    p3diso=dict(
        id='1KMp_82ylQn3ne_aNWEF1T9ElX2aWzYX6',
        description="""Protoplanetary disk with a Super earth planet [3D]""",
        size=220
    ),
    # Download link: https://drive.google.com/file/d/1Xzgk9qatZPNX8mLmB58R9NIi_YQUrHz9/view?usp=sharing
    p3disoj=dict(
        id='1Xzgk9qatZPNX8mLmB58R9NIi_YQUrHz9',
        description="""Protoplanetary disk with a Jovian planet [3D]""",
        size=84
    ),
    # Download link: https://drive.google.com/file/d/1KSQyxH_kbAqHQcsE30GQFRVgAPhMAcp7/view?usp=drive_link
    fargo_multifluid=dict(
        id='1KSQyxH_kbAqHQcsE30GQFRVgAPhMAcp7',
        description="""Protoplanetary disk with several fluids (dust) and a Jovian planet in 2D""",
        size=100
    ),
    # Download link: https://drive.google.com/file/d/12ZWoQS_9ISe6eDij5KWWbqR-bHyyVs2N/view?usp=drive_link
    binary=dict(
        id='12ZWoQS_9ISe6eDij5KWWbqR-bHyyVs2N',
        description="""Disk around a binary with the properties of Kepler-38 in 2D""",
        size=140
    ),
)

if not fargopy.IN_COLAB:
    # SIGCHLD is only available on Unix-like systems (Linux, macOS)
    if hasattr(signal, 'SIGCHLD'):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)

###############################################################
# Classes
###############################################################

class Simulation(fargopy.Fargobj):
    """
    Main class for managing a FARGOpy simulation. Handles setup, compilation, running, and data access
    for FARGO3D simulations, including units, directories, and output management.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments for simulation configuration. See examples below.

    Examples
    --------
    Create an empty simulation (no setup chosen):
        >>> sim = fargopy.Simulation()

    Create a simulation using a specific base FARGO3D directory (no setup chosen):
        >>> sim = fargopy.Simulation(fargo3d_dir='/tmp/public')

    Create a simulation starting with setup directory:
        >>> sim = fargopy.Simulation(setup='fargo')

    Connect to an existing simulation:
        >>> sim = fargopy.Simulation(output_dir='/tmp/public/outputs/fargo')

    Load an already existing simulation:
        >>> sim = fargopy.Simulation(setup='fargo', load=True)
    """
    def __init__(self,**kwargs):
        """
        Initialize a simulation instance and optionally bind it to an existing setup/output.

        Parameters
        ----------
        **kwargs : dict
            Accepted keys include:
            - ``setup``: name of the setup directory.
            - ``fargo3d_dir``: root directory where FARGO3D is installed.
            - ``output_dir``: path to an existing output directory.
            - ``load``: when True, load metadata from ``fargopy_simulation.json`` inside the setup.

        Examples
        --------
        Create an empty simulation:
            >>> sim = fargopy.Simulation()

        Point to a specific FARGO3D installation:
            >>> sim = fargopy.Simulation(fargo3d_dir='/tmp/public')

        Attach to a setup and load metadata:
            >>> sim = fargopy.Simulation(setup='fargo', load=True)

        Connect to previously generated outputs:
            >>> sim = fargopy.Simulation(output_dir='/tmp/public/outputs/fargo')
        """
        super().__init__(**kwargs)

        # Default to CGS units
        self.units_system = 'CGS'
        self._set_constants_cgs()

        # Load simulation configuration from a file
        if ('load' in kwargs.keys()) and kwargs['load']:
            
            if not 'setup' in kwargs.keys():
                raise AssertionError(f"You must provide a setup name.")
            else:
                setup = kwargs['setup']
            if 'fargo3d_dir' in kwargs.keys():
                fargo3d_dir = kwargs['fargo3d_dir']
            else:
                fargo3d_dir = fargopy.Conf.FP_FARGO3D_DIR
            
            # Load simulation
            load_from = os.path.join(fargo3d_dir, 'setups', setup)
            if not os.path.isdir(load_from):
                print(f"Directory for loading simulation '{load_from}' not found.")
            json_file = os.path.join(load_from, 'fargopy_simulation.json')

            print(f"Loading simulation from '{json_file}'")
            if not os.path.isfile(json_file):
                print(f"Loading data '{json_file}' not found.")

            with open(json_file) as file_handler:
                attributes = json.load(file_handler)

                # Check if there are not serializable items and set to None
                for key,item in attributes.items():
                    if item == '<not serializable>':
                        attributes[key] = None

                # Update the object
                self.__dict__.update(attributes)

                # Set the directories
                self.set_fargo3d_dir(self.fargo3d_dir)
                self.set_setup(self.setup)
                self.set_output_dir(self.output_dir)
            return
        
        # Set units by default
        self.set_units(UL=AU,UM=MSUN)
        
        # Set properties
        self.set_property('fargo3d_dir',
                          fargopy.Conf.FP_FARGO3D_DIR,
                          self.set_fargo3d_dir)
        self.set_property('setup',
                          None,
                          self.set_setup)
        self.set_property('output_dir',
                          None,
                          self.set_output_dir)
        self.set_property('fargo3d_compilation_options',
                          dict(parallel=0,gpu=0,options=''))

        # Check if binary is already compiled
        fargo3d_binary,compile_options = self._generate_binary_name(parallel=self.fargo3d_compilation_options['parallel'],
                                                    gpu=self.fargo3d_compilation_options['gpu'],
                                                    options=self.fargo3d_compilation_options['options'])
        if os.path.isfile(f"{self.fargo3d_dir}/{fargo3d_binary}"):
            print(f"FARGO3D binary '{fargo3d_binary}' found.")
            self.fargo3d_binary = fargo3d_binary
        else:
            self.fargo3d_binary = None

        # Simulation process does not exist   
        self.set_property('fargo3d_process',
                          None)
        
    # ##########################################################################
    # Set special properties
    # #########################################a#################################  
    def set_fargo3d_dir(self,dir=None):
        """
        Set the FARGO3D directory for the simulation.

        Parameters
        ----------
        dir : str, optional
            Directory where FARGO3D is installed.

        Returns
        -------
        bool or None
            True if the directory exists and the header file is found, False otherwise.
        """
        if dir is None:
            return
        if not os.path.isdir(dir):
            print(f"FARGO3D directory '{dir}' does not exist.")
            return
        else:
            fargo_header = os.path.join(dir, fargopy.Conf.FP_FARGO3D_HEADER)
            if not os.path.isfile(fargo_header):
                print(f"No header file for FARGO3D found in '{fargo_header}'")
            else:
                print(f"Your simulation is now connected with '{dir}'")
        
        # Set derivative dirs
        self.fargo3d_dir = dir
        self.outputs_dir = os.path.join(self.fargo3d_dir, 'outputs')
        self.setups_dir = os.path.join(self.fargo3d_dir, 'setups')
    
    def set_setup(self,setup):
        """
        Connect the simulation to a given setup.

        Parameters
        ----------
        setup : str
            Name of the setup.

        Returns
        -------
        str or None
            The setup name if successful, None otherwise.
        """
        if setup is None:
            self.setup_dir = None
            return None
        setup_dir = os.path.join(self.setups_dir, setup)
        if self.set_setup_dir(setup_dir):
            self.setup = setup
            self.logfile = os.path.join(self.setup_dir, f"{self.setup}.log")
        return setup
    
    def set_setup_dir(self,dir):
        """
        Set the setup directory.

        Parameters
        ----------
        dir : str
            Directory where setup is available.

        Returns
        -------
        bool
            True if the setup directory exists, False otherwise.
        """
        if dir is None:
            return False
        if not os.path.isdir(dir):
            print(f"Setup directory '{dir}' does not exist.")
            return False
        else:
            print(f"Now your simulation setup is at '{dir}'")

        self.setup_dir = dir
        return True

    def set_output_dir(self,dir):
        """
        Connect a simulation with a directory where the outputs are stored.

        Parameters
        ----------
        dir : str
            Output directory path.
        """
        if dir is None:
            return
        if not os.path.isdir(dir):
            print(f"Output directory '{dir}' does not exist.")
            return
        else:        
            print(f"Now you are connected with output directory '{dir}'")
            self.output_dir = dir
            # Check if there are results 
            par_file = os.path.join(dir, 'variables.par')
            if os.path.isfile(par_file):
                print(f"Found a variables.par file in '{dir}', loading properties")
                self.load_properties()
        
        return
    
    ################################
    ##  UNITS
    ################################

    def units(self, system):
        """
        Set the units system for the simulation.

        Parameters
        ----------
        system : str
            The units system to use ('CGS' or 'MKS').
        """
        if system.upper() == 'CGS':
            self._set_constants_cgs()
            self.units_system = 'CGS'
            print("Units set to CGS.")
        elif system.upper() == 'MKS':
            self._set_constants_mks()
            self.units_system = 'MKS'
            print("Units set to MKS.")
        else:
            raise ValueError("Invalid units system. Use 'CGS' or 'MKS'.")

    def _set_constants_cgs(self):
        """
        Set physical constants in CGS units.
        """
        self.KB = 1.380650424e-16  # Boltzmann constant: erg/K
        self.MP = 1.672623099e-24  # Mass of the proton, g
        self.GCONST = 6.67259e-8  # Gravitational constant, cm^3/g/s^2
        self.RGAS = 8.314472e7  # Gas constant, erg/K/mol
        self.MSUN = 1.9891e33  # Solar mass, g
        self.AU = 1.49598e13  # Astronomical unit, cm
        self.YEAR = 31557600.0  # Year, s

    def _set_constants_mks(self):
        """
        Set physical constants in MKS units.
        """
        self.KB = 1.380649e-23  # Boltzmann constant: J/K
        self.MP = 1.6726219e-27  # Mass of the proton, kg
        self.GCONST = 6.67430e-11  # Gravitational constant, m^3/kg/s^2
        self.RGAS = 8.314462618  # Gas constant, J/K/mol
        self.MSUN = 1.9891e30  # Solar mass, kg
        self.AU = 1.49598e11  # Astronomical unit, m
        self.YEAR = 31557600.0  # Year, s

    def set_units(self,UM=MSUN,UL=AU,G=1,mu=2.35):
        """
        Set the units of the simulation.

        Parameters
        ----------
        UM : float
            Mass unit (default: MSUN).
        UL : float
            Length unit (default: AU).
        G : float
            Gravitational constant (default: 1).
        mu : float
            Mean molecular weight (default: 2.35).
        """
        # Basic
        self.UM = UM
        self.UL = UL
        self.G = G
        self.UT = (G*self.UL**3/(GCONST*self.UM))**0.5 # In seconds

        # Thermodynamics
        self.UTEMP = (GCONST*MP*mu/KB)*self.UM/self.UL # In K
        
        # Derivative
        self.USIGMA = self.UM/self.UL**2 # In g/cm^2
        self.URHO = self.UM/self.UL**3 # In kg/m^3
        self.UEPS = self.UM/(self.UL*self.UT**2)  # In J/m^3
        self.UV = self.UL/self.UT


    # ##########################################################################
    # Control methods
    # ##########################################################################  
    def compile(self,setup=None,parallel=0,gpu=0,options='',force=False):
        """
        Compile the FARGO3D binary for the current setup.

        Parameters
        ----------
        setup : str, optional
            Name of the setup to compile.
        parallel : int, optional
            Use parallel compilation (default: 0).
        gpu : int, optional
            Use GPU compilation (default: 0).
        options : str, optional
            Additional compilation options.
        force : bool, optional
            If True, clean the directory before compiling.
        """
        if setup is not None:
            if not self.set_setup(setup):
                print("Failed")
                return 

        # Clean directrory
        if force:
            print(f"Cleaning FARGO3D directory {self.fargo3d_dir}...")
            cmd = f"make -C {self.fargo3d_dir} clean mrproper"
            # Clean all binaries
            compl = f"rm -rf {self.fargo3d_dir}/fargo3d_*"
            error,output_clean = fargopy.Sys.run(cmd + '&&' + compl)
            
        # Prepare compilation
        fargo3d_binary,compile_options = self._generate_binary_name(parallel=parallel,gpu=gpu,options=options)
        #compile_options = f"SETUP={self.setup} PARALLEL={parallel} GPU={gpu} "+options
        #fargo3d_binary = f"fargo3d-{compile_options.replace(' ','-').replace('=','_').strip('-')}"

        # Compile binary
        print(f"Compiling {fargo3d_binary}...")
        cmd = f"cd {self.fargo3d_dir};make {compile_options} 2>&1 |tee {self.setup_dir}/compilation.log"
        compl = f"mv fargo3d {fargo3d_binary}"
        pipe = f""
        error,output_compilation = fargopy.Sys.run(cmd+' && '+compl)

        # Check compilation result
        if os.path.isfile(f"{self.fargo3d_dir}/{fargo3d_binary}"):
            self.fargo3d_binary = fargo3d_binary
            print(f"Succesful compilation of FARGO3D binary {self.fargo3d_binary}")
            self.fargo3d_compilation_options=dict(
                parallel=parallel,
                gpu=gpu,
                options=options
            )
        else:
            print(f"Something failed when compiling FARGO3D. For details check '{self.setup_dir}/compilation.log")
            
    def _generate_binary_name(self,parallel=0,gpu=0,options=''):
        """
        Generate the binary name for the compiled FARGO3D executable.

        Parameters
        ----------
        parallel : int
            Parallel option.
        gpu : int
            GPU option.
        options : str
            Additional options.

        Returns
        -------
        tuple
            (binary_name, compile_options)
        """
        compile_options = f"SETUP={self.setup} PARALLEL={parallel} GPU={gpu} "+options
        fargo3d_binary = f"fargo3d-{compile_options.replace(' ','-').replace('=','_').strip('-')}"
        return fargo3d_binary, compile_options

    def run(self,
            mode='async',
            options='-m',
            mpioptions='-np 1',
            resume=False,
            cleanrun=False,
            test=False,
            unlock=True):
        """
        Run the FARGO3D simulation.

        Parameters
        ----------
        mode : str, optional
            'async' for asynchronous or 'sync' for synchronous execution.
        options : str, optional
            Command-line options for FARGO3D.
        mpioptions : str, optional
            MPI options for parallel runs.
        resume : bool, optional
            Resume from previous run.
        cleanrun : bool, optional
            Clean output directory before running.
        test : bool, optional
            If True, do not actually run the simulation.
        unlock : bool, optional
            If True, unlock the simulation if locked.
        """

        if self.fargo3d_binary is None:
            print("You must first compile your simulation with: <simulation>.compile(<option>).")
            return

        if self._is_running():
            print(f"There is a running process. Please stop it before running/resuming")
            return

        lock_info = fargopy.Sys.is_locked(self.setup_dir)        
        if lock_info:
            print(f"The process is locked by PID {lock_info['pid']}")
            return

        # Mandatory options
        options = options + " -t"
        if 'fargo3d_run_options' not in self.__dict__.keys():
            self.fargo3d_run_options = options
        
        # Clean output if available
        if cleanrun:
            # Check if there is an output director
            output_dir = os.path.join(self.outputs_dir, self.setup)
            if os.path.isdir(output_dir):
                self.output_dir = output_dir
                self.clean_output()
            else:
                print(f"No output directory {output_dir} yet created.")
        
        # Select command to run
        precmd=''
        if self.fargo3d_compilation_options['parallel']:
            precmd = f"mpirun {mpioptions} "

        # Preparing command
        run_cmd = f"{precmd} ./{self.fargo3d_binary} {options} setups/{self.setup}/{self.setup}.par"

        self.json_file = os.path.join(self.setup_dir, 'fargopy_simulation.json')
        if mode == 'sync':
            # Save object 
            self.save_object(self.json_file)

            # Run synchronously
            cmd = f"cd {self.fargo3d_dir};{run_cmd} |tee {self.logfile}"
            print(f"Running synchronously: {cmd}")
            fargopy.Sys.simple(cmd)
            self.fargo3d_process = None

        elif mode == 'async':
            # Run asynchronously
            
            # Select logfile mode accroding to if the process is resuming
            logmode = 'a' if resume else 'w'
            logfile_handler=open(self.logfile,logmode)

            # Launch process
            print(f"Running asynchronously (test = {test}): {run_cmd}")
            if not test:

                # Check it is not locked
                lock_info = fargopy.Sys.is_locked(dir=self.setup_dir)
                if lock_info:
                    if unlock:
                        self.unlock_simulation(lock_info)
                    else:
                        print(f"Output directory {self.setup_dir} is locked by a running process")
                        return

                process = subprocess.Popen(run_cmd.split(),cwd=self.fargo3d_dir,
                                           stdout=logfile_handler,stderr=logfile_handler,close_fds=True)
                
                # Introduce a short delay to verify if the process has failed
                time.sleep(1.0)

                if process.poll() is None:
                    # Check if program is effectively running
                    self.fargo3d_process = process            
                    
                    # Lock
                    fargopy.Sys.lock(
                        dir=self.setup_dir,
                        content=dict(pid=self.fargo3d_process.pid)
                    )

                    # Setup output directory 
                    self.set_output_dir(os.path.join(self.outputs_dir, self.setup))    

                    # Save object 
                    self.save_object(self.json_file)
                else:
                    print(f"Process running failed. Please check the logfile {self.logfile}")
        else:
            print(f"Mode {mode} not recognized (valid are 'sync', 'async')")
            return
                    
    def stop(self):
        """
        Stop the running FARGO3D process and unlock the simulation directory.
        """
        # Check if the directory is locked
        lock_info = fargopy.Sys.is_locked(self.setup_dir)
        
        if lock_info:
            print(f"The process is locked by PID {lock_info['pid']}")

        # If not process associated unlock the simulation
        if not self._check_process():
            self.unlock_simulation(lock_info)
            return

        poll = self.fargo3d_process.poll()
        if poll is None:
            print(f"Stopping FARGO3D process (pid = {self.fargo3d_process.pid})")
            subprocess.Popen.kill(self.fargo3d_process)
            self.unlock_simulation(lock_info)
            del self.fargo3d_process
            self.fargo3d_process = None
        else:
            self.unlock_simulation(lock_info)
            print(f"The process has finished. Check logfile {self.logfile}.")

    def unlock_simulation(self,lock_info=None,force=True):
        """
        Unlock a simulation directory, killing the process if necessary.

        Parameters
        ----------
        lock_info : dict, optional
            Lock information.
        force : bool, optional
            If True, force unlock.
        """
        if lock_info is None and force:
            lock_info = fargopy.Sys.is_locked(self.setup_dir)
            if lock_info:
                print(f"Unlocking simulation (pid = {lock_info['pid']})")
                
        if lock_info:
            pid = lock_info['pid']
            fargopy.Debug.trace(f"Unlocking simulation (pid = {pid})")
            error,output = fargopy.Sys.run(f"kill -9 {pid}")    
            fargopy.Sys.unlock(self.setup_dir)
        
    def status(self,mode='isrunning',verbose=True,**kwargs):
        """
        Check the status of the running process.

        Parameters
        ----------
        mode : str, optional
            Status mode ('isrunning', 'logfile', 'outputs', 'progress', 'summary', 'locking', or 'all').
        verbose : bool, optional
            If True, print detailed output.
        **kwargs : dict
            Additional keyword arguments.
        """
        # Bar separating output 
        bar = f"\n{''.join(['#']*80)}\n"
        
        # vprint
        vprint = print if verbose else lambda x:x

        if 'isrunning' in mode or mode=='all':
            vprint(bar+"Running status of the process:")
            if self.fargo3d_process:
                poll = self.fargo3d_process.poll()
                if poll is None:
                    vprint("\tThe process is running.")
                else:
                    # Unlock any remaining process
                    lock_info = fargopy.Sys.is_locked(self.setup_dir)
                    self.unlock_simulation(lock_info)
                    
                    vprint(f"\tThe process has ended with termination code {poll}.")
            else:
                vprint(f"\tThe process is stopped.")

        if 'logfile' in mode or mode=='all':
            vprint(bar+"Logfile content:")
            if 'logfile' in self.__dict__.keys() and os.path.isfile(self.logfile):
                vprint("The latest 10 lines of the logfile:\n")
                if verbose:
                    os.system(f"tail -n 10 {self.logfile}")
            else:
                vprint("No log file created yet")

        if 'outputs' in mode or mode=='all':
            vprint(bar+"Output content:")
            error,output = fargopy.Sys.run(f"ls {self.output_dir}/*.dat")
            if not error:
                files = [file.split('/')[-1] for file in output[:-1]]
                file_list = ""
                for i,file in enumerate(files):
                    file_list += f"{file}, "
                    if ((i+1)%10) == 0:
                        file_list += "\n"
                file_list = file_list.strip("\n,")
                vprint(f"\n{len(files)} available datafiles:\n")
                self.output_datafiles = files
                vprint(file_list)
            else:
                vprint("No datafiles yet available")

        if 'summary' in mode or mode=='all':
            vprint(bar+"Summary:")
            nsnaps = self._get_nsnaps()
            print(f"The simulation has been ran for {nsnaps} time-steps (including the initial one).")

        if 'locking' in mode or mode=='all':
            vprint(bar+"Locking status:")
            lock_info = fargopy.Sys.is_locked(self.setup_dir)
            print(lock_info)

        if 'progress' in mode:
            vprint(bar)
            numstatus = 5
            if 'numstatus' in kwargs.keys():
                numstatus = int(kwargs['numstatus'])
            self._status_progress(numstatus=numstatus)

        print(f"\nOther status modes: 'isrunning', 'logfile', 'outputs', 'progress', 'summary'")

    def _status_progress(self,minfreq=0.1,numstatus=100):
        """
        Show a progress of the execution.

        Parameters
        ----------
        minfreq : float, optional
            Minimum seconds between status checks.
        numstatus : int, optional
            Number of status updates before stopping.
        """
        # Prepare
        if 'status_frequency' not in self.__dict__.keys():
            frequency = minfreq
        else:
            frequency = self.status_frequency
        previous_output = ''
        previous_resumable_snapshot = 1e100
        time_previous = time.time()

        # Infinite loop checking for output
        n = 0
        print(f"Progress of the simulation (interrupt by pressing 'enter' or the stop button):")
        while True and (n<numstatus):
            
            # Check if the process is running locally
            if not self._is_running():
                print("The simulation is not running anymore")
                return
            
            error,output = fargopy.Sys.run(f"grep OUTPUT {self.logfile} |tail -n 1")
            
            if not error:
                # Get the latest output
                latest_output = output[-2]
                if latest_output != previous_output:
                    print(f"{n+1}:{latest_output} [output pace = {frequency:.1f} secs] <Press 'enter' to interrupt>")
                    n += 1 
                    # Fun the number of the output
                    find = re.findall(r'OUTPUTS\s+(\d+)',latest_output)
                    resumable_snapshot = int(find[0])
                    # Get the time elapsed since last status check
                    time_now = time.time()
                    frequency = max(time_now - time_previous,minfreq)/2
                    if (resumable_snapshot - previous_resumable_snapshot)>1:
                        # Reduce frequency if snapshots are accelerating
                        frequency = frequency/2
                        self.status_frequency = frequency
                    previous_resumable_snapshot = resumable_snapshot
                    time_previous = time_now
                previous_output = latest_output
            
            if fargopy.IN_COLAB:
                # In colab the control is much more limited
                try:
                    time.sleep(frequency)
                except KeyboardInterrupt:
                    print("Interrupted by user. In some environment (IPython, Colab) stopping the progress status will stop the simulation. In that case just resume.")
                    return
                
            else:
                # Suitable for running in Jupyter and IPython
                if fargopy.Sys.sleep_timeout(frequency):
                    return
                
    def resume(self,snapshot=-1,mpioptions='-np 1'):
        """
        Resume the simulation from a given snapshot.

        Parameters
        ----------
        snapshot : int, optional
            Snapshot number to resume from. If -1, resumes from the latest.
        mpioptions : str, optional
            MPI options for parallel runs.
        """
        latest_snapshot_resumable = self._is_resumable()
        if latest_snapshot_resumable<0:
            return
        if self._is_running():
            print(f"There is a running process. Please stop it before resuming")
            return
        if 'fargo3d_run_options' not in self.__dict__.keys():
            print(f"The process has not been run before.")
            return
        # Resume
        if snapshot<0:
            snapshot = latest_snapshot_resumable
        print(f"Resuming from snapshot {snapshot}...")
        self.run(mode='async',mpioptions=mpioptions,resume=True,
                 options=self.fargo3d_run_options+f' -S {snapshot}',test=False)

    def _has_finished(self):
        """
        Check if the simulation process has finished.

        Returns
        -------
        bool
            True if finished, False otherwise.
        """
        if self.fargo3d_process:
            poll = self.fargo3d_process.poll()
            if poll is None:
                return False
            else:
                print(f"The process has ended with termination code {poll}.")
                return True

    def _is_resumable(self):
        """
        Check if the simulation can be resumed.

        Returns
        -------
        int
            Latest resumable snapshot number, or -1 if not resumable.
        """
        if self.logfile is None:
            print(f"The simulation has not been ran yet. Run <simulation>.run() before resuming")
            return -1
        latest_snapshot_resumable = max(self._get_nsnaps() - 2, 0)
        return latest_snapshot_resumable
        
    def clean_output(self):
        """
        Clean the output directory by removing all files.
        """
        if self.output_dir is None:
            print(f"Output directory has not been set.")
            return

        if self._is_running():
            print(f"There is a running process. Please stop it before cleaning")
            return
        
        print(f"Cleaning output directory {self.output_dir}")
        cmd = f"rm -rf {self.output_dir}/*"
        error,output = fargopy.Sys.run(cmd)

    def _is_running(self,verbose=False):
        """
        Check if the simulation process is currently running.

        Parameters
        ----------
        verbose : bool, optional
            If True, print status messages.

        Returns
        -------
        bool
            True if running, False otherwise.
        """
        lock_info = fargopy.Sys.is_locked(self.setup_dir)
        if lock_info:
            # Check if process is up
            error,output = fargopy.Sys.run(f"ps -p {lock_info['pid']}")
            if error == 0:
                return True

        if not self.has('fargo3d_process'):
            if verbose:
                print("The simulation has not been run before.")
            return False

        if self.fargo3d_process:
            poll = self.fargo3d_process.poll()
            if poll is None:
                if verbose:
                    print(f"The process is already running with pid '{self.fargo3d_process.pid}'")
                return True
            else:
                return False
        else:
            return False

    def _check_process(self):
        """
        Check if the FARGO3D process handler is available.

        Returns
        -------
        bool
            True if process handler exists, False otherwise.
        """
        if self.fargo3d_process is None:
            print(f"There is no FARGO3D process handler available.")
            return False
        return True

    # ##########################################################################
    # Operations on the FARGO3D directories
    # ##########################################################################  
    def list_fields(self,quiet=False):
        """
        List all field files in the output directory.

        Parameters
        ----------
        quiet : bool, optional
            If True, suppress detailed output.

        Returns
        -------
        list
            List of field filenames.
        """
        if self.output_dir is None:
            print(f"You have to set forst the outputs directory with <sim>.set_outputs('<directory>')")
        else:
            error,output = fargopy.Sys.run(f"ls {self.output_dir}")
        if error == 0:
            files = output[:-1]
            print(f"{len(files)} files in output directory")
            if not quiet:
                file_list = ""
                for i,file in enumerate(files):
                    file_list += f"{file}, "
                    if ((i+1)%10) == 0:
                        file_list += "\n"
                print(file_list)
            return files
            

    def load_macros(self, summaryfile='summary0.dat'):
        """
        Extract global simulation parameters from the PREPROCESSOR MACROS SECTION in summary0.dat.

        Parameters
        ----------
        summaryfile : str, optional
            Name of the summary file.

        Returns
        -------
        dict
            Dictionary with macro parameters.
        """
        summary_path = os.path.join(self.output_dir, summaryfile)
        if not os.path.isfile(summary_path):
            print(f"No summary file '{summary_path}' found.")
            return {}

        macros = {}
        in_macros_section = False
        with open(summary_path, 'r') as f:
            for line in f:
                # Detect the start of the macros section
                if 'PREPROCESSOR MACROS SECTION' in line:
                    in_macros_section = True
                    continue
                # Stop if another section starts
                if in_macros_section and 'SECTION' in line and 'PREPROCESSOR MACROS SECTION' not in line:
                    break
                if in_macros_section:
                    # Skip empty lines and lines that don't contain '='
                    if line.strip() == '' or '=' not in line or line.strip().startswith('=') or line.strip().startswith('#'):
                        continue
                    # Extract parameter name and value after the last '='
                    parts = line.split('=')
                    if len(parts) >= 2:
                        key = parts[0].strip()
                        value_str = parts[-1].strip()
                        # Try to convert to float if possible
                        try:
                            value = float(value_str)
                        except ValueError:
                            value = value_str
                        macros[key] = value
        self.simulation_macros = macros
        return macros

    def load_planets(self, snapshot=0, summaryfile=None):
        """
        Load planet data from a summary file.

        Parameters
        ----------
        snapshot : int, optional
            Snapshot index to load.
        summaryfile : str, optional
            Name of the summary file.

        Returns
        -------
        list of Planet
            List of Planet objects for the given snapshot.
        """
        import os

        # Determine summary file
        if summaryfile is None:
            summaryfile = f"summary{snapshot}.dat"
        summary_path = os.path.join(self.output_dir, summaryfile)
        if not os.path.isfile(summary_path):
            print(f"No summary file '{summary_path}' found.")
            return []

        # Get stellar mass
        if not hasattr(self, "simulation_macros") or 'MSTAR' not in self.simulation_macros:
            self.load_macros()
        mstar = self.simulation_macros['MSTAR']

        # Parse initial positions from the top of the file (if available)
        initial_planets = []
        with open(summary_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if line.strip().startswith('#') or not line.strip() or set(line.strip()) == set('-'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    float(parts[0])
                    continue  # skip if first field is a number
                except ValueError:
                    pass
                try:
                    name = parts[0]
                    x0 = float(parts[1])
                    y0 = float(parts[2]) if len(parts) > 3 else 0.0
                    z0 = 0.0
                    initial_planets.append({'name': name, 'posi': (x0, y0, z0)})
                except Exception:
                    continue

        # Find PLANETARY SYSTEM SECTION
        planet_indices = []
        in_section = False
        for i, line in enumerate(lines):
            if 'PLANETARY SYSTEM SECTION' in line:
                in_section = True
                continue
            if in_section and line.strip().startswith('#### Planet'):
                planet_indices.append(i+1)  # next line has the data
            if in_section and line.strip().startswith('***'):
                break

        planets = []
        for idx, data_idx in enumerate(planet_indices):
            data_line = lines[data_idx].strip()
            parts = data_line.split()
            if len(parts) < 7:
                continue
            x, y, z = map(float, parts[0:3])
            vx, vy, vz = map(float, parts[3:6])
            mass = float(parts[-1])  # Always take the last column as mass

            # Get initial position if available
            if idx < len(initial_planets):
                name = initial_planets[idx]['name']
                posi = initial_planets[idx]['posi']
            else:
                name = f"Planet {idx}"
                posi = (x, y, z)
            planet = Planet(name=name, pos=(x, y, z), vel=(vx, vy, vz), mass=mass, posi=posi, mstar=mstar)
            planets.append(planet)
        return planets
                
 

    def load_properties(self, quiet=False,
                                varfile='variables.par',
                                domain_prefix='domain_',
                                dimsfile='dims.dat',
                                summaryfile='summary0.dat'):
        """
        Load and print simulation properties, including variables, domains, dimensions, and planets.

        Parameters
        ----------
        quiet : bool, optional
            If True, suppress output.
        varfile : str, optional
            Name of the variables file.
        domain_prefix : str, optional
            Prefix for domain files.
        dimsfile : str, optional
            Name of the dimensions file.
        summaryfile : str, optional
            Name of the summary file.

        Returns
        -------
        str
            Summary information string.
        """
        info = []
        if self.output_dir is None:
            msg = f"You have to set first the outputs directory with <sim>.set_outputs('<directory>')"
            print(msg)
            return msg

        # Read variables
        vars = self._load_variables(varfile)
        if not vars:
            return "No variables found."
        info.append(f"Simulation in {vars.DIM} dimensions")
        print(f"Simulation in {vars.DIM} dimensions")
        
        # Read domains 
        domains = self._load_domains(vars, domain_prefix)

        # Store the variables in the object
        self.vars = vars
        self.domains = domains

        # Optionally read dims
        dims = self._load_dims(dimsfile)
        if len(dims):
            self.dims = dims 

        # Read the summary files
        self.nsnaps = self._get_nsnaps()
        info.append(f"Number of snapshots in output directory: {self.nsnaps}")
        print(f"Number of snapshots in output directory: {self.nsnaps}")

        # Read planets from summary.dat using load_planet_states
        self.planets = self.load_planets(snapshot=0, summaryfile=summaryfile)
        if self.planets:
            info.append("Planets found in summary.dat:")
            print("Planets found in summary.dat:")
            for planet in self.planets:
                planet_info = f"  Name: {planet.name}, Initial pos: {planet.posi}, Mass: {planet.mass}"
                info.append(planet_info)
                print(planet_info)
        else:
            info.append("No planet data found in summary.dat.")
            print("No planet data found in summary.dat.")

        # Devuelve el string para mostrar en el cuadro de diálogo
        return "\n".join(info)

    def _get_nsnaps(self):
        """
        Get the number of snapshots in the output directory.

        Returns
        -------
        int
            Number of snapshot files.
        """
        try:
            # List all files in the output directory
            files = [f for f in os.listdir(self.output_dir) if f.startswith("summary") and f.endswith(".dat")]
            nsnaps = len(files)
            return nsnaps
        except FileNotFoundError:
            print(f"No summary file in {self.output_dir}")
            return 0

    def _load_dims(self, dimsfile):
        """
        Parse the dimensions file.

        Parameters
        ----------
        dimsfile : str
            Name of the dimensions file.

        Returns
        -------
        np.ndarray or list
            Array of dimensions or empty list if not found.
        """
        dimsfile = os.path.join(self.output_dir, dimsfile)
        if not os.path.isfile(dimsfile):
            #print(f"No file with dimensions '{dimsfile}' found.")
            return []
        dims = np.loadtxt(dimsfile)
        return dims

    def _load_variables(self,varfile):
        """
        Parse the file with the simulation variables.

        Parameters
        ----------
        varfile : str
            Name of the variables file.

        Returns
        -------
        fargopy.Dictobj
            Object containing simulation variables.
        """
        varfile = os.path.join(self.output_dir, varfile)
        if not os.path.isfile(varfile):
            print(f"No file with variables named '{varfile}' found.")
            return

        print(f"Loading variables")
        variables = np.genfromtxt(
            varfile,dtype={'names': ("parameters","values"),
            'formats': ("|S30","|S300")}
        ).tolist()
        
        vars = dict()
        for posicion in variables:
            str_value = posicion[1].decode("utf-8")
            try:
                value = int(str_value)
            except:
                try:
                    value = float(str_value)
                except:
                    value = str_value
            vars[posicion[0].decode("utf-8")] = value
        
        vars = fargopy.Dictobj(dict=vars)
        print(f"{len(vars.__dict__.keys())} variables loaded")

        # Create additional variables
        variables = ['x', 'y', 'z']
        if vars.COORDINATES == 'cylindrical':
            variables = ['phi', 'r', 'z']
        elif vars.COORDINATES == 'spherical':
            variables = ['phi', 'r', 'theta']
        vars.VARIABLES = variables

        vars.__dict__[f'N{variables[0].upper()}'] = vars.NX
        vars.__dict__[f'N{variables[1].upper()}'] = vars.NY
        vars.__dict__[f'N{variables[2].upper()}'] = vars.NZ

        # Dimension of the domain
        vars.DIM = 2 if vars.NZ == 1 else 3
        
        return vars

    def _load_domains(self,vars,domain_prefix,
                      borders=[[],[3,-3],[3,-3]],
                      middle=True):
        """
        Load domain coordinate arrays from files.

        Parameters
        ----------
        vars : fargopy.Dictobj
            Simulation variables.
        domain_prefix : str
            Prefix for domain files.
        borders : list, optional
            List of border slices for each variable.
        middle : bool, optional
            If True, average between cell coordinates.

        Returns
        -------
        fargopy.Dictobj
            Object containing domain arrays.
        """
        # Coordinates
        variable_suffixes = ['x', 'y', 'z']
        print(f"Loading domain in {vars.COORDINATES} coordinates:")

        # Correct dims in case of 2D
        if vars.DIM == 2:
            borders[-1] = []

        # Load domains
        domains = dict()
        domains['extrema'] = dict()

        for i,variable_suffix in enumerate(variable_suffixes):
            domain_file = os.path.join(self.output_dir, f"{domain_prefix}{variable_suffix}.dat")
            if os.path.isfile(domain_file):

                # Load data from file
                domains[vars.VARIABLES[i]] = np.genfromtxt(domain_file)

                if len(borders[i]) > 0:
                    # Drop the border of the domain
                    domains[vars.VARIABLES[i]] = domains[vars.VARIABLES[i]][borders[i][0]:borders[i][1]]

                if middle:
                    # Average between domain cell coordinates
                    domains[vars.VARIABLES[i]] = 0.5*(domains[vars.VARIABLES[i]][:-1]+domains[vars.VARIABLES[i]][1:])
                
                # Show indices and value map
                domains['extrema'][vars.VARIABLES[i]] = [[0,domains[vars.VARIABLES[i]][0]],[-1,domains[vars.VARIABLES[i]][-1]]]
                
                print(f"\tVariable {vars.VARIABLES[i]}: {len(domains[vars.VARIABLES[i]])} {domains['extrema'][vars.VARIABLES[i]]}")
            else:
                print(f"\tDomain file {domain_file} not found.")
        domains = fargopy.Dictobj(dict=domains)

        return domains        


    def load_field(self, fields, slice=None, snapshot=None, type=None, interpolate=False, cut=None):
        """
        Load a field or multiple fields from the simulation.
        """

        # Ensure fields is a list
        if isinstance(fields, str):
            fields = [fields]

        # ==========================================================
        # ===  CASE 1: interpolate=True  → unify multiple fields ===
        # ==========================================================
        if interpolate is True:

            # Crear un *solo* FieldInterpolator
            handler = fargopy.FieldInterpolator(self)

            # Llamar load_data UNA vez pasando todos los fields
            handler.load_data(
                fields=fields,
                slice=slice,
                snapshots=snapshot,
                cut=cut
            )

            # Devolver solo este objeto (no lista)
            return handler

        # ==========================================================
        # ===  CASE 2: interpolate=False → comportamiento antiguo ===
        # ==========================================================
        else:
            if not self.has('vars'):
                dims, vars, domains = self.load_properties()

            snapshot = 0 if snapshot is None else snapshot
            loaded_fields = []

            for field in fields:
                # Infer field type unless provided
                field_type = type
                if field_type is None:
                    if field in ['gasdens', 'gasenergy']:
                        field_type = 'scalar'
                    elif field == 'gasv':
                        field_type = 'vector'
                    else:
                        raise ValueError(f"Field type for '{field}' could not be inferred.")

                # Load scalar
                if field_type == 'scalar':
                    file_name = f"{field}{snapshot}.dat"
                    file_field = os.path.join(self.output_dir, file_name)
                    data = self._load_field_scalar(file_field)

                # Load vector
                elif field_type == 'vector':
                    data = []
                    components = ['x','y'] + (['z'] if self.vars.DIM == 3 else [])
                    for comp in components:
                        file_name = f"{field}{comp}{snapshot}.dat"
                        file_field = os.path.join(self.output_dir, file_name)
                        data.append(self._load_field_scalar(file_field))
                    data = np.array(data)

                # Create Field
                loaded_field = fargopy.Field(
                    data=np.array(data),
                    coordinates=self.vars.COORDINATES,
                    domains=self.domains,
                    type=field_type
                )

                # Apply slicing
                if slice:
                    sliced_data, mesh = loaded_field.meshslice(slice=slice)
                    loaded_field = fargopy.Dictobj(dict=dict(data=sliced_data, mesh=mesh))

                loaded_fields.append(loaded_field)

            return loaded_fields if len(loaded_fields) > 1 else loaded_fields[0]


    def _load_field_scalar(self,file):
        """
        Load a scalar field from a file.

        Parameters
        ----------
        file : str
            Path to the field file.

        Returns
        -------
        np.ndarray
            Field data array.
        """
        if os.path.isfile(file):
            field_data = np.fromfile(file).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            return field_data
        else:
            raise AssertionError(f"File with field '{file}' not found")
        
    def load_allfields(self,fluid,snapshot=None,type='scalar'):
        """
        Load all fields in the output directory for a given fluid.

        Parameters
        ----------
        fluid : str
            Name of the fluid (e.g., 'gas').
        snapshot : int, optional
            Snapshot index to load. If None, loads all snapshots.
        type : str, optional
            Field type ('scalar' or 'vector').

        Returns
        -------
        fargopy.Dictobj
            Object containing all loaded fields.
        """
        qall = False
        if snapshot is None:
            qall = True
            fields = fargopy.Dictobj()
        else:
            fields = fargopy.Dictobj()
        
        # Search for field files
        pattern = os.path.join(self.output_dir, f"{fluid}*.dat")
        error,output = fargopy.Sys.run(f"ls {pattern}")

        if not error:
            size = 0
            for file_field in output[:-1]:
                comps = Simulation._parse_file_field(file_field)
                if comps:
                    if qall:
                        # Store all snapshots
                        field_name = comps[0]
                        field_snap = int(comps[1])

                        if type == 'scalar':
                            field_data = self._load_field_scalar(file_field)
                        elif type == 'vector':
                            field_data = []
                            variables = ['x','y'] 
                            if self.vars.DIM == 3:
                                variables += ['z'] 
                            for i,variable in enumerate(variables):
                                file_name = f"{fluid}{variable}{str(field_snap)}.dat"
                                file_field = os.path.join(self.output_dir, file_name)
                                field_data += [self._load_field_scalar(file_field)]
                            field_data = np.array(field_data)
                            field_name = field_name[:-1]

                        if str(field_snap) not in fields.keys():
                            fields.__dict__[str(field_snap)] = fargopy.Dictobj()
                        size += field_data.nbytes
                        (fields.__dict__[str(field_snap)]).__dict__[f"{field_name}"] = fargopy.Field(data=field_data, coordinates=self.vars.COORDINATES, domains=self.domains, type=type)

                    else:
                        # Store a specific snapshot
                        if int(comps[1]) == snapshot:
                            field_name = comps[0]

                            if type == 'scalar':
                                field_data = self._load_field_scalar(file_field)
                            elif type == 'vector':
                                field_data = []
                                variables = ['x','y'] 
                                if self.vars.DIM == 3:
                                    variables += ['z'] 
                                for i,variable in enumerate(variables):
                                    file_name = f"{fluid}{variable}{str(field_snap)}.dat"
                                    file_field = os.path.join(self.output_dir, file_name)
                                    field_data += [self._load_field_scalar(file_field)]
                                field_data = np.array(field_data)
                                field_name = field_name[:-1]

                            size += field_data.nbytes
                            fields.__dict__[f"{field_name}"] = fargopy.Field(data=field_data, coordinates=self.vars.COORDINATES, domains=self.domains, type=type)

        else:
            raise ValueError(f"No field found with pattern '{pattern}'. Change the fluid")
    
        if qall:
            fields.snapshots = sorted([int(s) for s in fields.keys() if s != 'size'])
        fields.size = size/1024**2
        return fields

    @staticmethod
    def _parse_file_field(file_field):
        """
        Parse a field filename to extract the field name and snapshot number.

        Parameters
        ----------
        file_field : str
            Filename of the field.

        Returns
        -------
        list or None
            List with field name and snapshot number, or None if not matched.
        """
        basename = os.path.basename(file_field)
        comps = None
        match = re.match('([a-zA-Z]+)(\d+).dat',basename)
        if match is not None:
            comps = [match.group(i) for i in range(1,match.lastindex+1)]
        return comps

    def __repr__(self):
        """
        String representation of the Simulation object.

        Returns
        -------
        str
        """
        repr = f"""FARGOPY simulation (fargo3d_dir = '{self.fargo3d_dir}', setup = '{self.setup}')"""
        return repr

    def __str__(self):
        """
        Detailed string with simulation information.

        Returns
        -------
        str
        """
        str = f"""Simulation information:
    FARGO3D directory: {self.fargo3d_dir}
        Outputs: {self.outputs_dir}
        Setups: {self.setups_dir}
    Units:
        G = {self.G} UL^3/(UM UT^2)
        UL, UM, UT = {self.UL} m, {self.UM} kg, {self.UT} s
        UE = {self.UEPS} J/m^3
        UV = {self.UV} m/s
        URHO = {self.URHO} kg/m^3
        USIGMA = {self.USIGMA} kg/m^2
    Setup: {self.setup}
    Setup directory: {self.setup_dir}
    Output directory: {self.output_dir}
"""
        return str
    


    # ##########################################################################
    # Static method
    # ##########################################################################
    @staticmethod
    def list_setups():
        """
        Print all valid setup directories detected under ``FP_FARGO3D_DIR``.

        Returns
        -------
        None
            The setups are printed to stdout; nothing is returned.
        """
        error,output = fargopy.Sys.run(f"ls -d {fargopy.Conf.FP_FARGO3D_DIR}/setups/*")
        list = ''
        for setup_dir in output[:-1]:
            setup_dir = setup_dir.replace('//','/')
            setup_name = setup_dir.split('/')[-1]
            setup_par = os.path.join(setup_dir, f"{setup_name}.par")
            if os.path.isfile(setup_par):
                list += f"Setup '{setup_name}' in '{setup_dir}'\n"
        print(list)
                 
    @staticmethod
    def list_precomputed():
        """
        Display the catalog of downloadable precomputed simulations with descriptions and sizes.

        Returns
        -------
        None
            The listing is printed to stdout.
        """
        for key,item in PRECOMPUTED_SIMULATIONS.items():
            print(f"{key}:\n\tDescription: {item['description']}\n\tSize: {item['size']} MB")
    
    @staticmethod
    def download_precomputed(setup=None,download_dir=None,quiet=False,clean=True):
        """
        Download and extract a precomputed output archive from the public repository.

        Parameters
        ----------
        setup : str, optional
            Name of the entry in ``PRECOMPUTED_SIMULATIONS``.
        download_dir : str, optional
            Destination directory for the compressed file and extracted output.
        quiet : bool, optional
            When True, suppress download progress indicators.
        clean : bool, optional
            Remove the downloaded archive after extraction when set to True.

        Returns
        -------
        str
            Absolute path to the extracted output directory, or an empty string on failure.
        """
        if setup is None:
            print(f"You must provide a setup name. Available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        
        # Set default download directory based on OS
        if download_dir is None:
            if os.name == 'nt':  # Windows
                download_dir = os.path.join(os.environ.get('TEMP', 'C:\\temp'), 'fargopy_data')
            else:  # Unix-like systems
                download_dir = '/tmp'
            # Create directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)
        
        if not os.path.isdir(download_dir):
            print(f"Download directory '{download_dir}' does not exist.")
            return ''
        if setup not in PRECOMPUTED_SIMULATIONS.keys():
            print(f"Precomputed setup '{setup}' is not among the available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        
        output_dir = os.path.join(download_dir, setup)
        if os.path.isdir(output_dir):
            print(f"Precomputed output directory '{output_dir}' already exist")
            return output_dir
        else:
            filename = setup + '.tgz'
            fileloc = os.path.join(download_dir, filename)
            if os.path.isfile(fileloc):
                print(f"Precomputed file '{fileloc}' already downloaded")
            else:
                # Download the setups
                print(f"Downloading {filename} from cloud (compressed size around {PRECOMPUTED_SIMULATIONS[setup]['size']} MB) into {download_dir}")
                url = PRECOMPUTED_BASEURL + PRECOMPUTED_SIMULATIONS[setup]['id']
                gdown.download(url,fileloc,quiet=quiet)
            
            # Uncompress the setups - Windows compatible
            print(f"Uncompressing {filename} into {output_dir}")
            try:
                import tarfile
                with tarfile.open(fileloc, 'r:gz') as tar:
                    tar.extractall(path=download_dir)
                print(f"Done.")
                
                # Clean up the tar file if requested
                if clean:
                    os.remove(fileloc)
                    
            except Exception as e:
                print(f"Error uncompressing file: {e}")
                # Fallback to system command for Unix-like systems
                if os.name != 'nt':
                    fargopy.Sys.simple(f"cd {download_dir};tar zxf {filename}")
                    if clean:
                        fargopy.Sys.simple(f"cd {download_dir};rm -rf {filename}")
                else:
                    print("Failed to decompress on Windows. Please install tar or 7-zip.")
                    return ''
                    
            return output_dir

    def time_scale(self, scale='orbits'):
        """
        Calculates the time scale of the simulation in different units.

        Parameters
        ----------
        scale : str, optional
            'orbits' to calculate the number of orbits completed by the planet,
            'duration' for the total simulation time in simulation units.

        Returns
        -------
        float
            Number of orbits or total simulation time, depending on scale.
        """
        import contextlib
        import io
        import numpy as np

        with contextlib.redirect_stdout(io.StringIO()):
            # Load necessary parameters
            self.load_macros()
            self.load_planet_summary()

            # Extract parameters from macros and planet summary
            G = self.G  # Gravitational constant in simulation units
            M_star = self.simulation_macros['MSTAR']  # Stellar mass in simulation units
            planet = self.planets[0]  # Assume the first planet for calculations
            a = planet['distance']  # Orbital radius in simulation units

            # Calculate orbital period (T) using Kepler's third law
            T = 2 * np.pi * np.sqrt(a**3 / (G * M_star))

            # Extract simulation parameters
            NINTERM = self._load_variables('variables.par').NINTERM
            DT = self._load_variables('variables.par').DT
            NTOT = self._load_variables('variables.par').NTOT

            # Calculate total simulation time
            total_time = NTOT * DT  # Total time in simulation units

            if scale == 'orbits':
                # Calculate the number of orbits completed by the planet
                orbits_num = total_time / T
                return orbits_num

            elif scale == 'duration':
                # Return the total simulation time in simulation units
                return total_time

            else:
                raise ValueError("Invalid scale. Choose either 'orbits' or 'duration'.")



    def to_paraview(self, snapshot=0, dir='.', basename=None):
        
        """
        Export a snapshot to a VTU (UnstructuredGrid) file with physical XYZ coordinates
        and point data arrays for density and Cartesian velocity.

        Parameters
        ----------
        snapshot : int
            Snapshot index to export (default: 0).
        dir : str
            Output directory where the .vtu file will be written (default: current directory).
        basename : str or None
            Base name for the output file (without extension). If None, a default name
            using the simulation setup and snapshot is created.

        Notes
        -----
        - Requires the 'vtk' Python package (and vtk.util.numpy_support).
        - Expects self.load_field('gasdens') and self.load_field('gasv') to return
          fargopy.Field-like objects where .data is a numpy array with shapes:
            rho: (nt, nr, nphi)
            vel: either (3, nt, nr, nphi) or (nt, nr, nphi, 3)
        - Domains expected in self.domains as theta, r, phi arrays.
        - Output is a single .vtu unstructured grid with VTK_VOXEL cells and point data arrays:
            - "rho" (scalar)
            - "vel_cart" (3-component vector)
        """
        import os
        import numpy as np
        try:
            import vtk
            from vtk.util import numpy_support as ns
        except Exception as e:
            raise RuntimeError("VTK is required for to_paraview. Install the 'vtk' package.") from e

        # prepare output path
        os.makedirs(dir, exist_ok=True)
        if basename is None:
            base = getattr(self, 'setup', 'simulation')
            basename = f"{base}_snap{snapshot}"
            
        filename = os.path.join(dir, basename)

        # Load fields (tolerant to different return types)
        gasdens = self.load_field(fields='gasdens', snapshot=snapshot, interpolate=False)
        gasv = self.load_field(fields='gasv', snapshot=snapshot, interpolate=False)

        
        rho = np.log10(getattr(gasdens, 'data', gasdens)*self.URHO)  # convert to physical units and log10
        vel = getattr(gasv, 'data', gasv)*self.UL/self.UT*1e-5

        # Normalize shapes
        if rho.ndim == 4 and rho.shape[0] == 1:
            rho = rho[0]
        if rho.ndim != 3:
            raise ValueError(f"Unexpected rho shape: {rho.shape}. Expected (nt,nr,nphi).")

        # velocity can be (3,nt,nr,nphi) or (nt,nr,nphi,3)
        if vel.ndim == 4 and vel.shape[0] == 3:
            vel_components = vel
        elif vel.ndim == 4 and vel.shape[-1] == 3:
            vel_components = np.moveaxis(vel, -1, 0)
        else:
            raise ValueError(f"Unexpected vel shape: {vel.shape}. Expected (3,nt,nr,nphi) or (nt,nr,nphi,3).")

        # Get spherical coordinate grids from self.domains
        try:
            theta = self.domains.theta 
            r = self.domains.r  *self.UL/self.AU
            phi = self.domains.phi
        except Exception:
            # If domain attribute names differ, try common alternatives
            try:
                theta = self.domains.theta
                r = self.domains.r*self.UL/self.AU
                phi = self.domains.phi
            except Exception as e:
                raise RuntimeError("Could not find theta, r, phi in self.domains") from e

        # Create meshgrid (vectorized)
        TT, RR, PP = np.meshgrid(theta, r, phi, indexing='ij')  # shape (nt,nr,nphi)

        # Coordinates in Cartesian
        X = (RR * np.sin(TT) * np.cos(PP)).ravel(order='C').astype(np.float64)
        Y = (RR * np.sin(TT) * np.sin(PP)).ravel(order='C').astype(np.float64)
        Z = (RR * np.cos(TT)).ravel(order='C').astype(np.float64)
        pts = np.column_stack([X, Y, Z])

        # Cartesian velocity components (vectorized)
        v_theta = vel_components[0]
        v_r = vel_components[1]
        v_phi = vel_components[2]

        v_x = (v_r * np.sin(TT) * np.cos(PP) +
               v_theta * np.cos(TT) * np.cos(PP) -
               v_phi * np.sin(PP)).ravel(order='C').astype(np.float64)

        v_y = (v_r * np.sin(TT) * np.sin(PP) +
               v_theta * np.cos(TT) * np.sin(PP) +
               v_phi * np.cos(PP)).ravel(order='C').astype(np.float64)

        v_z = (v_r * np.cos(TT) -
               v_theta * np.sin(TT)).ravel(order='C').astype(np.float64)

        vel_cart = np.column_stack([v_x, v_y, v_z])

        # Density flattened
        rho_flat = rho.ravel(order='C').astype(np.float64)

        # VTK points
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(pts, deep=True))

        # Build VOXEL connectivity (vectorized)
        nt, nr, nphi = rho.shape
        ntm = nt - 1
        nrm = nr - 1
        npm = nphi - 1
        ncells = ntm * nrm * npm

        it, ir_, ip = np.meshgrid(
            np.arange(ntm),
            np.arange(nrm),
            np.arange(npm),
            indexing='ij'
        )
        it = it.ravel()
        ir_ = ir_.ravel()
        ip = ip.ravel()
        base = it * nr * nphi + ir_ * nphi + ip

        p000 = base
        p001 = base + 1
        p010 = base + nphi
        p011 = base + nphi + 1
        p100 = base + nr * nphi
        p101 = base + nr * nphi + 1
        p110 = base + nr * nphi + nphi
        p111 = base + nr * nphi + nphi + 1

        cells = np.column_stack([
            np.full(ncells, 8, dtype=np.int64),
            p000, p001, p010, p011,
            p100, p101, p110, p111
        ]).ravel()

        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(ncells, ns.numpy_to_vtkIdTypeArray(cells, deep=True))

        # Unstructured grid
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(vtk_points)
        grid.SetCells(vtk.VTK_VOXEL, vtk_cells)

        # Point data arrays
        vtk_rho = ns.numpy_to_vtk(rho_flat, deep=True)
        vtk_rho.SetName("rho")
        vtk_vel = ns.numpy_to_vtk(vel_cart, deep=True)
        vtk_vel.SetNumberOfComponents(3)
        vtk_vel.SetName("vel_cart")

        grid.GetPointData().AddArray(vtk_rho)
        grid.GetPointData().AddArray(vtk_vel)

        # Write VTU
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(filename + ".vtu")
        writer.SetInputData(grid)
        if writer.Write() == 0:
            raise RuntimeError(f"Failed writing VTU file '{filename}.vtu'.")

        return filename + ".vtu"




    ######################################
    # PLOTS
    #####################################



    def plot_interactive(self):
        """
        Interactive plot for the simulation using ipywidgets.

        Allows selection of density, energy, or velocity (and component) for the colormap.
        Provides controls for slice, resolution, interpolation, streamlines, and Hill radius overlay.
        """
        import ipywidgets as widgets
        import matplotlib.pyplot as plt
        from IPython.display import display, clear_output

        # --- Widgets ---
        time_slider = widgets.IntSlider(min=0, max=self._get_nsnaps()-1, step=1, value=1, description='Snapshot')
        slice_text = widgets.Text(value='theta=1.568', description='Slice')
        res_slider = widgets.IntSlider(min=50, max=1000, step=10, value=500, description='Res')
        interp_toggle = widgets.ToggleButton(value=False, description='Interpolate', icon='check')
        progress = widgets.Label(value='')
        streamlines_toggle = widgets.ToggleButton(value=False, description='Streamlines', icon='random')
        density_slider = widgets.FloatSlider(min=1, max=10, step=0.5, value=3, description='Stream density')
        hill_frac_slider = widgets.FloatSlider(min=0.1, max=2.0, step=0.05, value=1.0, description='Hill frac')
        show_circle_toggle = widgets.ToggleButton(value=False, description='Show Hill', icon='circle')
        cmap_options = ['Spectral_r', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'YlGnBu', 'cubehelix', 'twilight', 'turbo']
        cmap_dropdown = widgets.Dropdown(options=cmap_options, value='Spectral_r', description='Colormap')
        update_button = widgets.Button(description="Update", icon="refresh")
        map_options = ['Densidad', 'Energia', 'Velocidad']
        map_dropdown = widgets.Dropdown(options=map_options, value='Densidad', description='Mapa')
        vel_components = ['vx', 'vy', 'vz']
        vel_dropdown = widgets.Dropdown(options=vel_components, value='vx', description='Componente v')
        vel_dropdown.layout.display = 'none'  # Ocultar por defecto

        def is_fixed(var, slice_str):
            import re
            match = re.search(rf'{var}=([^\[\],]+)', slice_str.replace(' ', ''))
            return match is not None

        # show or hide velocity component dropdown based on map selection
        def on_map_change(change):
            if change['new'] == 'Velocidad':
                vel_dropdown.layout.display = ''
            else:
                vel_dropdown.layout.display = 'none'
        map_dropdown.observe(on_map_change, names='value')

        def plot_density(change=None):
            clear_output(wait=True)
            display(
                time_slider, slice_text, res_slider, interp_toggle, streamlines_toggle,
                density_slider, hill_frac_slider, show_circle_toggle, cmap_dropdown, map_dropdown, vel_dropdown, progress, update_button
            )
            import numpy as np
            import re

            slice_str = slice_text.value
            res = res_slider.value
            interpolate = interp_toggle.value
            show_streamlines = streamlines_toggle.value
            stream_density = density_slider.value
            hill_frac = hill_frac_slider.value
            show_circle = show_circle_toggle.value
            cmap = cmap_dropdown.value
            map_type = map_dropdown.value
            vel_comp = vel_dropdown.value

            # --- Ejes y nombres de malla ---
            if is_fixed('theta', slice_str):
                xlabel, ylabel = 'X', 'Y'
                mesh_x_name = 'var1_mesh'
                mesh_y_name = 'var2_mesh'
            elif is_fixed('phi', slice_str):
                xlabel, ylabel = 'X', 'Z'
                mesh_x_name = 'var1_mesh'
                mesh_y_name = 'var3_mesh'
            else:
                print("Warning: Please fix either theta or phi for a valid 2D slice (XY or XZ plane).")
                return

            n = time_slider.value

            if mesh_y_name == 'var2_mesh':
                vel_dropdown.options = ['vx', 'vy']
                if vel_dropdown.value not in vel_dropdown.options:
                    vel_dropdown.value = 'vx'
            elif mesh_y_name == 'var3_mesh':
                vel_dropdown.options = ['vx', 'vz']
                if vel_dropdown.value not in vel_dropdown.options:
                    vel_dropdown.value = 'vx'

            # --- Carga de datos según selección ---
            if map_type == 'Densidad':
                gasdens, gasv = self.load_field(
                    fields=['gasdens', 'gasv'],
                    slice=slice_str,
                    snapshot=[n],
                    interpolate=True
                )
            elif map_type == 'Energia':
                gasenergy = self.load_field(
                    fields='gasenergy',
                    slice=slice_str,
                    snapshot=[n],
                    interpolate=True
                )
                gasv = self.load_field(
                    fields='gasv',
                    slice=slice_str,
                    snapshot=[n],
                    interpolate=True
                )
            elif map_type == 'Velocidad':
                gasv = self.load_field(
                    fields='gasv',
                    slice=slice_str,
                    snapshot=[n],
                    interpolate=True
                )

            # --- Interpolación y selección de variable a graficar ---
            if not interpolate:
                if map_type == 'Densidad':
                    X = getattr(gasdens, mesh_x_name)[0]
                    Y = getattr(gasdens, mesh_y_name)[0]
                    data_map = np.log10(gasdens.gasdens_mesh[0] * self.URHO)
                elif map_type == 'Energia':
                    X = getattr(gasenergy, mesh_x_name)[0]
                    Y = getattr(gasenergy, mesh_y_name)[0]
                    data_map = np.log10(gasenergy.gasenergy_mesh[0])
                elif map_type == 'Velocidad':
                    X = getattr(gasv, mesh_x_name)[0]
                    Y = getattr(gasv, mesh_y_name)[0]
                    idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                    data_map = gasv.gasv_mesh[0][idx]
                vx = vy = vmag = None
            else:
                progress.value = "Interpolando..."
                if mesh_y_name == 'var2_mesh':
                    xmin, xmax = getattr(gasv, mesh_x_name)[0].min(), getattr(gasv, mesh_x_name)[0].max()
                    ymin, ymax = getattr(gasv, mesh_y_name)[0].min(), getattr(gasv, mesh_y_name)[0].max()
                    xs = np.linspace(xmin, xmax, res)
                    ys = np.linspace(ymin, ymax, res)
                    X, Y = np.meshgrid(xs, ys)
                    if map_type == 'Densidad':
                        data_map = gasdens.evaluate(time=n, var1=X, var2=Y)
                        data_map = np.log10(data_map * self.URHO)
                        vel = gasv.evaluate(time=n, var1=X, var2=Y)
                        vx = vel[0]
                        vy = vel[1]
                        vmag = np.sqrt(vx**2 + vy**2)
                    elif map_type == 'Energia':
                        data_map = gasenergy.evaluate(time=n, var1=X, var2=Y)
                        #data_map = np.log10(data_map)
                        vel = gasv.evaluate(time=n, var1=X, var2=Y)
                        vx = vel[0]
                        vy = vel[1]
                        vmag = np.sqrt(vx**2 + vy**2)
                    elif map_type == 'Velocidad':
                        vel = gasv.evaluate(time=n, var1=X, var2=Y)
                        idx = {'vx': 0, 'vy': 1, 'vz': 2}[vel_comp]
                        data_map = vel[idx]
                        vx = vel[0]
                        vy = vel[1]
                        vmag = np.sqrt(vx**2 + vy**2)
                else:
                    xmin, xmax = getattr(gasv, mesh_x_name)[0].min(), getattr(gasv, mesh_x_name)[0].max()
                    zmin, zmax = getattr(gasv, mesh_y_name)[0].min(), getattr(gasv, mesh_y_name)[0].max()
                    xs = np.linspace(xmin, xmax, res)
                    zs = np.linspace(zmin, zmax, res)
                    X, Y = np.meshgrid(xs, zs)
                    if map_type == 'Densidad':
                        data_map = gasdens.evaluate(time=n, var1=X, var3=Y)
                        data_map = np.log10(data_map * self.URHO)
                        vel = gasv.evaluate(time=n, var1=X, var3=Y)
                        vx = vel[0]
                        vy = vel[2]
                        vmag = np.sqrt(vx**2 + vy**2)
                    elif map_type == 'Energia':
                        data_map = gasenergy.evaluate(time=n, var1=X, var3=Y)
                        #data_map = np.log10(data_map)
                        vel = gasv.evaluate(time=n, var1=X, var3=Y)
                        vx = vel[0]
                        vy = vel[2]
                        vmag = np.sqrt(vx**2 + vy**2)
                    elif map_type == 'Velocidad':
                        vel = gasv.evaluate(time=n, var1=X, var3=Y)
                       
                        idx = {'vx':  0, 'vy': 1, 'vz': 2}[vel_comp]
                        data_map = vel[idx]
                        vx = vel[0]
                        vy = vel[2]
                        vmag = np.sqrt(vx**2 + vy**2)

            # --- Máscara por rango r (igual que antes) ---
            r = np.sqrt(X**2 + Y**2)
            r_match = re.search(r"r=\[([0-9\.]+),([0-9\.]+)\]", slice_str.replace(" ", ""))
            if r_match:
                r_min = float(r_match.group(1))
                r_max = float(r_match.group(2))
            else:
                r_min = None
                r_max = None

            if r_min is not None and r_max is not None:
                mask = (r >= r_min) & (r <= r_max)
                data_map = np.where(mask, data_map, np.nan)
                if show_streamlines and vx is not None and vy is not None and vmag is not None:
                    vx = np.where(mask, vx, np.nan)
                    vy = np.where(mask, vy, np.nan)
                    vmag = np.where(mask, vmag, np.nan)

            # --- Plot ---
            fig, ax = plt.subplots(figsize=(7,5))
            pcm = ax.pcolormesh(X*self.UL/self.AU, Y*self.UL/self.AU, data_map, shading='auto', cmap=cmap)

            # Mostrar streamlines para cualquier tipo de mapa si están disponibles
            stream_obj = None
            if interpolate and show_streamlines and vx is not None and vy is not None:
                stream_obj = ax.streamplot(
                    X*self.UL/self.AU, Y*self.UL/self.AU, vx, vy,
                    color=vmag*self.UL/self.UT*1e-5 if vmag is not None else None,
                    linewidth=0.5,
                    density=stream_density,
                    cmap='viridis',
                    arrowsize=1
                )

            # --- Hill radius
            planets = self.load_planets(snapshot=n)
            if planets:
                center_x = planets[0].pos.x
                center_y = planets[0].pos.y
                radius = hill_frac * planets[0].hill_radius
            else:
                center_x = 0
                center_y = 0
                radius = 0

            if show_circle:
                if is_fixed('theta', slice_str):
                    circle = plt.Circle((center_x*self.UL/self.AU, center_y*self.UL/self.AU), radius*self.UL/self.AU, color='black', fill=False, linestyle='--', linewidth=1)
                    ax.add_patch(circle)
                elif is_fixed('phi', slice_str):
                    theta = np.linspace(0, np.pi, 100)
                    x = center_x + radius * np.cos(theta)
                    y = center_y + radius * np.sin(theta)
                    ax.plot(x*self.UL/self.AU, y*self.UL/self.AU, color='black', linewidth=2)

            ax.set_xlabel(xlabel+' [AU]')
            ax.set_ylabel(ylabel+' [AU]')
            #ax.axis('equal')

            if interpolate and show_streamlines and stream_obj is not None and vmag is not None:
                # Colorbar for velocity magnitude (streamlines)
                cbar = fig.colorbar(stream_obj.lines, ax=ax, label=r'$|v|$ [km/s]')
            else:
                # Colorbar for main map
                if map_type == 'Densidad':
                    cbar_label = r'$\log_{10}(\rho) [g/cm^3]$'
                elif map_type == 'Energia':
                    cbar_label = r'$\log_{10}(\mathrm{energy})$'
                else:
                    cbar_label = f'{vel_comp} [AU]'
                fig.colorbar(pcm, ax=ax, label=cbar_label)

            fargopy.Plot.fargopy_mark(ax)
            plt.show()

        # --- Events ---
        update_button.on_click(plot_density)
        slice_text.on_submit(plot_density)
        show_circle_toggle.observe(plot_density, names='value')
        interp_toggle.observe(plot_density, names='value')
        streamlines_toggle.observe(plot_density, names='value')
        cmap_dropdown.observe(plot_density, names='value')
        map_dropdown.observe(plot_density, names='value')
        vel_dropdown.observe(plot_density, names='value')

        # --- Display inicial ---
        display(
            time_slider, slice_text, res_slider, interp_toggle, streamlines_toggle,
            density_slider, hill_frac_slider, show_circle_toggle, cmap_dropdown, map_dropdown, vel_dropdown, progress, update_button
        )
        plot_density()



    def plot_mesh(self, snapshot=0, slice='theta=1.56', planet=0, draw_hill=True, hill_frac=1.0,
                  figsize=(8,8), point_size=1, line_alpha=0.5, cmap='viridis', show=True):
        """
        Plot the simulation mesh in the XY plane and (optionally) the planet Hill circle.

        Returns
        -------
        (fig, ax, nr_celdas_radial, nr_celdas_azimutal, n_inside)
            Matplotlib figure and axes, max contiguous radial cells, max contiguous azimuthal cells,
            and the number of mesh cells inside the hill_frac * Hill radius.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        # Load a 2D interpolated field (keeps same interface used elsewhere)
        gasdens = self.load_field(fields=['gasdens'], snapshot=snapshot, slice=slice, interpolate=True)

        # Expect interpolator result with var1_mesh / var2_mesh (as used in plot_interactive)
        try:
            X = gasdens.var1_mesh[0]
            Y = gasdens.var2_mesh[0]
        except Exception:
            # Fallback: if a raw Field-like object is returned with mesh names var1_mesh/var2_mesh attributes
            X = getattr(gasdens, 'var1_mesh', None)
            Y = getattr(gasdens, 'var2_mesh', None)
            if X is None or Y is None:
                raise RuntimeError("Could not obtain var1_mesh/var2_mesh from loaded field. Use interpolate=True and a valid slice.")

        # Prepare figure
        plt.close('all')
        fig, ax = plt.subplots(figsize=figsize)

        # Plot points (convert to AU for axis if simulation units defined)
        scale = getattr(self, 'UL', 1.0) / getattr(self, 'AU', 1.0)
        ax.scatter((X * scale).ravel(), (Y * scale).ravel(), s=point_size, c=(X*0+0.5).ravel(),
                   cmap=cmap, marker='.', linewidths=0)

        # If mesh is 2D arrays, draw grid lines
        if X.ndim == 2 and Y.ndim == 2:
            # rows
            for i in range(X.shape[0]):
                ax.plot(X[i, :]*scale, Y[i, :]*scale, color='gray', linewidth=0.5, alpha=line_alpha)
            # columns
            for j in range(X.shape[1]):
                ax.plot(X[:, j]*scale, Y[:, j]*scale, color='gray', linewidth=0.5, alpha=line_alpha)

        # Planet selection
        planets = self.load_planets(snapshot=snapshot)
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
                    if getattr(p, 'name', None) == planet:
                        sel = p
                        break
                if sel is None:
                    sel = planets[0]
            # planet object expected to have pos.x / pos.y and hill_radius property
            center_x = sel.pos.x
            center_y = sel.pos.y
            if draw_hill:
                radius = hill_frac * getattr(sel, 'hill_radius', 0.0)

        # Draw Hill circle if requested and compute counts
        nr_celdas_radial = 0
        nr_celdas_azimutal = 0
        n_inside = 0
        if draw_hill and center_x is not None and center_y is not None and radius > 0:
            circle = patches.Circle((center_x*scale, center_y*scale), radius*scale,
                                    edgecolor='red', facecolor='lightblue', linestyle='-', linewidth=1.5)
            ax.add_patch(circle)

            # Count mesh cells (points) inside the requested fraction of Hill radius
            try:
                # X,Y are in simulation length units (same as center_x, center_y)
                mask_inside = ((X - center_x)**2 + (Y - center_y)**2) <= (radius**2)
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
                    nr_celdas_azimutal = int(max_az)

                    # Radial: along columns (axis 0) -> for each column find longest contiguous True segment
                    max_rad = 0
                    for j in range(mask_inside.shape[1]):
                        l = max_contiguous_true(mask_inside[:, j])
                        if l > max_rad:
                            max_rad = l
                    nr_celdas_radial = int(max_rad)

                else:
                    # Unstructured/1D mesh: only total count is reliable
                    nr_celdas_radial = 0
                    nr_celdas_azimutal = 0

            except Exception:
                nr_celdas_radial = nr_celdas_azimutal = 0
                n_inside = 0

        ax.set_xlabel("X [AU]")
        ax.set_ylabel("Y [AU]")
        ax.set_title(f"Simulation Mesh")
        ax.set_aspect('equal', adjustable='box')
        fargopy.Plot.fargopy_mark(ax)

         # Print statistics

        print(f'Number of mesh cells inside {hill_frac} x Hill radius: {n_inside}')
        print(f'Max contiguous radial cells inside: {nr_celdas_radial}')
        print(f'Max contiguous azimutal cells inside: {nr_celdas_azimutal}')
        if show:
            plt.show()

        return fig, ax

class Planet:
    """
    Represents a planet in the simulation, holding its physical state and properties.

    Attributes
    ----------
    name : str
        Name or label of the planet.
    mass : float
        Planet mass (in Msun or simulation units).
    pos : Planet.Vector
        Current cartesian position (x, y, z) in AU.
    vel : Planet.Vector
        Current cartesian velocity (vx, vy, vz) in AU/UT.
    posi : Planet.Vector
        Initial cartesian position (x, y, z) in AU.
    mstar : float
        Stellar mass (in Msun or simulation units).

    Properties
    ----------
    hill_radius : float
        The Hill radius of the planet (in AU), computed from its current position and mass.

    Example
    -------
    >>> jupiter = planets[0]
    >>> print(jupiter.pos.x, jupiter.vel.y, jupiter.hill_radius)
    """
    def __init__(self, name, pos, vel, mass, posi, mstar):
        self.name = name
        self.mass = mass
        self.pos = Planet._Vector(*pos)
        self.vel = Planet._Vector(*vel)
        self.posi = Planet._Vector(*posi)
        self.mstar = mstar

    class _Vector:
        """
        Simple vector class for 3D coordinates, allowing attribute and index access.

        Attributes
        ----------
        x : float
            X coordinate.
        y : float
            Y coordinate.
        z : float
            Z coordinate.
        """
        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z

        def __getitem__(self, idx):
            """
            Return the coordinate at index ``idx`` (0→x, 1→y, 2→z).
            """
            return [self.x, self.y, self.z][idx]

        def __array__(self):
            """
            Expose the vector as a NumPy array for downstream numerical routines.
            """
            import numpy as np
            return np.array([self.x, self.y, self.z])

        def __repr__(self):
            """
            Debug-friendly string showing the three Cartesian components.
            """
            return f"[{self.x}, {self.y}, {self.z}]"

    @property
    def hill_radius(self):
        """
        .. :no-index:

        Returns the Hill radius in AU, using the current position and mass.

        Returns
        -------
        float
            Hill radius in AU.
        """
        AU_to_cm = 1.495978707e13
        Mjup_to_g = 1.898e30
        Msun_to_g = 1.989e33

        # Distance to star (AU)
        r_au = (self.pos.x**2 + self.pos.y**2 + self.pos.z**2)**0.5
        m_jup = self.mass * 1e3  # Msun to Mjup
        mstar_g = float(self.mstar) * Msun_to_g
        r_cm = r_au * AU_to_cm
        m_p = m_jup * Mjup_to_g

        r_hill_cm = r_cm * (m_p / (3 * mstar_g))**(1/3)
        r_hill_au = r_hill_cm / AU_to_cm
        return r_hill_au

    def __repr__(self):
        """
        String representation of the Planet object.

        Returns
        -------
        str
        """
        return f"Planet(name={self.name}, mass={self.mass}, pos={self.pos}, vel={self.vel}, posi={self.posi})"
