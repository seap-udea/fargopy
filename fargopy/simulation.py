###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import os
import numpy as np
import re
import subprocess
import time
import gdown
import os

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
    fargo=dict(id='1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3',size=55),
    # Download link: https://drive.google.com/file/d/1KMp_82ylQn3ne_aNWEF1T9ElX2aWzYX6/view?usp=drive_link
    p3diso=dict(id='1KMp_82ylQn3ne_aNWEF1T9ElX2aWzYX6',size=220),
    # Download link: https://drive.google.com/file/d/14mL2KCcCtjptChiyISGyJxAEOl4KAanI/view?usp=drive_link
    p3disof=dict(id='14mL2KCcCtjptChiyISGyJxAEOl4KAanI',size=440),
    # Download link: https://drive.google.com/file/d/1KSQyxH_kbAqHQcsE30GQFRVgAPhMAcp7/view?usp=drive_link
    fargo_multifluid=dict(id='1KSQyxH_kbAqHQcsE30GQFRVgAPhMAcp7',size=100),
    # Download link: https://drive.google.com/file/d/12ZWoQS_9ISe6eDij5KWWbqR-bHyyVs2N/view?usp=drive_link
    binary=dict(id='12ZWoQS_9ISe6eDij5KWWbqR-bHyyVs2N',size=140),
)

###############################################################
# Classes
###############################################################
class Simulation(fargopy.Fargobj):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        # Load simulation configuration from a file

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
        self.set_property('fargo3d_binary',
                          None)
        self.set_property('fargo3d_compilation_options',
                          dict(parallel=0,gpu=0,options=''))
        self.set_property('fargo3d_process',
                          None)
        self.set_property('logfile',
                          None)

    # ##########################################################################
    # Set special properties
    # ##########################################################################  
    def set_fargo3d_dir(self,dir=None):
        """Set fargo3d directory

        Args:
            dir: string, default = None:
                Directory where FARGO3D is installed.

        Returns:
            True if the FARGO3D directory exists and the file
            'src/<fargo_header>' is found. False otherwise.
        """
        if dir is None:
            return
        if not os.path.isdir(dir):
            print(f"FARGO3D directory '{dir}' does not exist.")
            return
        else:
            fargo_header = f"{dir}/{fargopy.Conf.FP_FARGO3D_HEADER}".replace('//','/')
            if not os.path.isfile(fargo_header):
                print(f"No header file for FARGO3D found in '{fargo_header}'")
            else:
                print(f"Your simulation is now connected with '{dir}'")
        
        # Set derivative dirs
        self.fargo3d_dir = dir
        self.outputs_dir = (self.fargo3d_dir + '/outputs').replace('//','/')
        self.setups_dir = (self.fargo3d_dir + '/setups').replace('//','/')
    
    def set_setup(self,setup):
        """Connect the simulation to a given setup.

        Args:
            setup: string:
                Name of the setup.
        
        Returns:
            True if the setup_dir <faro3d_dir>/setups/<setup> is found. 
            False otherwise.
        """
        if setup is None:
            self.setup_dir = None
            return None
        setup_dir = f"{self.setups_dir}/{setup}".replace('//','/')
        if self.set_setup_dir(setup_dir):
            self.setup = setup
        return setup
    
    def set_setup_dir(self,dir):
        """Set setup directory

        Args:
            dir: string:
                Directory where setup is available.

        Returns:
            True if the FARGO3D directory exists and the file
            <fargo3d_dir>/src/<fargo_header> is found. False otherwise.
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
        if dir is None:
            return
        if not os.path.isdir(dir):
            print(f"Output directory '{dir}' does not exist.")
            return
        else:
            print(f"Now you are connected with output directory '{dir}'")
        self.output_dir = dir
        return

    def set_units(self,UM=MSUN,UL=AU,G=1,mu=2.35):
        """Set units of the simulation
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
        """Compile FARGO3D binary
        """
        if setup is not None:
            if not self.set_setup(setup):
                print("Failed")
                return 

        # Clean directrory
        if force:
            print(f"Cleaning FARGO3D directory {self.fargo3d_dir}...")
            cmd = f"make -C {self.fargo3d_dir} clean mrproper"
            compl = f"rm -rf {self.fargo3d_dir}/fargo3d_*"
            error,self.output_clean = fargopy.Sys.run(cmd + '&&' + compl)
            
        # Prepare compilation
        compile_options = f"SETUP={self.setup} PARALLEL={parallel} GPU={gpu} "+options
        fargo3d_binary = f"fargo3d_{compile_options.replace(' ','_').replace('=','-').strip('_')}"

        # Compile binary
        print(f"Compiling {fargo3d_binary}...")
        cmd = f"cd {self.fargo3d_dir};make {compile_options}"
        compl = f"mv fargo3d {fargo3d_binary}"
        error,self.output_compilation = fargopy.Sys.run(cmd+' && '+compl)

        # Check compilation result
        if os.path.isfile(f"{self.fargo3d_dir}/{fargo3d_binary}"):
            self.fargo3d_binary = fargo3d_binary
            print(f"Succesful compilation of FARGO3D binary {self.fargo3d_binary}")
            self.fargo3d_compilation_options=dict(
                parallel=parallel,
                gpu=gpu,
                options=options
            )
            return True
        else:
            print(f"Something failed when compiling FARGO3D. For details check Simulation.output_compilation")
            return False
        
    def run(self,
            mode='async',
            options='-m',
            mpioptions='-np 1',
            resume=False,
            cleanrun=False,
            test=False):

        if self.fargo3d_binary is None:
            print("You must first compile your simulation with: <simulation>.compile(<option>).")
            return

        if self._is_running():
            print(f"There is a running process. Please stop it before running/resuming")
            return

        # Mandatory options
        options = options + " -t"
        if 'run_options' not in self.__dict__.keys():
            self.fargo3d_run_options = options
        
        self.logfile = f"{self.setup_dir}/{self.setup}.log"

        # Clean output if available
        if cleanrun:
            # Check if there is an output director
            output_dir = f"{self.outputs_dir}/{self.setup}"
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
        
        if mode == 'sync':
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
                process = subprocess.Popen(run_cmd.split(),cwd=self.fargo3d_dir,
                                        stdout=logfile_handler,stderr=logfile_handler)
                # Introduce a short delay to verify if the process has failed
                time.sleep(1.0)

                if process.poll() is None:
                    # Check if program is effectively running
                    self.fargo3d_process = process            
                    
                    # Create a lock on fargopy with the process id
                    # fargopy.lock(self.frago3d_process.pid)

                    # Setup output directory 
                    self.set_output_dir(f"{self.outputs_dir}/{self.setup}".replace('//','/'))    
                else:
                    print(f"Process running failed. Please check the logfile {self.logfile}")
                    
    def stop(self):
        if not self._check_process():
            return

        poll = self.fargo3d_process.poll()
        if poll is None:
            print(f"Stopping FARGO3D process (pid = {self.fargo3d_process.pid})")
            subprocess.Popen.kill(self.fargo3d_process)
            del self.fargo3d_process
            self.fargo3d_process = None
        else:
            print(f"The process has already finished. Check logfile {self.logfile}.")

    def _save_simultation(self):
        """Save simulation configuration 
        """
        pass

    def status(self,mode='isrunning',verbose=True):
        """Check the status of the running process

        Parameters:
            mode: string, defaul='isrunning':
                Available modes:
                    'isrunning': Just show if the process is running.
                    'logfile': Show the latest lines of the logfile
                    'outputs': Show (and return) a list of outputs
                    'snapshots': Show (and return) a list of snapshots
                    'progress': Show progress in realtime

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

        if 'progress' in mode:
            self._status_progress()

    def _status_progress(self,minfreq=0.1,numstatus=5):
        """Show a progress of the execution

        Parameters:
            minfreq: float, default = 0.1:
                Minimum amount of seconds between status check.

            numstatus: int, default = 5:
                Number of status shown before automatically stopping.
        """
        # Prepare
        frequency = minfreq
        previous_output = ''
        previous_resumable_snapshot = 1e100
        time_previous = time.time()

        # Infinite loop checking for output
        i = 0
        while True and (i<numstatus):
            if not self._is_running():
                print("The simulation is not running anymore")
                return
            error,output = fargopy.Sys.run(f"grep OUTPUT {self.logfile} |tail -n 1")
            
            if not error:
                # Get the latest output
                latest_output = output[-2]
                if latest_output != previous_output:
                    print(f"{latest_output} [output pace = {frequency:.1f} secs]")
                    # Fun the number of the output
                    find = re.findall(r'OUTPUTS\s+(\d+)',latest_output)
                    resumable_snapshot = int(find[0])
                    # Get the time elapsed since last status check
                    time_now = time.time()
                    frequency = max(time_now - time_previous,minfreq)/2
                    if (resumable_snapshot - previous_resumable_snapshot)>1:
                        # Reduce frequency if snapshots are accelerating
                        frequency = frequency/2
                    previous_resumable_snapshot = resumable_snapshot
                    time_previous = time_now
                previous_output = latest_output
            try:
                time.sleep(frequency)
                i += 1
            except KeyboardInterrupt:
                return

    def resume(self,snapshot=-1,mpioptions='-np 1'):
        latest_snapshot_resumable = self._is_resumable()
        if latest_snapshot_resumable<0:
            return
        if self._is_running():
            print(f"There is a running process. Please stop it before resuming")
            return
        if self._has_finished():
            return
        # Resume
        if snapshot<0:
            snapshot = latest_snapshot_resumable
        print(f"Resuming from snapshot {snapshot}...")
        self.run(mode='async',mpioptions=mpioptions,resume=True,
                 options=self.fargo3d_run_options+f' -S {snapshot}',test=False)

    def _has_finished(self):
        if self.fargo3d_process:
            poll = self.fargo3d_process.poll()
            if poll is None:
                return False
            else:
                print(f"The process has ended with termination code {poll}.")
                return True

    def _is_resumable(self):
        if self.logfile is None:
            print(f"The simulation has not been ran yet. Run <simulation>.run() before resuming")
            return -1
        latest_snapshot_resumable = max(self._get_nsnaps() - 2, 0)
        return latest_snapshot_resumable
        
    def clean_output(self):
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
        if self.fargo3d_process is None:
            print(f"There is no FARGO3D process handler available.")
            return False
        return True

    # ##########################################################################
    # Operations on the FARGO3D directories
    # ##########################################################################  
    def list_outputs(self,quiet=False):
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

    def load_properties(self,quiet=False,
                        varfile='variables.par',
                        domain_prefix='domain_',
                        dimsfile='dims.dat'
                        ):
        if self.output_dir is None:
            print(f"You have to set first the outputs directory with <sim>.set_outputs('<directory>')")

        # Read variables
        vars = self._load_variables(varfile)
        print(f"Simulation in {vars.DIM} dimensions")
        
        # Read domains 
        domains = self._load_domains(vars,domain_prefix)

        # Store the variables in the object
        self.vars = vars
        self.domains = domains

        # Optionally read dims
        dims = self._load_dims(dimsfile)
        if len(dims):
            self.dims = dims 

        # Read the summary files
        self.nsnaps = self._get_nsnaps()
        print(f"Number of snapshots in output directory: {self.nsnaps}")
    
        print("Configuration variables and domains load into the object. See e.g. <sim>.vars")

    def _get_nsnaps(self):
        """Get the number of snapshots in an output directory
        """
        error,output = fargopy.Sys.run(f"ls {self.output_dir}/summary[0-9]*.dat")
        if error == 0:
            files = output[:-1]
            nsnaps = len(files)
            return nsnaps
        else:
            print(f"No summary file in {self.output_dir}")
            return 0

    def _load_dims(self,dimsfile):
        """Parse the dim directory
        """
        dimsfile = f"{self.output_dir}/{dimsfile}".replace('//','/')
        if not os.path.isfile(dimsfile):
            #print(f"No file with dimensions '{dimsfile}' found.")
            return []
        dims = np.loadtxt(dimsfile)
        return dims

    def _load_variables(self,varfile):
        """Parse the file with the variables
        """

        varfile = f"{self.output_dir}/{varfile}".replace('//','/')
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
            domain_file = f"{self.output_dir}/{domain_prefix}{variable_suffix}.dat".replace('//','/')
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

    def load_field(self,field,snapshot=None,type='scalar'):

        if not self.has('vars'):
            # If the simulation has not loaded the variables
            dims, vars, domains = self.load_properties()
        
        # In case no snapshot has been provided use 0
        snapshot = 0 if snapshot is None else snapshot

        field_data = []
        if type == 'scalar':
            file_name = f"{field}{str(snapshot)}.dat"
            file_field = f"{self.output_dir}/{file_name}".replace('//','/')
            field_data = self._load_field_scalar(file_field)
        elif type == 'vector':
            field_data = []
            variables = ['x','y'] 
            if self.vars.DIM == 3:
                variables += ['z'] 
            for i,variable in enumerate(variables):
                file_name = f"{field}{variable}{str(snapshot)}.dat"
                file_field = f"{self.output_dir}/{file_name}".replace('//','/')
                field_data += [self._load_field_scalar(file_field)]
        else:
            raise ValueError(f"fargopy.Field type '{type}' not recognized.")

        field = fargopy.Field(data=np.array(field_data), coordinates=self.vars.COORDINATES, domains=self.domains, type=type)
        return field
    
    def _load_field_scalar(self,file):
        """Load scalar field from file a file.
        """
        if os.path.isfile(file):
            field_data = np.fromfile(file).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            """
            if self.vars.NZ > 1:
                # 3D field
                field_data = np.fromfile(file).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            else:
                # 2D field
                field_data = np.fromfile(file).reshape(int(self.vars.NY),int(self.vars.NX))
            """
            return field_data
        else:
            raise AssertionError(f"File with field '{file}' not found")
        
    def load_allfields(self,fluid,snapshot=None,type='scalar'):
        """Load all fields in the output
        """
        qall = False
        if snapshot is None:
            qall = True
            fields = fargopy.Dictobj()
        else:
            fields = fargopy.Dictobj()
        
        # Search for field files
        pattern = f"{self.output_dir}/{fluid}*.dat"
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
                                file_field = f"{self.output_dir}/{file_name}".replace('//','/')
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
                                    file_field = f"{self.output_dir}/{file_name}".replace('//','/')
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
        basename = os.path.basename(file_field)
        comps = None
        match = re.match('([a-zA-Z]+)(\d+).dat',basename)
        if match is not None:
            comps = [match.group(i) for i in range(1,match.lastindex+1)]
        return comps

    def __repr__(self):
        return self.__str__()

    def __str__(self):
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
    def download_precomputed(setup=None,download_dir='/tmp',quiet=True,clean=True):
        """Download a precomputed output from Google Drive FARGOpy public repository.

        Args:
            setup: string, default = None:
                Name of the setup. For a list see fargopu.PRECOMPUTED_SIMULATIONS dictionary.

            download_dir: string, default = '/tmp':
                Directory where the output will be downloaded and uncompressed.

        Optional args:
            quiet: bool, default = True:
                If True download quietly (no progress bar).
            
            clean: bool, default = False:
                If True remove the tgz file after uncompressing it.

        Return:
            If successful returns the output directory.

        """
        if setup is None:
            print(f"You must provide a setup name. Available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        if not os.path.isdir(download_dir):
            print(f"Download directory '{download_dir}' does not exist.")
            return ''
        if setup not in PRECOMPUTED_SIMULATIONS.keys():
            print(f"Precomputed setup '{setup}' is not among the available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        
        output_dir = (download_dir + '/' + setup).replace('//','/')
        if os.path.isdir(output_dir):
            print(f"Precomputed output directory '{output_dir}' already exist")
            return output_dir
        else:
            filename = setup + '.tgz'
            fileloc = download_dir + '/' + filename
            if os.path.isfile(fileloc):
                print(f"Precomputed file '{fileloc}' already downloaded")
            else:
                # Download the setups
                print(f"Downloading {filename} from cloud (compressed size around {PRECOMPUTED_SIMULATIONS[setup]['size']} MB) into {download_dir}")
                url = PRECOMPUTED_BASEURL + PRECOMPUTED_SIMULATIONS[setup]['id']
                gdown.download(url,fileloc,quiet=quiet)
            # Uncompress the setups
            print(f"Uncompressing {filename} into {output_dir}") 
            fargopy.Sys.simple(f"cd {download_dir};tar zxf {filename}")
            print(f"Done.")
            fargopy.Sys.simple(f"cd {download_dir};rm -rf {filename}")
            return output_dir


    
        