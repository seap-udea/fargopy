###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import subprocess
import os
import re
import time
import numpy as np

###############################################################
# Module constants
###############################################################

###############################################################
# Classes
###############################################################
class Simulation(fargopy.Conf):
    """Simulation class.
    """
    def __init__(self,**kwargs):
        super().__init__()

        if 'setup' in kwargs.keys():
            self.load_setup(setup=kwargs['setup'])
        else:
            self.setup = None

        # Compilation options
        self.parallel = fargopy.Conf.FARGO3D_PARALLEL
        self.gpu = fargopy.Conf.FARGO3D_GPU

        # Other options
        self.compile_options = None
        self.fargo3d_binary = None
        self.fargo3d_process = None

    def load_setup(self,setup=None):
        if not setup:
            setups = self.list_setups()
            print(f"You must select a setup, from a custom directory or from the list:{setups}")
        
        output = fargopy.Conf._check_setup(setup)
        if output:
            self.setup = setup
            self.setup_directory = output[0]
            self.setup_parameters = output[1]
            self.setup_outputs = f"{self.setup_directory}/../../outputs/{self.setup}"
            self.logfile = f"{fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_directory}/{self.setup}.log"

    def compile(self,
                parallel=None,gpu=None,
                options='',
                force=False
                ):
        if not self._is_setup():
            return
        
        parallel = self.parallel if parallel is None else parallel
        gpu = self.gpu if gpu is None else gpu

        self.compile_options = f"SETUP={self.setup} PARALLEL={parallel} GPU={gpu} "+options
        self.fargo3d_binary = fargopy.Conf._compile_fargo3d(options=self.compile_options,force=force)
        
    def run(self,
            mode='async',
            options='-m',
            mpioptions = '-np 1',
            resume=False
            ):
        
        if not self._is_setup():
            return
        if self._is_running():
            return 
        
        if self.fargo3d_binary is None:
            print("You must first compile your simulation with: <simulation>.compile().")
            return
        
        # Mandatory options
        options = options + " -t"
        if 'run_options' not in self.__dict__.keys():
            self.run_options = options
        
        self.logfile = f"{fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_directory}/{self.setup}.log"

        # Select command to run
        precmd=''
        if fargopy.Conf.FARGO3D_PARALLEL:
            precmd = f"mpirun {mpioptions} "

        self.fargo3d_process = fargopy.Conf._run_fargo3d(f"{precmd}./{self.fargo3d_binary} {options}",
                                                         logfile=self.logfile,
                                                         mode=mode,
                                                         resume=resume,
                                                         options=self.setup_parameters)
        
    def resume(self,since=0,mpioptions = '-np 1'):
        """Resume a simulation from a given snapshot
        """
        if not self._is_setup():
            return
        if not self._is_resumable():
            return 
        if self._is_running():
            return 
        
        print(f"Resuming from snapshot {since}")
        self.run(mode='async',mpioptions=mpioptions,resume=True,options=self.run_options+f' -S {since}')

    def _is_resumable(self):
        outputs_directory = f"{fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/"
        if not os.path.isdir(outputs_directory):
            print("There is no output to resume at '{outputs_directory.replace('//','/')}'.")
            return False
        return True
    
    def _get_snapshots(self):
        if not self._is_setup():
            return
        if not self._is_resumable():
            return
        
        error,output = fargopy.Util.sysrun(f"grep OUTPUT {self.logfile}",verbose=False)
        if not error:
            snapshots = output[:-1]
        else:
            snapshots = []
        return snapshots
    
    def get_resumable_snapshot(self):
        error,output = fargopy.Util.sysrun(f"grep OUTPUT {self.logfile}",verbose=False)
        if not error:
            find = re.findall(r'OUTPUTS\s+(\d+)',output[-2])
            resumable_snapshot = int(find[0])
        else:
            resumable_snapshot = 0
        return resumable_snapshot
        
    def _is_running(self,verbose=True):
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

    def stop(self):
        """Stop the present running process
        """
        if not self._is_setup():
            return
        
        if not self._is_running():
            print("The process is stopped.")
            return
        
        if self.fargo3d_process:
            poll = self.fargo3d_process.poll()
            if poll is None:
                print(f"Stopping FARGO3D process (pid = {self.fargo3d_process.pid})")
                subprocess.Popen.kill(self.fargo3d_process)
                del self.fargo3d_process
                self.fargo3d_process = None
            else:
                print("The process has already finished.")    

    def _is_setup(self):
        if self.setup is None:
            print("The simulation has not been set-up yet. Run <simulation>.load_setup('<setup>')")
            return False
        return True

    def status(self,
               mode='isrunning',
               verbose=True
               ):
        """Check the status of the running process

        Parameters:
            mode: string, defaul='isrunning':
                Available modes:
                    'isrunning': Just show if the process is running.
                    'logfile': Show the latest lines of the logfile
                    'outputs': Show (and return) a list of outputs
                    'snapshots': Show (and return) a list of snapshots

        """
        if not self._is_setup():
            return
        
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
            error,output = fargopy.Util.sysrun(f"ls {fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*.dat",verbose=False)
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

        if 'snapshots' in mode or mode=='all':
            vprint(bar+"Snapshots:")
            self.output_snapshots = self._get_snapshots()
            nsnapshots = len(self.output_snapshots)
            self.resumable_snapshot = None
            if nsnapshots:
                vprint(f"\tNumber of available snapshots: {nsnapshots}")
                if nsnapshots > 1:
                    find = re.findall(r'OUTPUTS\s+(\d+)',self.output_snapshots[-2])
                    self.resumable_snapshot = int(find[0])
                    vprint(f"\tLatest resumable snapshot: {self.resumable_snapshot}")
                else:
                    vprint(f"\tNo resumable snapshot")
            else:
                vprint("No snapshots yet")

        if 'progress' in mode:
            self._status_progress()

    def _status_progress(self,minfreq =0.1):
        """Show a progress of the execution

        Parameters:
            minfreq: float, default = 0.1:
                Minimum amount of seconds between status check.
        """
        # Prepare
        frequency = minfreq
        previous_output = ''
        previous_resumable_snapshot = 1e100
        time_previous = time.time()

        # Infinite loop checking for output
        while True:
            if not self._is_running(verbose=False):
                print("The simulation is not running anymore")
                return
            error,output = fargopy.Util.sysrun(f"grep OUTPUT {self.logfile} |tail -n 1",verbose=False)
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
            except KeyboardInterrupt:
                return

    def clean_output(self):
        """
        Clean all outputs
        """
        if not self._is_setup():
            return
        if self._is_running():
            return 
        
        print(f"Cleaning simulation outputs...")
        run_cmd = f"rm -r {fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*"
        error,output = fargopy.Util.sysrun(run_cmd,verbose=False)
        if error:
            print("Output directory is clean already.")
        if not error:
            print("Done.")

    def load_domain(self):

        # Load    
        self.dim = 0
        self.domains = []

        print(f"Loading domain...")
        print("Domain size:")
        output_dir = fargopy.Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
        file_x = output_dir + "/domain_x.dat"
        if os.path.isfile(file_x):
            self.domain_x = np.genfromtxt(file_x)
            self.dim += 1
            self.domains += [self.domain_x]
            print(f"\tVariable 1 (periodic): {len(self.domain_x)}")
        file_y = output_dir + "/domain_y.dat"
        if os.path.isfile(file_y):
            self.domain_y = np.genfromtxt(file_y)[3:-3]
            self.dim += 1
            self.domains += [self.domain_y]
            print(f"\tVariable 2: {len(self.domain_y)}")
        file_z = output_dir + "/domain_z.dat"
        if os.path.isfile(file_z):
            self.domain_z = np.genfromtxt(file_z)[3:-3]
            self.dim += 1
            self.domains += [self.domain_z]
            print(f"\tVariable 3: {len(self.domain_z)}")

        print(f"Problem in {self.dim} dimensions")
        
        # Get grid
        print(f"Building vargrids...")
        self.var1_12, self.var2_12 = np.meshgrid(self.domain_x,self.domain_y)
        self.var1_12, self.var3_12 = np.meshgrid(self.domain_x,self.domain_z)
        self.var2_23, self.var3_23 = np.meshgrid(self.domain_y,self.domain_z)
        print(f"Done.")

    def load_variables(self):
        output_dir = fargopy.Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
        variables = np.genfromtxt(output_dir+"/variables.par",
                                    dtype={'names': ("parameters","values"),
                                            'formats': ("|S30","|S300")}).tolist()
        print(f"Loading variables")
        self.vars = dict()
        for posicion in variables:
            self.vars[posicion[0].decode("utf-8")] = posicion[1].decode("utf-8")
        self.vars = fargopy.Dictobj(dict=self.vars)
        print(f"{len(self.vars.__dict__.keys())} variables load. See <sim>.vars")

    def load_field(self,fluid='gas',field='dens',snapshot=None):
        if 'vars' not in self.__dict__.keys():
            self.load_variables()
        if not self._is_resumable():
            print("Simulation has not produced any output yet.")
        if snapshot is None:
            snapshot = self._get_resumable_snapshot()

        output_dir = fargopy.Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
        file_name = fluid+field+str(snapshot)+".dat"
        file_field = output_dir+"/"+file_name
        if os.path.isfile(file_field):
            self.dim = 3 if 'NZ' in self.vars.__dict__.keys() else 2
            if self.dim == 2:
                field = np.fromfile(file_field).reshape(int(self.vars.NY),int(self.vars.NX))
            else:
                field = np.fromfile(file_field).reshape(int(self.vars.NZ),int(self.vars.NY),int(self.vars.NX))
            return field
        else:
            print(f"File with field '{file_name}' not found")