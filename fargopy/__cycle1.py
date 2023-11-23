###############################################################
# Import montu modules
###############################################################
from fargopy.version import *

###############################################################
# Required packages
###############################################################
import os
import subprocess
import inspect
import signal

# Remove zombie subprocesses
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

###############################################################
# Module constants
###############################################################
FARGOPY_CONFIGURATION = dict(
    VERBOSE = False,
    FARGO3D_CLONE_REPO_CMD = 'git clone https://bitbucket.org/fargo3d/public.git',
    FARGO3D_BASEDIR = './',
    FARGO3D_PACKDIR = 'public/',
    FARGO3D_FULLDIR = './public/',
    FARGO3D_HEADER = 'src/fargo3d.h',
    FARGO3D_BINARY = 'fargo3d',
    SYS_STDOUT = '',
    SYS_STDERR = '',
)

###############################################################
# Classes
###############################################################

#/////////////////////////////////////
# UTIL CLASS
#/////////////////////////////////////
class Util(object):

    QERROR = True
    STDERR = ''
    STDOUT = ''

    @staticmethod
    def get_methods(my_class):
        """Get a list of the methods for class my_class
        """
        return sorted([member[0] for member in inspect.getmembers(my_class) if '__' not in member[0]])

    @staticmethod
    def data_path(filename,check=False):
        """Get the full path of the `datafile` which is one of the datafiles provided with the package.
        
        Parameters:
            filename: Name of the data file, string.
            
        Return:
            Full path to package datafile in the python environment.
            
        """
        file_path = os.path.join(os.path.dirname(__file__),'data',filename)
        if check and (not os.path.isfile(file_path)):
            raise ValueError(f"File '{filename}' does not exist in data directory")
        return file_path

    @staticmethod
    def sysrun(cmd):
        if Conf.VERBOSE:
            print(f"::FARGOpy:: Running: {cmd}")
        os.system(cmd)

    @staticmethod
    def _run(cmd):
        p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=True)
        while True:
            line = p.stdout.readline().rstrip()
            if not line:
                break
            yield line
        (output,error)=p.communicate()
        yield p.returncode,error

    @staticmethod
    def sysrun(cmd,verbose=True):
        """Run a system command
        """

        if Conf.VERBOSE:
            print(f"::FARGOpy::sysrun::cmd = {cmd}")
        
        out=[]
        for path in Util._run(cmd):
            try:
                if verbose:
                    print(path.decode('utf-8'))
                out += [path.decode('utf-8')]
            except:
                out += [(path[0],path[1].decode('utf-8'))]
        
        Util.STDOUT = ''
        if len(out)>1:
            Util.STDOUT = '\n'.join(out[:-1])
        
        Util.STDERR = out[-1][1]
        if len(Util.STDERR)>0:
            Util.QERROR = out[-1][0]
            if Util.QERROR == 0:
                Util.QERROR = -1
        else:
            Util.QERROR = 0

        if Util.QERROR and verbose:
            print(f"Error:\n{Util.STDERR}")

        if Conf.VERBOSE:
            error = out[-1][0]
            if Util.QERROR>0:
                print(f"::FARGOpy::sysrun::Error check Util.STDERR.")
            elif Util.QERROR<0:
                print(f"::FARGOpy::sysrun::Done. Still some issues must be check. Check Util.STDOUT and Util.STDERR for details.")
            elif Util.QERROR==0:
                print(f"::FARGOpy::sysrun::Done. You're great. Check Util.STDOUT for details.")
        
        return Util.QERROR,out

#/////////////////////////////////////
# CONFIGURATION CLASS
#/////////////////////////////////////
class Conf(object):
    """Configuration class.
    """
    VERBOSE = False
    FARGO3D_CLONE_REPO_CMD = 'git clone https://bitbucket.org/fargo3d/public.git'
    FARGO3D_BASEDIR = './'
    FARGO3D_PACKDIR = 'public/'
    FARGO3D_SETUPS = 'setups/'
    FARGO3D_FULLDIR = './public/'
    FARGO3D_HEADER = 'src/fargo3d.h'
    FARGO3D_BINARY = 'fargo3d'
    FARGO3D_IS_HERE = False
    FARGO3D_IS_COMPILING = False
    FARGO3D_IS_COMPILING_PARALLEL = False
    FARGO3D_IS_COMPILING_GPU = False
    FARGO3D_PARALLEL = 0
    FARGO3D_GPU = 0
    SYS_STDOUT = ''
    SYS_STDERR = ''

    def __init__(self):
        # Common attributes
        self.parallel = 0
        self.gpu = 0

    def compile_fargo3d(self,clean=True):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):    
            if clean:
                Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)
            
            error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} PARALLEL={self.parallel} GPU={self.gpu}',verbose=False)
            if error:
                if not Conf._check_fargo_binary(Conf.FARGO3D_FULLDIR,quiet=True):
                    print("An error compiling the code arose. Check dependencies.")
                    print(Util.STDERR)
                return False
            
            return True

    def list_setups(self):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):
            error,output = Util.sysrun(f"ls {Conf.FARGO3D_FULLDIR}/{Conf.FARGO3D_SETUPS}",verbose=False)
            if error:
                print("No setups found")
                return []
            else:
                setups = output[:-1]
                return setups

    @staticmethod
    def clean_fargo3d():
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):
            error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)

    @staticmethod
    def set_fargo3d(basedir=None,react='get compile parallel gpu'):
        """Check if FARGO3D is properly installed in the sysem

        Parameters:
            basedir: string, default = None:
                Set the basedir where FARGO3D installation will be looked-for.
                If None, it will assume the by-default value Conf.FARGO3D_BASEDIR
                (see constants at the beginning of this file)

                FARGO3D installation will be:
                    <basedir>/public/...

            react: string, default = 'get compile parallel gpu'
                How to react upon a finding. For instance, if FARGO3D installation
                is not found in basedir and you set react = 'get', this method
                will attempt to get a copy of FARGO3D. Otherwise you may get the 
                copy by yourself.

                The same for 'compile', that after testing that FARGO3D directory
                is present will attempto to compile a basic version of the package.

                'parallel' and 'gpu' will attempt to compile parallel and GPU 
                versions of `fargo3d`.
        Return:
            If everything is correct and the react is configured the method will
            end with the package downloaded to basedir, and three basic binaries
            `.fargo3d_`, `.fargo3d_PARALLEL-1` and `.fargo3d_GPU-1` in the installation
            directory.

            The configuration variables FARGO3D_BASEDIR, FARGO3D_PACKDIR, FARGO3D_FULLDIR
            will also be set with this routine.

            Last but not least the following configuration variables will be also set:
                FARGO3D_IS_HERE: indicating the package is present.
                FARGO3D_IS_COMPILING: indicating the package is properly compiling.
                FARGO3D_PARALLEL: 0/1 depending if parallel capabilities are available.
                FARGO3D_GPU: 0/1 depending if GPU capabilities are available.
        """
        no_basedir = False
        if not basedir:
            no_basedir = True
            basedir = Conf.FARGO3D_BASEDIR
        else:
            if not os.path.isdir(basedir):
                raise AssertionError(f"Provided basedir '{basedir}' not even exist")
            else:
                basedir = basedir[0]+basedir[1:].strip('/')+'/'

        # Check if FARGO3D has been downloaded 
        print("> Checking for FARGO3D directroy:")
        if not Conf._check_fargo(basedir+Conf.FARGO3D_PACKDIR):
            if 'get' in react:
                print("\tGetting FARGO3D public repo...")
                Conf._get_fargo(basedir)
                Conf.FARGO3D_IS_HERE = True
            else:
                options = "" if no_basedir else f"basedir='{basedir}',"
                print(f"\tDownload it with `set_fargo3d({options}react='get')` or set variable Conf.configure_fargo3d('{basedir}','public').""")
                return
        else:
            print(f"\t✓FARGO3D source code is available in your system at '{basedir+Conf.FARGO3D_PACKDIR}'")
            Conf.FARGO3D_IS_HERE = True
            Conf.configure_fargo3d(basedir,Conf.FARGO3D_PACKDIR)

        # Check if FARGO3D can be compiled normally
        print("> Checking for FARGO3D normal binary:")
        options = ''
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'compile' in react:
                print("\tCompiling FARGO3D (it may take a while)...")
                if Conf._compile_fargo3d(options,quiet=True):
                    Conf.FARGO3D_IS_COMPILING = True
                    print(f"\t✓Binary in normal mode compiling correctly")
        else:
            Conf.FARGO3D_IS_COMPILING = True
            print(f"\t✓Binary in normal mode compiling correctly")

        # Check if FARGO3D can be compiled in parallel
        print("> Checking for FARGO3D parallel binary:")
        options='PARALLEL=1'
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'parallel' in react:
                #print("\tCompiling FARGO3D in parallel (it may take a while)...")
                if Conf._compile_fargo3d(options,quiet=True):
                    Conf.FARGO3D_PARALLEL = 1
                    Conf.FARGO3D_IS_COMPILING_PARALLEL = True
                    print(f"\t✓Binary in parallel mode compiling correctly")
        else:
            Conf.FARGO3D_PARALLEL = 1
            Conf.FARGO3D_IS_COMPILING_PARALLEL = True
            print(f"\t✓Binary in parallel mode compiling correctly")

        # Check if FARGO3D can be compiled in parallel
        print("> Checking for FARGO3D GPU binary:")
        options='GPU=1'
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'gpu' in react:
                print("\tCompiling FARGO3D with GPU (it may take a while)...")
                if Conf._compile_fargo3d(options,quiet=True):
                    Conf.FARGO3D_GPU = 1
                    Conf.FARGO3D_IS_COMPILING_GPU = True
                    print(f"\t✓Binary in GPU mode compiling correctly")
                else:
                    print(f"\tNo GPU available")

        else:
            Conf.FARGO3D_GPU = 1
            Conf.FARGO3D_IS_COMPILING_GPU = True
            print(f"\t✓Binary in GPU mode compiling correctly")

    @staticmethod
    def _check_fargo(fulldir):
        if not os.path.isfile(fulldir+Conf.FARGO3D_HEADER):
            print(f"FARGO3D source code is not available at '{fulldir}'")
            return False
        
        return True
    
    @staticmethod
    def _is_fargo_here():
        if not Conf.FARGO3D_IS_HERE:
            raise AssertionError(f"FARGO3D source code has not been checked. Run Conf.set_fargo3d().")
        return True
    
    @staticmethod
    def _check_fargo_binary(fulldir,options='',quiet=False):
        if not os.path.isfile(fulldir+"."+Conf.FARGO3D_BINARY+f"_{Conf._binary_name(options)}"):
            if not quiet:
                print(f"FARGO3D binary with options '{options}' not compiled at '{fulldir}'")
            return False
        return True
    
    @staticmethod
    def _get_fargo(basedir):
        error,out = Util.sysrun(f'cd {basedir};{Conf.FARGO3D_CLONE_REPO_CMD}',verbose=False)
        if Util.QERROR <= 0:
            Conf.configure_fargo3d(basedir,Conf.FARGO3D_PACKDIR)
            print(f"\t✓Package downloaded to '{Conf.FARGO3D_FULLDIR}'")
            FARGO3D_IS_HERE = True
        else:
            print("\tError downloading. Check Util.STDOUT and Util.STDERR for verifying.")

    @staticmethod
    def _compile_fargo3d(options,quiet=False,force=False):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):

            # Binary filename
            binary_file = f".{Conf.FARGO3D_BINARY}_{Conf._binary_name(options)}"

            # Clean directory
            Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)

            # Check if binary already exist
            lets_compile = True
            if not force:
                if os.path.isfile(f"{Conf.FARGO3D_FULLDIR}/{binary_file}"):
                    print(f"Binary {Conf.FARGO3D_FULLDIR}/{binary_file} already compiled.")
                    lets_compile = False

            if lets_compile:

                if not quiet:
                    print(f"Compiling FARGO3D with options '{options}' (it may take a while... go for a coffee)")

                # Compile with options
                error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} {options}',verbose=False)
                
                # Check result of compilation
                if error:
                    if not os.path.isfile(Conf.FARGO3D_FULLDIR+Conf.FARGO3D_BINARY):
                        if not quiet:
                            print("An error compiling the code arose. Check dependencies.")
                            print(Util.STDERR)
                        return None
                    
                # Create a copy of binary compiled with options
                Util.sysrun(f"cp -rf {Conf.FARGO3D_FULLDIR}/{Conf.FARGO3D_BINARY} {Conf.FARGO3D_FULLDIR}/{binary_file}")

            # Clean all results
            Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)
            return binary_file

    @staticmethod
    def _run_fargo3d(fargo3d_binary,
                     logfile='/tmp/fargo.log',
                     options='',
                     mode='async',
                     resume=False,
                     verbose=True):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):

            # Run command
            run_cmd = f"cd {Conf.FARGO3D_FULLDIR};nohup {fargo3d_binary} {options} &> {logfile} & echo $!"
            
            # Let's run
            if mode == 'sync':
                # Run it synchronously, showing the output while running
                run_cmd = f"cd {Conf.FARGO3D_FULLDIR};{fargo3d_binary} {options} |tee {logfile}"
                print(f"Running synchronously: {run_cmd.replace('//','/')}")
                Util.sysrun(run_cmd,verbose=verbose)

            elif mode == 'async':
                # Run asynchronously in the background
                run_cmd = f"{fargo3d_binary} {options}"
                print(f"Running asynchronously: {run_cmd.replace('//','/')}")
                logmode = 'a' if resume else 'w'
                logfile_handler=open(logfile,logmode)
                process = subprocess.Popen(run_cmd.split(),cwd=Conf.FARGO3D_FULLDIR,
                                           stdout=logfile_handler,stderr=logfile_handler)
                print(f"Command is running in background")
                return process

    @staticmethod
    def _binary_name(options):
        return options.replace(' ','_').replace('=','-')
    
    @staticmethod
    def configure_fargo3d(basedir='.',packdir='public',parallel=None,gpu=None):
        
        # Configure directories
        Conf.FARGO3D_BASEDIR = (basedir + "/").replace('//','/')
        Conf.FARGO3D_PACKDIR = (packdir + "/").replace('//','/')
        Conf.FARGO3D_FULLDIR = (basedir + packdir).replace('//','/')

        # Configure compiling options
        if parallel is not None:
            Conf.parallel = parallel
        if gpu is not None:
            Conf.gpu = gpu

    @staticmethod
    def show_fargo3d_configuration():
        print("Is FARGO3D installed: ",Conf.FARGO3D_IS_HERE)
        print("Is FARGO3D compiling: ",Conf.FARGO3D_IS_COMPILING)
        print("Is FARGO3D compiling in parallel: ",Conf.FARGO3D_IS_COMPILING_PARALLEL)
        print("Is FARGO3D compiling in GPU: ",Conf.FARGO3D_IS_COMPILING_GPU)
        print("FARGO3D clone repositoty command: ",Conf.FARGO3D_CLONE_REPO_CMD)
        print("FARGO3D directories: ")
        print("\tBase directory: ",Conf.FARGO3D_BASEDIR)
        print("\tPackage directory: ",Conf.FARGO3D_PACKDIR)
        print("\tBasic package header: ",Conf.FARGO3D_HEADER)
        print("\tSetups location: ",Conf.FARGO3D_SETUPS)
        print("\tSetups location: ",Conf.FARGO3D_FULLDIR)
        print("Compile in parallel: ",Conf.FARGO3D_PARALLEL)
        print("Compile in GPU: ",Conf.FARGO3D_GPU)
    
    @staticmethod
    def _check_setup(setup):
        setup_dir = f"{Conf.FARGO3D_SETUPS}/{setup}"
        if not os.path.isdir(f"{Conf.FARGO3D_FULLDIR}/{setup_dir}"):
            print(f"No setup dir '{setup_dir}'")
            return None
        
        parameters_file = f"{setup_dir}/{setup}.par"
        if not os.path.isfile(f"{Conf.FARGO3D_FULLDIR}/{parameters_file}"):
            print(f"No parameter files '{setup}.par' in '{setup_dir}'")
            return None
        
        return setup_dir,parameters_file

class Dictobj(object):
    """Convert a dictionary to an object

    Examples:
        ob = Dictobj(a=2,b=3)
        print(ob.a,ob.b)
        ob = Dictobj(dict=dict(a=2,b=3))
        print(ob.a,ob.b)
        ob = Dictobj(dict={'a':2,'b':3})
        print(ob.a,ob.b)
    """

    def __init__(self, **kwargs):
        if 'dict' in kwargs.keys():
            kwargs.update(kwargs['dict'])
        for key, value in kwargs.items():
            if key == 'dict':continue
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

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
class Simulation(Conf):
    """Simulation class.
    """
    def __init__(self,**kwargs):
        super().__init__()

        if 'setup' in kwargs.keys():
            self.load_setup(setup=kwargs['setup'])
        else:
            self.setup = None

        # Compilation options
        self.parallel = Conf.FARGO3D_PARALLEL
        self.gpu = Conf.FARGO3D_GPU

        # Other options
        self.compile_options = None
        self.fargo3d_binary = None
        self.fargo3d_process = None

    def load_setup(self,setup=None):
        if not setup:
            setups = self.list_setups()
            print(f"You must select a setup, from a custom directory or from the list:{setups}")
        
        output = Conf._check_setup(setup)
        if output:
            self.setup = setup
            self.setup_directory = output[0]
            self.setup_parameters = output[1]
            self.setup_outputs = f"{self.setup_directory}/../../outputs/{self.setup}"
            self.logfile = f"{Conf.FARGO3D_FULLDIR}/{self.setup_directory}/{self.setup}.log"

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
        self.fargo3d_binary = Conf._compile_fargo3d(options=self.compile_options,force=force)
        
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
        
        self.logfile = f"{Conf.FARGO3D_FULLDIR}/{self.setup_directory}/{self.setup}.log"

        # Select command to run
        precmd=''
        if Conf.FARGO3D_PARALLEL:
            precmd = f"mpirun {mpioptions} "

        self.fargo3d_process = Conf._run_fargo3d(f"{precmd}./{self.fargo3d_binary} {options}",
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
        outputs_directory = f"{Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/"
        if not os.path.isdir(outputs_directory):
            print("There is no output to resume at '{outputs_directory.replace('//','/')}'.")
            return False
        return True
    
    def _get_snapshots(self):
        if not self._is_setup():
            return
        if not self._is_resumable():
            return
        
        error,output = Util.sysrun(f"grep OUTPUT {self.logfile}",verbose=False)
        if not error:
            snapshots = output[:-1]
        else:
            snapshots = []
        return snapshots
    
    def get_resumable_snapshot(self):
        error,output = Util.sysrun(f"grep OUTPUT {self.logfile}",verbose=False)
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
            error,output = Util.sysrun(f"ls {Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*.dat",verbose=False)
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
            error,output = Util.sysrun(f"grep OUTPUT {self.logfile} |tail -n 1",verbose=False)
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
        run_cmd = f"rm -r {Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*"
        error,output = Util.sysrun(run_cmd,verbose=False)
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
        output_dir = Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
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
        output_dir = Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
        variables = np.genfromtxt(output_dir+"/variables.par",
                                    dtype={'names': ("parameters","values"),
                                            'formats': ("|S30","|S300")}).tolist()
        print(f"Loading variables")
        self.vars = dict()
        for posicion in variables:
            self.vars[posicion[0].decode("utf-8")] = posicion[1].decode("utf-8")
        self.vars = Dictobj(dict=self.vars)
        print(f"{len(self.vars.__dict__.keys())} variables load. See <sim>.vars")

    def load_field(self,fluid='gas',field='dens',snapshot=None):
        if 'vars' not in self.__dict__.keys():
            self.load_variables()
        if not self._is_resumable():
            print("Simulation has not produced any output yet.")
        if snapshot is None:
            snapshot = self._get_resumable_snapshot()

        output_dir = Conf.FARGO3D_FULLDIR + "/" + self.setup_outputs
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

###############################################################
# Aliases
###############################################################

###############################################################
# External modules
###############################################################
import numpy as np
import warnings

###############################################################
# Constants
###############################################################
# Numerical Constants
RAD = 180/np.pi
DEG = 1/RAD

###############################################################
# Initialization
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Showing version 
print(f"Running FARGOpy version {version}")
