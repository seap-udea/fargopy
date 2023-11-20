###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

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
                    print(f"\t✓Binary in parallel mode compiling correctly")
        else:
            Conf.FARGO3D_PARALLEL = 1
            print(f"\t✓Binary in parallel mode compiling correctly")

        # Check if FARGO3D can be compiled in parallel
        print("> Checking for FARGO3D GPU binary:")
        options='GPU=1'
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'gpu' in react:
                print("\tCompiling FARGO3D with GPU (it may take a while)...")
                if Conf._compile_fargo3d(options,quiet=True):
                    Conf.FARGO3D_GPU = 1
                    print(f"\t✓Binary in GPU mode compiling correctly")
                else:
                    print(f"\tNo GPU available")

        else:
            Conf.FARGO3D_GPU = 1
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
