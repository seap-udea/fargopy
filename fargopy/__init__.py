###############################################################
# Import montu modules
###############################################################
from fargopy.version import *
from fargopy.util import *

###############################################################
# Aliases
###############################################################

###############################################################
# External modules
###############################################################
import os
import sys
import subprocess
 
import numpy as np
import warnings

###############################################################
# Constants
###############################################################
# Numerical Constants
RAD = 180/np.pi
DEG = 1/RAD

###############################################################
# Classes
###############################################################
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

        if Conf.VERBOSE:
            error = out[-1][0]
            if Util.QERROR>0:
                print(f"::FARGOpy::sysrun::Error check Util.STDERR.")
            elif Util.QERROR<0:
                print(f"::FARGOpy::sysrun::Done. Still some issues must be check. Check Util.STDOUT and Util.STDERR for details.")
            elif Util.QERROR==0:
                print(f"::FARGOpy::sysrun::Done. You're great. Check Util.STDOUT for details.")
        
        return Util.QERROR,out

class Conf(object):
    """Configuration class.
    """
    VERBOSE = False
    FARGO3D_CLONE_REPO_CMD = 'git clone https://bitbucket.org/fargo3d/public.git'
    FARGO3D_BASEDIR = './'
    FARGO3D_PACKDIR = 'public/'
    FARGO3D_FULLDIR = './public/'
    FARGO3D_HEADER = 'src/fargo3d.h'
    FARGO3D_BINARY = 'fargo3d'
    SYS_STDOUT = ''
    SYS_STDERR = ''

    def __init__(self):
        # Common attributes
        self.parallel = 0
        self.gpu = 0

    @staticmethod
    def set_fargo3d(basedir=None,react='get compile parallel gpu'):
        """Check if FARGO3D is properly installed in the sysem
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
            else:
                options = "" if no_basedir else f"basedir='{basedir}',"
                print(f"\tDownload it with `set_fargo3d({options}react='get')` or set variable Conf.update_fargo3d_dir('{basedir}','public').""")
                return
        else:
            print(f"\t✓FARGO3D source code is available in your system at '{basedir+Conf.FARGO3D_PACKDIR}'")
            Conf.update_fargo3d_dir(basedir,Conf.FARGO3D_PACKDIR)

        # Check if FARGO3D can be compiled normally
        print("> Checking for FARGO3D normal binary:")
        options = ''
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'compile' in react:
                print("\tCompiling FARGO3D (it may take a while)...")
                if Conf._test_compile_fargo3d(options):
                    print(f"\t✓Binary in normal mode compiling correctly")
        else:
            print(f"\t✓Binary in normal mode compiling correctly")

        # Check if FARGO3D can be compiled in parallel
        print("> Checking for FARGO3D parallel binary:")
        options='PARALLEL=1'
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'parallel' in react:
                print("\tCompiling FARGO3D in parallel (it may take a while)...")
                if Conf._test_compile_fargo3d(options):
                    print(f"\t✓Binary in parallel mode compiling correctly")
        else:
            print(f"\t✓Binary in parallel mode compiling correctly")

        # Check if FARGO3D can be compiled in parallel
        print("> Checking for FARGO3D GPU binary:")
        options='GPU=1'
        if not Conf._check_fargo_binary(basedir+Conf.FARGO3D_PACKDIR,options):
            if 'gpu' in react:
                print("\tCompiling FARGO3D with GPU (it may take a while)...")
                if Conf._test_compile_fargo3d(options,quiet=True):
                    print(f"\t✓Binary in GPU mode compiling correctly")
                else:
                    print(f"\tNo GPU available")

        else:
            print(f"\t✓Binary in GPU mode compiling correctly")

    @staticmethod
    def _check_fargo(fulldir):
        if not os.path.isfile(fulldir+Conf.FARGO3D_HEADER):
            print(f"FARGO3D source code is not available at '{fulldir}'")
            return False
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
            Conf.update_fargo3d_dir(basedir,Conf.FARGO3D_PACKDIR)
            print(f"\t✓Package downloaded to '{Conf.FARGO3D_FULLDIR}'")
        else:
            print("\tError downloading. Check Util.STDOUT and Util.STDERR for verifying.")

    @staticmethod
    def clean_fargo3d():
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):
            error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)

    @staticmethod
    def _test_compile_fargo3d(options,quiet=False):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):
            # Clean directory
            Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)

            # Compile with options
            error,out = Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} {options}',verbose=False)
            
            # Check result of compilation
            if error:
                if not os.path.isfile(Conf.FARGO3D_FULLDIR+Conf.FARGO3D_BINARY):
                    if not quiet:
                        print("An error compiling the code arose. Check dependencies.")
                        print(Util.STDERR)
                    return False
                
            # Create a copy of binary compiled with options
            Util.sysrun(f"cp -rf {Conf.FARGO3D_FULLDIR}/{Conf.FARGO3D_BINARY} {Conf.FARGO3D_FULLDIR}/.{Conf.FARGO3D_BINARY}_{Conf._binary_name(options)}")

            # Clean result
            Util.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper',verbose=False)

            return True
        
    @staticmethod
    def _binary_name(options):
        return options.replace(' ','_').replace('=','-')

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

    @staticmethod
    def update_fargo3d_dir(basedir,packdir):
        Conf.FARGO3D_BASEDIR = basedir
        Conf.FARGO3D_PACKDIR = packdir
        Conf.FARGO3D_FULLDIR = basedir + packdir

class Simulation(Conf):
    """Simulation class.
    """
    def __init__(self):
        super().__init__()

    """
    def set_fargo3d(self):
        Conf.set_fargo3d()
    """

###############################################################
# Initialization
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Showing version 
print(f"Running FARGOpy version {version}")
