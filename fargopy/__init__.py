###############################################################
# Version
###############################################################
from fargopy.version import *

###############################################################
# External modules
###############################################################
import warnings
import os
import json
import sys
import numpy as np

###############################################################
# Aliases
###############################################################

###############################################################
# Constants
###############################################################
DEG = np.pi/180
RAD = 1/DEG

# Check if we are in colab
IN_COLAB = 'google.colab' in sys.modules

###############################################################
# Base classes
###############################################################
class Debug(object):
    """The Debug class control the Debugging messages of the package.

    Attribute:
        VERBOSE: bool, default = False:
            If True all the trace messages are shown.

    Static methods:
        trace(msg):
            Show a debugging message if VERBOSE=True
            Example:
            >>> import fargopy as fp
            >>> fp.Debug.VERBOSE = True
            >>> fp.initialize('configure') 
    """
    VERBOSE = False
    @staticmethod
    def trace(msg):
        if Debug.VERBOSE:
            print("::"+msg)

class Dictobj(object):
    """Convert a dictionary to an object

    Initialization attributes:
        dict: dictionary:
            Dictionary containing the attributes.

    Attributes:
        All the keys in the initialization dictionary.

    Examples:
        Three ways to initialize the same `Dictobj`:
        >>> ob = Dictobj(a=2,b=3)
        >>> ob = Dictobj(dict=dict(a=2,b=3))
        >>> ob = Dictobj(dict={'a':2,'b':3})

        In the three cases you may access the attributes using:
        >>> print(ob.a,ob.b)

    Methods:
        keys():
            It works like the keys() method of a dictionary.
        item(key):
            Recover the value of an attribute as it was a dictionary.
            Example:
                >>> ob.item('a') 
        print_keys():
            Print a list of keys
        
    """

    def __init__(self, **kwargs):
        if 'dict' in kwargs.keys():
            kwargs.update(kwargs['dict'])
        for key, value in kwargs.items():
            if key == 'dict':continue
            setattr(self, key, value)

    def __getitem__(self,key):
        return self.item(str(key))

    def keys(self):
        """Show the list of attributes of Dictobj

        This method works as the keys() method of a regular dictionary.
        """
        props = []
        for i,prop in enumerate(self.__dict__.keys()):
            if '__' in prop:
                continue
            props += [prop]
        return props

    def item(self,key):
        """Get the value of an item of a Dictobj.
        """
        if key not in self.keys():
            raise ValueError(f"Key 'key' not in Dictobj")
        return self.__dict__[key]
    
    def __getitem__(self,key):
        return self.item(str(key))
        
    def print_keys(self):
        """Print all the keys of a Dictobj.
        """
        prop_list=''
        for i,prop in enumerate(self.keys()):
            prop_list += f"{prop}, "
            if ((i+1)%10) == 0:
                prop_list += '\n'
        print(prop_list.strip(', '))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

class Fargobj(object):
    def __init__(self,**kwargs):
        self.fobject = True
        self.kwargs = kwargs

    def save_object(self,filename=None,verbose=False):
        """Save Fargobj into a filename in JSON format
        
        Args:
            filename: string, default = None:
                Path of the file where the object will be saved.
                If None the filename will be '/tmp/fargobj_{hash}.json' 
                where {hash} is the hash of the object attributes dictionary.
        """
        if filename is None:
            object_hash = str(abs(hash(str(self.__dict__))))
            filename = f"/tmp/fargobj_{object_hash}.json"
        if verbose:
            print(f"Saving object to {filename}...")
        with open(filename,'w') as file_object:
            file_object.write(json.dumps(self.__dict__,default=lambda obj:'<not serializable>'))
            file_object.close()
        
    def set_property(self,property,default,method=lambda prop:prop):
        """Set a property of object using a given method

        Examples:
            Simple example
            >>> obj = Fargobj()
            >>> obj.set_property('a',1)
            >>> print(obj.a)
            1

            Using a special method:
            >>> obj = Fargobj()
            >>> obj.set_property('a',2,lambda x:x**2)
            >>> print(obj.a)
            4
        """
        if property in self.kwargs.keys():
            method(self.kwargs[property])
            self.__dict__[property] = self.kwargs[property]
            return True
        else:
            method(default)
            self.__dict__[property] = default
            return False
        
    def has(self,key):
        """Check if a key is an attribute of Fargobj object

        Examples:
            >>> obj = Fargobj(a=1)
            >>> print(obj.has('a'))
            True
        """
        if key in self.__dict__.keys():
            return True
        else:
            return False

###############################################################
# Package configuration
###############################################################
# Basic (unmodifiable) variables
Conf = Dictobj()
Conf.FP_HOME = os.environ['HOME']
Conf.FP_DOTDIR = f"{Conf.FP_HOME}/.fargopy" 
Conf.FP_RCFILE = f"{Conf.FP_DOTDIR}/fargopyrc"

# Default configuration file content
Conf.FP_CONFIGURATION = f"""# This is the configuration variables for FARGOpy
# Package
FP_VERSION = '{version}'
# System
FP_HOME = '{Conf.FP_HOME}/'
# Directories
FP_DOTDIR = '{Conf.FP_DOTDIR}'
FP_RCFILE = '{Conf.FP_RCFILE}'
# Behavior
FP_VERBOSE = False
# FARGO3D variablles
FP_FARGO3D_CLONECMD = 'git clone https://bitbucket.org/fargo3d/public.git'
FP_FARGO3D_BASEDIR = '{Conf.FP_HOME}'
FP_FARGO3D_PACKDIR = 'fargo3d/'
FP_FARGO3D_BINARY = 'fargo3d'
FP_FARGO3D_HEADER = 'src/fargo3d.h'
"""

# Default initialization script
Conf.FP_INITIAL_SCRIPT = """
import sys
import fargopy as fp
get_ipython().run_line_magic('load_ext','autoreload')
get_ipython().run_line_magic('autoreload','2')
fp.initialize(' '.join(sys.argv))
"""

def initialize(options='', force=False, **kwargs):
    """Initialization routine

    Args:
        options: string, default = '':
            Action(s) to be performed. Valid actions include:
                'configure': configure the package.
                'download': download FARGO3D directory.
                'check': attempt to compile FARGO3D in the machine.
                'all': all actions.

        force: bool, default = False:
            If True, force any action that depends on a previous condition.
            For instance if options = 'configure' and force = True it will
            override FARGOpy directory.
    """
    if ('configure' in options) or ('all' in options):
        # Create configuration directory
        if not os.path.isdir(Conf.FP_DOTDIR) or force:
            Debug.trace(f"Configuring FARGOpy at {Conf.FP_DOTDIR}...")
            # Create directory
            os.system(f"mkdir -p {Conf.FP_DOTDIR}")
            # Create configuration variables
            f = open(f"{Conf.FP_DOTDIR}/fargopyrc",'w')
            f.write(Conf.FP_CONFIGURATION)
            f.close()
            # Create initialization script
            f = open(f"{Conf.FP_DOTDIR}/ifargopy.py",'w')
            f.write(Conf.FP_INITIAL_SCRIPT)
            f.close()
        else:
            Debug.trace(f"Configuration already in place.")

    if ('download' in options) or ('all' in options):
        fargo_dir = f"{Conf.FP_FARGO3D_BASEDIR}/{Conf.FP_FARGO3D_PACKDIR}".replace('//','/')
    
        print("Downloading FARGOpy...")
        if not os.path.isdir(fargo_dir) or force:
            if os.path.isdir(fargo_dir):
                print(f"Directory '{fargo_dir}' already exists. Removing it...")
                os.system(f"rm -rf {fargo_dir}")
            fargopy.Sys.simple(f"cd {Conf.FP_FARGO3D_BASEDIR};{Conf.FP_FARGO3D_CLONECMD} {Conf.FP_FARGO3D_PACKDIR}")
            print(f"\tFARGO3D downloaded to {fargo_dir}")
        else:
            print(f"\tFARGO3D directory already present in '{fargo_dir}'")
        
        fargo_header = f"{fargo_dir}/{Conf.FP_FARGO3D_HEADER}"
        if not os.path.isfile(fargo_header):
            print(f"No header file for fargo found in '{fargo_header}'")
        else:
            print(f"Header file for FARGO3D found in the fargo directory {fargo_dir}")
        
    if ('check' in options) or ('all' in options):
        fargo_dir = f"{Conf.FP_FARGO3D_BASEDIR}/{Conf.FP_FARGO3D_PACKDIR}".replace('//','/')
    
        print("Test compilation of FARGO3D")
        if not os.path.isdir(fargo_dir):
            print(f"Directory '{fargo_dir}' does not exist. Please download it with fargopy.initialize('download')")
          
        cmd_fun = lambda options,mode:f"make -C {fargo_dir} clean mrproper all {options} 2>&1 |tee /tmp/fargo_{mode}.log"
        
        for option,mode in zip(['PARALLEL=0 GPU=0','PARALLEL=0 GPU=1','PARALLEL=1 GPU=0'],
                                 ['regular','gpu','parallel']):
            # Verify if you want to check this mode
            if (mode in kwargs.keys()) and (kwargs[mode]==0):
                    print(f"\tSkipping {mode} compilation")
                    exec(f"Conf.FP_FARGO3D_{mode.upper()} = 0")
                    continue

            cmd = cmd_fun(option,mode)
            print(f"\tChecking normal compilation.\n\tRunning '{cmd}':")
            error,output = Sys.run(cmd)
            if not os.path.isfile(f"{fargo_dir}/{Conf.FP_FARGO3D_BINARY}"):
                print(f"\t\tCompilation failed for '{mode}'. Check log file '/tmp/fargo_{mode}.log'")
                exec(f"Conf.FP_FARGO3D_{mode.upper()} = 0")
            else:
                print(f"\t\tCompilation in mode {mode} successful.")
                exec(f"Conf.FP_FARGO3D_{mode.upper()} = 1")
        
        print(f"Summary of compilation modes:")
        print(f"\tRegular: {Conf.FP_FARGO3D_REGULAR}")
        print(f"\tGPU: {Conf.FP_FARGO3D_GPU}")
        print(f"\tParallel: {Conf.FP_FARGO3D_PARALLEL}")
        
###############################################################
# Initialization
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Read FARGOpy configuration variables
if not os.path.isdir(Conf.FP_DOTDIR):
    print(f"Configuring FARGOpy for the first time")
    initialize('configure')
Debug.trace(f"::Reading configuration variables")

# Load configuration variables into Conf
conf_dict = dict()
exec(open(f"{Conf.FP_RCFILE}").read(),dict(),conf_dict)
Conf.__dict__.update(conf_dict)

# Derivative configuration variables
Debug.VERBOSE = Conf.FP_VERBOSE
Conf.FP_FARGO3D_DIR = (Conf.FP_FARGO3D_BASEDIR + '/' + Conf.FP_FARGO3D_PACKDIR).replace('//','/')
Conf.FP_FARGO3D_LOCKFILE = f"{Conf.FP_DOTDIR}/fargopy.lock"

# Check if version in RCFILE is different from installed FARGOpy version
if Conf.FP_VERSION != version:
    print(f"Your configuration file version '{Conf.FP_VERSION}' it is different than the installed version of FARGOpy '{version}'")
    ans = input(f"Do you want to update configuration file '{Conf.FP_RCFILE}'? [Y/n]: ")
    if ans and ('Y' not in ans.upper()):
        if 'N' in ans.upper():
            print("We will keeping asking you this until you update it, sorry!")
    else:
        os.system(f"cp -rf {Conf.FP_RCFILE} {Conf.FP_RCFILE}.save")
        initialize('configure',force=True)

###############################################################
# Import package modules
###############################################################
from fargopy.util import *
from fargopy.sys import *
from fargopy.fields import *
from fargopy.simulation import *
from fargopy.plot import *
#from fargopy.Fsimulation

# Showing version 
print(f"Running FARGOpy version {version}")
