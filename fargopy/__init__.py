"""FARGOpy package.

To run it with IPython run:

$ ifargopy

To install FARGO3D run:

$ ifargopy download

To import use:

import fargopy as fp
"""

###############################################################
# Import montu modules
###############################################################
from fargopy.version import *
from fargopy.util import *
from fargopy.sys import *
from fargopy.fargo3d import *

###############################################################
# External modules
###############################################################
import warnings
import os
import numpy as np

###############################################################
# Aliases
###############################################################

###############################################################
# Constants
###############################################################
DEG = np.pi/180
RAD = 1/DEG

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

###############################################################
# Package configuration
###############################################################
# Basic (unmodifiable) variables
FP_HOME = os.environ['HOME']
FP_DOTDIR = f"{FP_HOME}/.fargopy" 
FP_RCFILE = f"{FP_DOTDIR}/fargopyrc"

# Default configuration file content
FP_CONFIGURATION = f"""# This is the configuration variables for FARGOpy
# Package
FP_VERSION = '{version}'
# System
FP_HOME = '{FP_HOME}/'
# Directories
FP_DOTDIR = '{FP_DOTDIR}'
FP_RCFILE = '{FP_RCFILE}'
# Behavior
FP_VERBOSE = False
# FARGO3D variablles
FP_FARGO3D_CLONECMD = 'git clone https://bitbucket.org/fargo3d/public.git'
FP_FARGO3D_BASEDIR = '{FP_HOME}'
FP_FARGO3D_PACKDIR = 'fargo3d/'
FP_FARGO3D_BINARY = 'fargo3d'
FP_FARGO3D_HEADER = 'src/fargo3d.h'
"""

# Default initialization script
FP_INITIAL_SCRIPT = """
import sys
import fargopy as fp
fp.initialize(' '.join(sys.argv))
"""

def initialize(options='', force=False):
    """Initialization routine

    Args:
        options: string, default = '':
            Action(s) to be performed. Valid actions include:
                'configure': configure the package.
                'download': download FARGO3D directory.
                'compile': attempt to compile FARGO3D in the machine.
                'all': all actions.

        force: bool, default = False:
            If True, force any action that depends on a previous condition.
            For instance if options = 'configure' and force = True it will
            override FARGOpy directory.
    """
    if ('configure' in options) or ('all' in options):
        # Create configuration directory
        if not os.path.isdir(FP_DOTDIR) or force:
            Debug.trace(f"Configuring FARGOpy at {FP_DOTDIR}...")
            # Create directory
            os.system(f"mkdir -p {FP_DOTDIR}")
            # Create configuration variables
            f = open(f"{FP_DOTDIR}/fargopyrc",'w')
            f.write(FP_CONFIGURATION)
            f.close()
            # Create initialization script
            f = open(f"{FP_DOTDIR}/ifargopy.py",'w')
            f.write(FP_INITIAL_SCRIPT)
            f.close()
        else:
            Debug.trace(f"Configuration already in place.")

    if ('download' in options) or ('all' in options):
        print("Downloading FARGOpy...")
        fargo_dir = f"{FP_FARGO3D_BASEDIR}/{FP_FARGO3D_PACKDIR}".replace('//','/')
        if not os.path.isdir(fargo_dir) or force:
            fargopy.Sys.simple(f"cd {FP_FARGO3D_BASEDIR};{FP_FARGO3D_CLONECMD} {FP_FARGO3D_PACKDIR}")
            print(f"\tFARGO3D downloaded to {fargo_dir}")
        else:
            print(f"\tFARGO3D directory already present in '{fargo_dir}'")
        
        fargo_header = f"{fargo_dir}/{FP_FARGO3D_HEADER}"
        if not os.path.isfile(fargo_header):
            print(f"No header file for fargo found in '{fargo_header}'")
        else:
            print(f"Header file for FARGO3D is in the fargo directory {fargo_dir}")
        
    if ('compile' in options) or ('all' in options):
        print("Test compilation")
        pass

###############################################################
# Initialization
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Read FARGOpy configuration variables
if not os.path.isdir(FP_DOTDIR):
    print(f"Configuring FARGOpy for the first time")
    initialize('configure')
Debug.trace(f"::Reading configuration variables")
exec(open(f"{FP_RCFILE}").read())
Debug.VERBOSE = FP_VERBOSE
FP_FARGO3D_DIR = (FP_FARGO3D_BASEDIR + '/' + FP_FARGO3D_PACKDIR).replace('//','/')

# Check if version in RCFILE is different from installed FARGOpy version
if FP_VERSION != version:
    print(f"Your configuration file version '{FP_VERSION}' it is different than the installed version of FARGOpy '{version}'")
    ans = input(f"Do you want to update configuration file '{FP_RCFILE}'? [Y/n]: ")
    if ans and ('Y' not in ans.upper()):
        if 'N' in ans.upper():
            print("We will keeping asking you this until you update it, sorry!")
    else:
        os.system(f"cp -rf {FP_RCFILE} {FP_RCFILE}.save")
        initialize('configure',force=True)

# Showing version 
print(f"Running FARGOpy version {version}")
