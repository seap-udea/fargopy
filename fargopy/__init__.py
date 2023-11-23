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

FP_HOME = os.environ['HOME']
FP_DOTDIR = f"{FP_HOME}/.fargopy" 
FP_RCFILE = f"{FP_DOTDIR}/fargopyrc"

# Default configuration
FP_CONFIGURATION = f"""# This is the configuration variables for FARGOpy
# Package
FP_VERSION = '{version}'
# System
FP_HOME = '{FP_HOME}'
# Directories
FP_DOTDIR = '{FP_DOTDIR}'
FP_RCFILE = '{FP_RCFILE}'
# Behavior
FP_VERBOSE = False
# FARGO3D variablles
FP_FARGO3D_CLONECMD = 'git clone https://bitbucket.org/fargo3d/public.git'
FP_FARGO3D_BASEDIR = './'
FP_FARGO3D_PACKDIR = 'fargo3d/'
FP_FARGO3D_BINARY = 'fargo3d'
FP_FARGO3D_HEADER = 'src/fargo3d.h'
"""

FP_INITIAL_SCRIPT = """
import sys
import fargopy as fp
fp.initialize(' '.join(sys.argv))
"""

###############################################################
# Base routines
###############################################################
class Debug(object):
    VERBOSE = False
    @staticmethod
    def trace(msg):
        if Debug.VERBOSE:
            print("::"+msg)

def initialize(options='', force=False):
    if ('configure' in options) or ('all' in options):
        # Create configuration directory
        if not os.path.isdir(FP_DOTDIR) or force:
            Debug.trace(f"Configuring FARGOpy...")
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

    if ('download' in options) or ('all' in options):
        print("Downloading FARGOpy...")
        fargo_dir = f"{FP_FARGO3D_BASEDIR}/{FP_FARGO3D_PACKDIR}".replace('//','/')
        if not os.path.isdir(fargo_dir):
            fargopy.Sys.simple(f"{FP_FARGO3D_CLONECMD} {FP_FARGO3D_PACKDIR}")
            print(f"\tFARGO3D downloaded to {fargo_dir}")
        else:
            print(f"\tFARGO3D directory already present in '{fargo_dir}'")
        
        fargo_header = f"{fargo_dir}/{FP_FARGO3D_HEADER}"
        if not os.path.isfile(fargo_header):
            print(f"No header file for fargo found in '{fargo_header}'")
        
    if ('compile' in options) or ('all' in options):
        print("Test compilation")
        pass

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

    def keys(self):
        props = []
        for i,prop in enumerate(self.__dict__.keys()):
            if '__' in prop:
                continue
            props += [prop]
        return props

    def print_keys(self):
        prop_list=''
        for i,prop in enumerate(self.keys()):
            prop_list += f"{prop}, "
            if ((i+1)%10) == 0:
                prop_list += '\n'
        print(prop_list.strip(', '))

    def item(self,key):
        if key not in self.keys():
            raise ValueError(f"Key 'key' not in Dictobj")
        return self.__dict__[key]

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

###############################################################
# Initialization
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Read FARGOpy configuration variables
if not os.path.isdir(FP_DOTDIR):
    Debug.trace(f"Configuring for the first time")
    initialize('configure')
Debug.trace(f"::Reading configuration variables")
exec(open(f"{FP_RCFILE}").read())
Debug.VERBOSE = FP_VERBOSE

# Check if version in RCFILE is different from installed FARGOpy version
if FP_VERSION != version:
    print(f"Your configure file version '{FP_VERSION}' it is different than the installed version of FARGOpy '{version}'")
    ans = input(f"Do you want to update configuration file '{FP_RCFILE}'? [Y/n]: ")
    if ans and ('Y' not in ans.upper()):
        if 'N' in ans.upper():
            print("We will keeping asking you until you update it, sorry!")
    else:
        os.system(f"cp -rf {FP_RCFILE} {FP_RCFILE}.save")
        initialize('configure',force=True)

# Showing version 
print(f"Running FARGOpy version {version}")
