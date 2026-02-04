###############################################################
# Package Base
###############################################################
from .base import *

###############################################################
# Import package modules
###############################################################
# Now we can import submodules at the top level because base is loaded
from fargopy.sys import *
from fargopy.fields import *
from fargopy.simulation import *
from fargopy.plot import *
from fargopy.flux import *

# Show version
welcome()

# Clean up namespace if needed (optional)
del welcome
