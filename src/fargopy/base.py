###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Package documentation
###############################################################
"""
FARGOpy Base Module
===================

This module contains the base classes, configuration, and initialization logic for FARGOpy.
It is separated from __init__.py to avoid circular import issues with other modules like sys.py.
"""

import warnings
import os
import json
import pickle
import sys
import numpy as np
import inspect

# Version
__version__ = "1.1.1"

__all__ = [
    "__version__",
    "Debug",
    "Dictobj",
    "Fargobj",
    "FieldsHandler",
    "Conf",
    "initialize",
    "DEG",
    "RAD",
    "IN_COLAB",
    "_welcome",
]


def _docstring_summary(doc):
    """
    Extract the first line or paragraph of a docstring as summary.

    Stops at common section headers (Parameters, Returns, etc.)
    so that the description does not include parameter lists.

    Parameters
    ----------
    doc : str or None
        The docstring.

    Returns
    -------
    str
        One-line summary or empty string if no docstring.
    """
    if not doc or not doc.strip():
        return ""
    doc = doc.strip()
    section_markers = (
        "Parameters",
        "Returns",
        "Examples",
        "Attributes",
        "Notes",
        "Methods",
        "Raises",
        "See Also",
        "Warnings",
    )
    lines = doc.splitlines()
    summary_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if any(
            stripped == marker
            or stripped.startswith(marker + " ")
            or stripped == marker + ":"
            for marker in section_markers
        ):
            break
        if stripped.endswith("---") or stripped.endswith("===="):
            break
        summary_lines.append(stripped)
    return " ".join(summary_lines).strip() if summary_lines else ""


###############################################################
# Constants
###############################################################
DEG = np.pi / 180
RAD = 1 / DEG

# Check if we are in colab
IN_COLAB = "google.colab" in sys.modules


###############################################################
# Base classes
###############################################################
class Debug(object):
    """The Debug class controls the debugging messages of the package.

    Attributes
    ----------
    VERBOSE : bool
        If True all the trace messages are shown. Default is False.

    Methods
    -------
    trace(msg)
        Show a debugging message if VERBOSE=True.
    """

    VERBOSE = False

    @staticmethod
    def trace(msg):
        if Debug.VERBOSE:
            print("::" + msg)


class Dictobj(object):
    """Convert a dictionary to an object

    Initialization attributes:
        dict: dictionary:
            Dictionary containing the attributes.

    Attributes:
        All the keys in the initialization dictionary.

    Methods:
        keys():
            It works like the keys() method of a dictionary.
        item(key):
            Recover the value of an attribute as it was a dictionary.
        print_keys():
            Print a list of keys
    """

    def __init__(self, **kwargs):
        if "dict" in kwargs.keys():
            kwargs.update(kwargs["dict"])
        for key, value in kwargs.items():
            if key == "dict":
                continue
            setattr(self, key, value)

    def __getitem__(self, key):
        return self.item(str(key))

    def keys(self):
        """Show the list of attributes of Dictobj"""
        props = []
        for i, prop in enumerate(self.__dict__.keys()):
            if "__" in prop:
                continue
            props += [prop]
        return props

    def item(self, key):
        """Get the value of an item of a Dictobj."""
        if key not in self.keys():
            raise ValueError(f"Key 'key' not in Dictobj")
        return self.__dict__[key]

    def __getitem__(self, key):
        return self.item(str(key))

    def print_keys(self):
        """Print all the keys of a Dictobj."""
        prop_list = ""
        for i, prop in enumerate(self.keys()):
            prop_list += f"{prop}, "
            if ((i + 1) % 10) == 0:
                prop_list += "\n"
        print(prop_list.strip(", "))

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class Fargobj(object):
    def __init__(self, **kwargs):
        self.fobject = True
        self.kwargs = kwargs

    def save_object(self, filename=None, verbose=False):
        """Save Fargobj into a filename in JSON format"""
        if filename is None:
            object_hash = str(abs(hash(str(self.__dict__))))
            filename = f"/tmp/fargobj_{object_hash}.json"
        if verbose:
            print(f"Saving object to {filename}...")
        with open(filename, "w") as file_object:
            file_object.write(
                json.dumps(self.__dict__, default=lambda obj: "<not serializable>")
            )
            file_object.close()

    def save_object_pkl(self, filename=None, verbose=False):
        """Save Fargobj into a pickle file (.pkl)

        Parameters
        ----------
        filename : str, optional
            Path to save the pickle file. If None, creates a temporary file.
        verbose : bool, optional
            If True, print saving message. Default is False.

        Examples
        --------
        >>> obj.save_object_pkl('myobject.pkl')
        >>> sim.save_object_pkl('simulation.pkl', verbose=True)
        """
        if filename is None:
            object_hash = str(abs(hash(str(self.__dict__))))
            filename = f"/tmp/fargobj_{object_hash}.pkl"
        if verbose:
            print(f"Saving object to {filename}...")
        with open(filename, "wb") as file_object:
            pickle.dump(self, file_object)
        return filename

    @classmethod
    def read_object(cls, filename, verbose=False):
        """Read a Fargobj from a pickle (.pkl) or JSON file

        This method automatically detects the file format and loads accordingly.
        It first tries to load as pickle, and if that fails, tries JSON format.

        Parameters
        ----------
        filename : str
            Path to the file to load (.pkl or .json)
        verbose : bool, optional
            If True, print loading message. Default is False.

        Returns
        -------
        object
            The loaded Fargobj or its subclass instance

        Examples
        --------
        >>> obj = fargopy.Fargobj.read_object('myobject.pkl')
        >>> sim = fargopy.Simulation.read_object('simulation.pkl')

        Notes
        -----
        JSON files saved with save_object() can be loaded, but they will only
        restore the object's __dict__ attributes, not the full object state.
        For full object serialization, use save_object_pkl() and .pkl files.
        """
        if verbose:
            print(f"Loading object from {filename}...")

        # Try pickle format first
        try:
            with open(filename, "rb") as file_object:
                obj = pickle.load(file_object)
            return obj
        except (pickle.UnpicklingError, UnicodeDecodeError):
            # If pickle fails, try JSON format
            if verbose:
                print("Pickle format failed, trying JSON format...")
            try:
                with open(filename, "r") as file_object:
                    data = json.load(file_object)
                # Create instance without calling __init__ to avoid constructor issues
                obj = object.__new__(cls)
                # Restore all attributes from JSON data
                obj.__dict__.update(data)

                # Check if object has corrupted serialization markers
                for key, value in obj.__dict__.items():
                    if value == "<not serializable>":
                        print(
                            f"WARNING: Attribute '{key}' was not properly serialized."
                        )
                        print(
                            f"This object was saved with save_object() which uses JSON and cannot"
                        )
                        print(
                            f"serialize complex objects. Use save_object_pkl() instead for full serialization."
                        )

                return obj
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"File '{filename}' is neither a valid pickle nor JSON file: {e}"
                )

    def set_property(self, property, default, method=lambda prop: prop):
        """Set a property of object using a given method"""
        if property in self.kwargs.keys():
            method(self.kwargs[property])
            self.__dict__[property] = self.kwargs[property]
            return True
        else:
            method(default)
            self.__dict__[property] = default
            return False

    def has(self, key):
        """Check if a key is an attribute of Fargobj object"""
        if key in self.__dict__.keys():
            return True
        else:
            return False

    @classmethod
    def methods(cls):
        """
        Show the list of public methods for this instance's class with their
        short description taken from each method's docstring.

        Can be called on an instance (e.g. obj.describe()) or on the class
        (e.g. mn.DensityPlot.describe()). Intended for discovery of available
        functionality on any FargoPyBase subclass (e.g. DensityPlot, CMND).
        """
        methods = []
        for name in dir(cls):
            if name.startswith("_"):
                continue
            obj = getattr(cls, name)
            if not callable(obj):
                continue
            methods.append((name, obj))
        methods.sort(key=lambda x: x[0])
        lines = [
            f"\nAvailable methods for this object/class",
            "=" * (30 + len(cls.__name__)),
        ]
        for name, meth in methods:
            if name == "describe":
                continue
            doc = inspect.getdoc(meth)
            summary = _docstring_summary(doc) if doc else "(sin descripción)"
            summary = summary.replace("\n", " ").strip()
            # if len(summary) > 70:
            #     summary = summary[:67] + "..."
            lines.append(f"  {name}()")
            lines.append(f"    {summary}")
            lines.append("")
        print("\n".join(lines))


###############################################################
# Package configuration
###############################################################
# Basic (unmodifiable) variables
Conf = Dictobj()

# Cross-platform home directory detection
if os.name == "nt":  # Windows
    Conf.FP_HOME = os.environ.get("USERPROFILE", os.path.expanduser("~"))
else:  # Unix-like systems (Linux, macOS)
    Conf.FP_HOME = os.environ.get("HOME", os.path.expanduser("~"))

Conf.FP_DOTDIR = os.path.join(Conf.FP_HOME, ".fargopy")
Conf.FP_RCFILE = os.path.join(Conf.FP_DOTDIR, "fargopyrc")

# Default configuration file content
Conf.FP_CONFIGURATION = f"""# This is the configuration variables for FARGOpy
# Package
FP_VERSION = '{__version__}'
# System
FP_HOME = '{Conf.FP_HOME}/'
# Directories
FP_DOTDIR = '{Conf.FP_DOTDIR}'
FP_RCFILE = '{Conf.FP_RCFILE}'
# Behavior
FP_VERBOSE = False
# FARGO3D variables
FP_FARGO3D_CLONECMD = 'GIT_TERMINAL_PROMPT=0 git clone https://bitbucket.org/fargo3d/public.git'
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


def initialize(options="", force=False, **kwargs):
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

    # Import fargopy inside the function to avoid circular imports
    # when accessing fargopy.Sys or other components that might rely on base.py
    import fargopy

    if ("configure" in options) or ("all" in options):
        # Create configuration directory
        if not os.path.isdir(Conf.FP_DOTDIR) or force:
            Debug.trace(f"Configuring FARGOpy at {Conf.FP_DOTDIR}...")
            # Create directory
            os.system(f"mkdir -p {Conf.FP_DOTDIR}")
            # Create configuration variables
            f = open(f"{Conf.FP_DOTDIR}/fargopyrc", "w")
            f.write(Conf.FP_CONFIGURATION)
            f.close()
            # Create initialization script
            f = open(f"{Conf.FP_DOTDIR}/ifargopy.py", "w")
            f.write(Conf.FP_INITIAL_SCRIPT)
            f.close()
        else:
            Debug.trace(f"Configuration already in place.")

    if ("download" in options) or ("all" in options):
        fargo_dir = f"{Conf.FP_FARGO3D_BASEDIR}/{Conf.FP_FARGO3D_PACKDIR}".replace(
            "//", "/"
        )

        print("Downloading FARGOpy...")
        if not os.path.isdir(fargo_dir) or force:
            if os.path.isdir(fargo_dir):
                print(f"Directory '{fargo_dir}' already exists. Removing it...")
                os.system(f"rm -rf {fargo_dir}")
            # Ensure GIT_TERMINAL_PROMPT=0 is set to avoid stalling if authentication is requested
            clone_cmd = Conf.FP_FARGO3D_CLONECMD
            if "GIT_TERMINAL_PROMPT" not in clone_cmd:
                clone_cmd = "GIT_TERMINAL_PROMPT=0 " + clone_cmd

            fargopy.Sys.simple(
                f"cd {Conf.FP_FARGO3D_BASEDIR} && {clone_cmd} {Conf.FP_FARGO3D_PACKDIR}"
            )
            print(f"\tFARGO3D downloaded to {fargo_dir}")
        else:
            print(f"\tFARGO3D directory already present in '{fargo_dir}'")

        fargo_header = f"{fargo_dir}/{Conf.FP_FARGO3D_HEADER}"
        if not os.path.isfile(fargo_header):
            print(f"No header file for fargo found in '{fargo_header}'")
        else:
            print(f"Header file for FARGO3D found in the fargo directory {fargo_dir}")

    if ("check" in options) or ("all" in options):
        fargo_dir = f"{Conf.FP_FARGO3D_BASEDIR}/{Conf.FP_FARGO3D_PACKDIR}".replace(
            "//", "/"
        )

        print("Test compilation of FARGO3D")
        if not os.path.isdir(fargo_dir):
            print(
                f"Directory '{fargo_dir}' does not exist. Please download it with fargopy.initialize('download')"
            )

        cmd_fun = lambda options, mode: (
            f"make -C {fargo_dir} clean mrproper all {options} 2>&1 |tee /tmp/fargo_{mode}.log"
        )

        for option, mode in zip(
            ["PARALLEL=0 GPU=0", "PARALLEL=0 GPU=1", "PARALLEL=1 GPU=0"],
            ["regular", "gpu", "parallel"],
        ):
            # Verify if you want to check this mode
            if (mode in kwargs.keys()) and (kwargs[mode] == 0):
                print(f"\tSkipping {mode} compilation")
                exec(f"Conf.FP_FARGO3D_{mode.upper()} = 0")
                continue

            cmd = cmd_fun(option, mode)
            print(f"\tChecking normal compilation.\n\tRunning '{cmd}':")
            # fargopy.Sys is used here
            error, output = fargopy.Sys.run(cmd)
            if not os.path.isfile(f"{fargo_dir}/{Conf.FP_FARGO3D_BINARY}"):
                print(
                    f"\t\tCompilation failed for '{mode}'. Check log file '/tmp/fargo_{mode}.log'"
                )
                exec(f"Conf.FP_FARGO3D_{mode.upper()} = 0")
            else:
                print(f"\t\tCompilation in mode {mode} successful.")
                exec(f"Conf.FP_FARGO3D_{mode.upper()} = 1")

        print(f"Summary of compilation modes:")
        print(f"\tRegular: {Conf.FP_FARGO3D_REGULAR}")
        print(f"\tGPU: {Conf.FP_FARGO3D_GPU}")
        print(f"\tParallel: {Conf.FP_FARGO3D_PARALLEL}")


# Showing version
def _welcome():
    """Welcome message"""
    print(
        f"Running FARGOpy version {__version__}.\n"
        "NOTE: Since alpha versions (<=0.X.X) a major refactor has been done in versions 1.1.X.\n"
        "Please check the documentation for more information."
    )


###############################################################
# Initialization logic (moved from __init__.py)
###############################################################
# Avoid warnings
warnings.filterwarnings("ignore")

# Read FARGOpy configuration variables
if not os.path.isdir(Conf.FP_DOTDIR):
    print(f"Configuring FARGOpy for the first time")
    initialize("configure")
Debug.trace(f"::Reading configuration variables")

# Load configuration variables into Conf
conf_dict = dict()
if os.path.isfile(Conf.FP_RCFILE):
    exec(open(f"{Conf.FP_RCFILE}").read(), dict(), conf_dict)
Conf.__dict__.update(conf_dict)

# Derivative configuration variables
Debug.VERBOSE = Conf.FP_VERBOSE
Conf.FP_FARGO3D_DIR = (Conf.FP_FARGO3D_BASEDIR + "/" + Conf.FP_FARGO3D_PACKDIR).replace(
    "//", "/"
)
Conf.FP_FARGO3D_LOCKFILE = f"{Conf.FP_DOTDIR}/fargopy.lock"

# Check if version in RCFILE is different from installed FARGOpy version
if Conf.FP_VERSION != __version__:
    print(
        f"Your configuration file version '{Conf.FP_VERSION}' it is different than the installed version of FARGOpy '{__version__}'"
    )
    # Interactive check only if stream is a TTY? Or just skip if not?
    # Original code asks for input. This might be blocking in some envs.
    # We'll keep it as is, but rely on it not triggering often in CI/tests if version matches.
    # However, since I'm creating a new file, I'll assume users might encounter this.
    try:
        # Check if sys.stdin has interactive capabilities
        if sys.stdin.isatty():
            ans = input(
                f"Do you want to update configuration file '{Conf.FP_RCFILE}'? [Y/n]: "
            )
            if ans and ("Y" not in ans.upper()):
                if "N" in ans.upper():
                    print("We will keeping asking you this until you update it, sorry!")
            else:
                os.system(f"cp -rf {Conf.FP_RCFILE} {Conf.FP_RCFILE}.save")
                initialize("configure", force=True)
    except Exception:
        pass


class FieldsHandler(Fargobj):
    """Base class for handling fields in FARGOpy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_interpolator(self):
        # Check if sim is properly loaded
        if (
            not hasattr(self, "sim")
            or self.sim == "<not serializable>"
            or isinstance(self.sim, str)
        ):
            raise AttributeError(
                "FieldsHandler.sim is not properly initialized. "
                "This typically happens when the object was loaded from a JSON file (saved with save_object()). "
                "FieldsHandler requires the full Simulation object which can only be preserved with pickle. "
                "Please save with save_object_pkl() instead and reload."
            )

        handler = fargopy.FieldInterpolator(self.sim)
        handler.load_data(
            fields=self.fields,
            slice=self.slice,
            snapshots=self.snapshot,
            cut=self.cut,
            coords=self.coords,
        )
        return handler

    def get_raw_data(self):
        # Check if sim is properly loaded
        if (
            not hasattr(self, "sim")
            or self.sim == "<not serializable>"
            or isinstance(self.sim, str)
        ):
            raise AttributeError(
                "FieldsHandler.sim is not properly initialized. "
                "This typically happens when the object was loaded from a JSON file (saved with save_object()). "
                "FieldsHandler requires the full Simulation object which can only be preserved with pickle. "
                "Please save with save_object_pkl() instead and reload."
            )

        if not self.sim.has("vars"):
            dims, vars, domains = self.sim.load_properties()

        snapshot = 0 if self.snapshot is None else self.snapshot
        loaded_fields = []

        for field in self.fields:
            # Infer field type unless provided
            field_type = self.type
            if field_type is None:
                if field in ["gasdens", "gasenergy"]:
                    field_type = "scalar"
                elif field == "gasv":
                    field_type = "vector"
                else:
                    raise ValueError(f"Field type for '{field}' could not be inferred.")

            # Load scalar
            if field_type == "scalar":
                file_name = f"{field}{snapshot}.dat"
                file_field = os.path.join(self.sim.output_dir, file_name)
                data = self.sim._load_field_scalar(file_field)

            # Load vector
            elif field_type == "vector":
                data = []
                components = ["x", "y"] + (["z"] if self.sim.vars.DIM == 3 else [])
                for comp in components:
                    file_name = f"{field}{comp}{snapshot}.dat"
                    file_field = os.path.join(self.sim.output_dir, file_name)
                    data.append(self.sim._load_field_scalar(file_field))
                data = np.array(data)

            # Create Field
            loaded_field = fargopy.Field(
                data=np.array(data),
                coordinates=self.sim.vars.COORDINATES,
                domains=self.sim.domains,
                type=field_type,
            )

            # Apply slicing
            if self.slice:
                sliced_data, mesh = loaded_field.meshslice(slice=self.slice)
                loaded_field = fargopy.Dictobj(dict=dict(data=sliced_data, mesh=mesh))

            loaded_fields.append(loaded_field)

        result = loaded_fields if len(loaded_fields) > 1 else loaded_fields[0]
        return result
