###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import os
import json
import subprocess
import psutil
import sys
from select import select

###############################################################
# Classes
###############################################################

#/////////////////////////////////////
# UTIL CLASS
#/////////////////////////////////////
class Sys(object):
    """System utilities for command execution, directory locking, and resource monitoring.

    The ``Sys`` class provides a collection of static methods to interact with
    the operating system. It handles shell command execution with output capturing,
    directory locking mechanisms for concurrent safety, and system pause functionalities.
    """

    QERROR = True
    STDERR = ''
    STDOUT = ''
    OUT = ''

    @staticmethod
    def run(cmd,quiet=True):
        """Run a system shell command.

        Executes a command string in the shell and captures its stdout and stderr.
        Useful for running FARGO3D executables or other system tools.

        Parameters
        ----------
        cmd : str
            The command string to execute.
        quiet : bool, optional
            If True, suppress printing output to stdout (default: True).

        Returns
        -------
        tuple
            (error_code, output_list)
            - error_code (int): 0 for success, non-zero for error.
            - output_list (list): List of output lines (strings).

        Examples
        --------
        Run a simple shell command:
        
        >>> err, out = fp.Sys.run("ls -l")
        """
        fargopy.Debug.trace(f"sysrun::cmd = {cmd}")

        out=[]
        for path in Sys._run(cmd):
            try:
                if not quiet:
                    print(path.decode('utf-8'))
                out += [path.decode('utf-8')]
            except:
                out += [(path[0],path[1].decode('utf-8'))]
        
        Sys.STDOUT = ''
        if len(out)>1:
            Sys.STDOUT = '\n'.join(out[:-1])
        
        Sys.STDERR = out[-1][1]
        if len(Sys.STDERR)>0:
            Sys.QERROR = out[-1][0]
            if Sys.QERROR == 0:
                Sys.QERROR = -1
        else:
            Sys.QERROR = 0

        if Sys.QERROR and not quiet:
            print(f"Error:\n{Sys.STDERR}")

        if fargopy.Debug.VERBOSE:
            error = out[-1][0]
            if Sys.QERROR>0:
                fargopy.Debug.trace(f"sysrun::Error check Sys.STDERR.")
            elif Sys.QERROR<0:
                fargopy.Debug.trace(f"sysrun::Done. Still some issues must be check. Check Sys.STDOUT and Sys.STDERR for details.")
            elif Sys.QERROR==0:
                fargopy.Debug.trace(f"sysrun::Done. You're great. Check Sys.STDOUT for details.")
        
        Sys.OUT = out
        return Sys.QERROR,out

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
    def simple(cmd):
        return os.system(cmd)

    @staticmethod
    def get_memory():
        svmem = psutil.virtual_memory()
        return svmem

    @staticmethod
    def lock(dir,content=dict()):
        """Create a lock file in a directory.

        Creates a ``fargopy.lock`` file containing the provided content to
        signal that the directory is in use or processing.

        Parameters
        ----------
        dir : str
            Path to the directory to lock.
        content : dict, optional
            Dictionary of metadata to store in the lock file.
        """
        if not os.path.isdir(dir):
            print(f"Locking directory '{dir}' not found.")
            return
        
        filename = f"{dir}/fargopy.lock"
        with open(filename,'w') as file_object:
            file_object.write(json.dumps(content,default=lambda obj:'<not serializable>'))
            file_object.close()

    @staticmethod
    def unlock(dir):
        """Remove the lock file from a directory.

        Parameters
        ----------
        dir : str
            Path to the directory to unlock.
        """
        if not os.path.isdir(dir):
            print(f"Locking directory '{dir}' not found.")
            return
        filename = f"{dir}/fargopy.lock"
        if os.path.isfile(filename):
            fargopy.Sys.simple(f"rm -rf {filename}")
    
    @staticmethod
    def is_locked(dir,verbose=False):
        if not os.path.isdir(dir):
            if verbose:
                print(f"Locking directory '{dir}' not found.")
            return False
        filename = f"{dir}/fargopy.lock"
        if os.path.isfile(filename):
            if verbose:
                print(f"The directory '{dir}' is locked")
            with open(filename) as file_handler:
                info = json.load(file_handler)
                return info
            
    @staticmethod
    def sleep_timeout(timeout=5,msg=None):
        """Sleep for a specified duration, interruptible by user input.

        Sleeps for ``timeout`` seconds. Checks for keyboard interrupt (Enter or Ctrl+C)
        during the sleep period.

        Parameters
        ----------
        timeout : int, optional
            Sleep duration in seconds (default: 5).
        msg : str, optional
            Message to display before sleeping.

        Returns
        -------
        bool
            True if interrupted, False if timeout completed.

        Examples
        --------
        Sleep for 10 seconds or until user interrupt:
        
        >>> if fp.Sys.sleep_timeout(10, "Press Enter to skip"):
        ...     print("Skipped")
        """
        try: 
            if msg:
                print(msg)
            rlist, wlist, xlist = select([sys.stdin], [], [], timeout)
            if rlist:
                return True
            else:
                return False
        except KeyboardInterrupt:
            print("Interrupting")
            return True