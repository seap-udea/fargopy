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

    QERROR = True
    STDERR = ''
    STDOUT = ''
    OUT = ''

    @staticmethod
    def run(cmd,quiet=True):
        """Run a system command
        
        Parameters:
            cmd: string:
                Command to run
            quiet: boolean, default = True:
                When False the output of the command is shown.
            
        Output:
            error: integer:
                Error code (0,-1,>0)

            output: list:
                List with output. 
                If error == 0, output[:-1] will contain 
                the output line by line.
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
        """Lock a directory using content information
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
        """UnLock a directory
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
        """This routine sleeps for a 'time'. In the meanwhile checks if there is a keyboard interrupt 
        (Enter or Ctrl+C) and interrupt sleeping

        Examples:
            >>> Sys.sleep_timeout()
            >>> Sys.sleep_timeout(10) 

            >>> for i in range(10):
                   print(f"i = {i}")
                   if Sys.sleep_timeout():break
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