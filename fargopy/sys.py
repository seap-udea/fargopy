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
import psutil

# Remove zombie subprocesses
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

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

    @staticmethod
    def run(cmd,quiet=True):
        """Run a system command
        
        Parameters:
            cmd: string:
                Command to run
            verbose: boolean, default = True:
                When True the output of the command is shown.
            
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
    def compile_fargo3d(self,clean=True):
        if Conf._check_fargo(Conf.FARGO3D_FULLDIR):    
            if clean:
                Sys.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} clean mrproper')
            
            error,out = Sys.sysrun(f'make -C {Conf.FARGO3D_FULLDIR} PARALLEL={self.parallel} GPU={self.gpu}',verbose=False)
            if error:
                if not Conf._check_fargo_binary(Conf.FARGO3D_FULLDIR,quiet=True):
                    print("An error compiling the code arose. Check dependencies.")
                    print(Sys.STDERR)
                return False
            
            return True
    
    @staticmethod
    def get_memory():
        svmem = psutil.virtual_memory()
        return svmem
    