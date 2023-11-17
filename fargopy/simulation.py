###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import subprocess
import os

###############################################################
# Module constants
###############################################################

###############################################################
# Classes
###############################################################
class Simulation(fargopy.Conf):
    """Simulation class.
    """
    def __init__(self,**kwargs):
        super().__init__()

        if 'setup' in kwargs.keys():
            self.load_setup(setup=kwargs['setup'])
        else:
            self.setup = None

        # Compilation options
        self.parallel = fargopy.Conf.FARGO3D_PARALLEL
        self.gpu = fargopy.Conf.FARGO3D_GPU

        # Other options
        self.compile_options = None
        self.fargo3d_binary = None
        self.fargo3d_process = None

    def load_setup(self,setup=None):
        if not setup:
            setups = self.list_setups()
            print(f"You must select a setup, from a custom directory or from the list:{setups}")
        
        output = fargopy.Conf._check_setup(setup)
        if output:
            self.setup = setup
            self.setup_directory = output[0]
            self.setup_parameters = output[1]
            self.setup_outputs = f"{self.setup_directory}/../../outputs/{self.setup}"

    def compile(self,
                parallel=None,gpu=None,
                options='',
                force=False
                ):
        if not self._is_setup():
            return
        
        parallel = self.parallel if parallel is None else parallel
        gpu = self.gpu if gpu is None else gpu

        self.compile_options = f"SETUP={self.setup} PARALLEL={parallel} GPU={gpu} "+options
        self.fargo3d_binary = fargopy.Conf._compile_fargo3d(options=self.compile_options,force=force)
        
    def run(self,
            mode='async',
            options='-m'):
        
        if not self._is_setup():
            return
        
        if self.fargo3d_binary is None:
            print("You must first compile your simulation with: <simulation>.compile().")
            return
        
        self.logfile = f"{fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/{self.setup}.log"
        self.fargo3d_process = fargopy.Conf._run_fargo3d(f"{self.fargo3d_binary} {options}",
                                                         logfile=self.logfile,
                                                         mode=mode,
                                                         options=self.setup_parameters)

    def stop(self):
        """Stop the present running process
        """
        if not self._is_setup():
            return
        
        if self.fargo3d_process:
            poll = self.fargo3d_process.poll()
            if poll is None:
                print(f"Stopping FARGO3D process (pid = {self.fargo3d_process.pid})")
                subprocess.Popen.kill(self.fargo3d_process)
                del self.fargo3d_process
                self.fargo3d_process = None
            else:
                print("The process has already finished.")    
        else:
            print("No running process.")

    def _is_setup(self):
        if self.setup is None:
            print("The simulation has not been set-up yet. Run <simulation>.load_setup('<setup>')")
            return False
        return True

    def status(self,
               mode='isrunning',
               ):
        """Check the status of the running process

        Parameters:
            mode: string, defaul='isrunning':
                Available modes:
                    'isrunning': Just show if the process is running.
                    'logfile': Show the latest lines of the logfile

        """
        if not self._is_setup():
            return
        
        if 'isrunning' in mode:
            if self.fargo3d_process:
                poll = self.fargo3d_process.poll()

                if poll is None:
                    print("The process is running.")
                else:
                    print(f"The process has ended with termination code {poll}.")

        if 'logfile' in mode:

            if 'logfile' in self.__dict__.keys() and os.path.isfile(self.logfile):
                print("The latest 10 lines of the logfile:\n")
                os.system(f"tail -n 10 {self.logfile}")
            else:
                print("No log file created yet")

        if 'outputs' in mode:
            error,output = fargopy.Util.sysrun(f"ls {fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*.dat",verbose=False)
            if not error:
                files = [file.split('/')[-1] for file in output[:-1]]
                file_list = ""
                for i,file in enumerate(files):
                    file_list += f"{file}, "
                    if ((i+1)%10) == 0:
                        file_list += "\n"
                file_list = file_list.strip("\n,")
                print(f"\n{len(files)} available datafiles:\n")
                self.output_datafiles = files
                print(file_list)
            else:
                print("No datafiles yet available")

    def clean_output(self):
        """
        Clean all outputs
        """
        if not self._is_setup():
            return
        
        print(f"Cleaning simulation outputs...")
        run_cmd = f"rm -r {fargopy.Conf.FARGO3D_FULLDIR}/{self.setup_outputs}/*"
        error,output = fargopy.Util.sysrun(run_cmd,verbose=False)
        if error:
            print("No output directory has been created yet.")
        if not error:
            print("Done.")
