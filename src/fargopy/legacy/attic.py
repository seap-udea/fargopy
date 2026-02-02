
class Fargo3d(Fargobj):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
   
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
