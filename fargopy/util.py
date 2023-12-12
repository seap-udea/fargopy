###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import gdown
import os
import matplotlib.pyplot as plt

###############################################################
# Constants
###############################################################
PRECOMPUTED_BASEURL = 'https://docs.google.com/uc?export=download&id='
PRECOMPUTED_SIMULATIONS = dict(
    # Download link: https://drive.google.com/file/d/1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3/view?usp=drive_link
    fargo=dict(id='1YXLKlf9fCGHgLej2fSOHgStD05uFB2C3',size=55),
)

###############################################################
# Classes
###############################################################

class Util(object):

    @staticmethod
    def download_precomputed(setup=None,download_dir='/tmp',quiet=True,clean=True):
        """Download a precomputed output from Google Drive FARGOpy public repository.

        Args:
            setup: string, default = None:
                Name of the setup. For a list see fargopu.PRECOMPUTED_SIMULATIONS dictionary.

            download_dir: string, default = '/tmp':
                Directory where the output will be downloaded and uncompressed.

        Optional args:
            quiet: bool, default = True:
                If True download quietly (no progress bar).
            
            clean: bool, default = False:
                If True remove the tgz file after uncompressing it.

        Return:
            If successful returns the output directory.

        """
        if setup is None:
            print(f"You must provide a setup name. Available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        if not os.path.isdir(download_dir):
            print(f"Download directory '{download_dir}' does not exist.")
            return ''
        if setup not in PRECOMPUTED_SIMULATIONS.keys():
            print(f"Precomputed setup '{setup}' is not among the available setups: {list(PRECOMPUTED_SIMULATIONS.keys())}")
            return ''
        
        output_dir = (download_dir + '/' + setup).replace('//','/')
        if os.path.isdir(output_dir):
            print(f"Precomputed output directory '{output_dir}' already exist")
            return output_dir
        else:
            filename = setup + '.tgz'
            fileloc = download_dir + '/' + filename
            if os.path.isfile(fileloc):
                print(f"Precomputed file '{fileloc}' already downloaded")
            else:
                # Download the setups
                print(f"Downloading {filename} from cloud (compressed size around {PRECOMPUTED_SIMULATIONS[setup]['size']} MB) into {download_dir}")
                url = PRECOMPUTED_BASEURL + PRECOMPUTED_SIMULATIONS[setup]['id']
                gdown.download(url,fileloc,quiet=quiet)
            # Uncompress the setups
            print(f"Uncompressing {filename} into {output_dir}") 
            fargopy.Sys.simple(f"cd {download_dir};tar zxf {filename}")
            print(f"Done.")
            fargopy.Sys.simple(f"cd {download_dir};rm -rf {filename}")
            return output_dir

    @staticmethod
    def fargopy_mark(ax):
        """Add a water mark to a 2d or 3d plot.
        
        Parameters:
        
            ax: Class axes: 
                Axe where the watermark will be placed.
        """
        #Get the height of axe
        axh=ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted()).height
        fig_factor=axh/4
        
        #Options of the water mark
        args=dict(
            rotation=270,ha='left',va='top',
            transform=ax.transAxes,color='pink',fontsize=6*fig_factor,zorder=100
        )
        
        #Text of the water mark
        mark=f"FARGOpy {fargopy.version}"
        
        #Choose the according to the fact it is a 2d or 3d plot
        try:
            ax.add_collection3d
            plt_text=ax.text2D
        except:
            plt_text=ax.text
            
        text=plt_text(1,1,mark,**args);
        return text
