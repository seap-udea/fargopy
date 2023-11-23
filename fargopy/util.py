###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import matplotlib.pyplot as plt

###############################################################
# Classes
###############################################################

class Util(object):

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
