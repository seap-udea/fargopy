###############################################################
# FARGOpy interdependencies
###############################################################
import fargopy

###############################################################
# Required packages
###############################################################
import matplotlib.pyplot as plt

###############################################################
# Constants
###############################################################

###############################################################
# Classes
###############################################################
class Plot(object):

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
    

    def plot_heatmap(data, x=None, y=None, title="Heatmap", xlabel="X", ylabel="Y", contour_levels=10):
        """
        Plots a 2D heatmap with pcolormesh and contour.

        Parameters:
            data (2D array): The data to plot.
            x (1D array, optional): X-axis values.
            y (1D array, optional): Y-axis values.
            title (str): Title of the plot.
            xlabel (str): Label for the X-axis.
            ylabel (str): Label for the Y-axis.
            contour_levels (int or list, optional): Number of contour levels or specific levels. Default is 10.
        """
        plt.figure(figsize=(8, 6))
        
        if x is not None and y is not None:
            extent = [x.min(), x.max(), y.min(), y.max()]
            X, Y = np.meshgrid(x, y)
            # Plot the heatmap with pcolormesh
            mesh = plt.pcolormesh(X, Y, data, shading='auto', cmap='Spectral_r')
            # Add contour lines
            contours = plt.contour(X, Y, data, levels=contour_levels, colors='black', linewidths=0.5)
            plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
        else:
            # Plot the heatmap with pcolormesh
            mesh = plt.pcolormesh(data, shading='auto', cmap='Spectral_r')
            # Add contour lines
            contours = plt.contour(data, levels=contour_levels, colors='black', linewidths=0.5)
            plt.clabel(contours, inline=True, fontsize=8, fmt="%.1f")
        
        plt.colorbar(mesh, label="Value")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()