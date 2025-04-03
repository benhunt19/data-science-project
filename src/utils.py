# Python module to ad-hoc functions and classes

import geopandas as gpd
import matplotlib.pyplot as plt

COUNTRIES_BOARDER_FILE_PATH = './data/geopandas/ne_110m_admin_0_countries/ne_110m_admin_0_countries.shp'

class WorldMap:
    """
    Description:
        Class to handle world map overplots in matplotlib
    """
    
    def __init__(self):
          pass
    
    @staticmethod
    def worldMapPlot():
        
        """
        Description:
            Overplot world border
        Example Susage:
            WorldMap.worldMapPlot().plot(linewidth=1, color='black')
            plt.show()
        """

        # Load world map from GeoPandas
        world = gpd.read_file(COUNTRIES_BOARDER_FILE_PATH)
        
        return world.boundary
        
if __name__ == '__main__':
    
    WorldMap.worldMapPlot().plot(linewidth=1, color='black')
    plt.show()