"""
Creates *.vrt file from a list of TIF files.
"""

from osgeo import gdal
import glob
import time

t1 = time.time()

# tif_list = [
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_380000_133000_dem_05m_buffered.tif",
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_380000_134000_dem_05m_buffered.tif",
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_380000_135000_dem_05m_buffered.tif",
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_381000_133000_dem_05m_buffered.tif",
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_381000_134000_dem_05m_buffered.tif",
#     "l:\\Slovenija\\DEM_D96_05m_buffered\\tile_381000_135000_dem_05m_buffered.tif"
# ]

tif_list = glob.glob("l:\\Slovenija\\DEM_D96_05m_buffered\\*.tif")

print(len(tif_list))

vrt_options = gdal.BuildVRTOptions()
my_vrt = gdal.BuildVRT('dem_slo_05_mosaic_full.vrt', tif_list, options=vrt_options)
my_vrt = None

t1 = time.time() - t1
print(f"Done in {t1} sec.")
