"""
STEP 3: Extract raster covered by polygon and calculate statistics. It is an
updated version of raster2table.py
27/07/2021

Version 4
- use buffer and remove polygons that are too small
+ calculate 75 percentile instead of mean for each image

+ for single index, read multiple TIFFs
+ FOR YEARS 2018 and 2019

"""

import glob
import time
from os import makedirs
from os.path import join, basename, dirname

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
from tqdm import tqdm


def r2t(pth_raster, pth_shp, dir_out, suffix):
    """Function creates shape file and csv file containing mean and standard
    deviation of the underlain raster for each polygon.

    Parameters
    ----------
    pth_raster : str
        Path to input raster.
    pth_shp : str
        Path to input shapefile.
    dir_out : str
        Path to directory for saving results.
    suffix : str
        Suffix to be added to results.

    Returns
    -------
        Finished message.
    """

    def ms_rio(geom, raster):
        """Returns subset of an array covered by the input polygon. The input
        polygon has to be in the GeoJSON format. The crop attribute sets values
        not covered by polygon to nan. All_touched is used to prevent empty rasters
        for very thin polygons."""
        out_image, _ = mask(raster, geom, crop=True, all_touched=True, nodata=np.nan)
        return out_image

    # Read polygons to data frame
    print("Reading shapefile...")
    t = time.time()
    tdf = gpd.read_file(pth_shp)
    t = time.time() - t
    print(f"Read SHP with GeoPandas: {t} seconds")

    # Uncomment for processing of a subset (useful for debugging)
    # tdf = tdf[tdf.REGIJA == 4].copy()
    # tdf = tdf[0:100].copy()
    # tdf = tdf[95427:95467].copy()

    # Add buffer and reformat geometry to GeoJSON format (append a new column)
    print("Adding buffer")
    tdf["buffered"] = tdf["geometry"].buffer(-5)
    tqdm.pandas(desc="geometry -> GeoJSON")
    tdf["geom"] = tdf["buffered"].progress_apply(lambda g: [mapping(g)])

    # Remove polygons with 0 area after buffer
    rows_a, _ = tdf.shape
    tdf = tdf[tdf["buffered"].area > 0].copy()
    rows_b, _ = tdf.shape
    rows_removed = rows_a - rows_b
    print(f"Filtering... Removed {rows_removed} rows")

    # Extract rasters
    with rasterio.open(pth_raster) as src:
        tqdm.pandas(desc=f"Extracting arrays from {pth_raster}")
        tdf["ext_rst"] = tdf.geom.progress_apply(lambda g: ms_rio(g, src))

    # Calculate 75th percentile by dates
    tqdm.pandas(desc="75th percentile")
    tdf["all_rstrs"] = tdf.ext_rst.progress_apply(lambda g: np.nanpercentile(g, 75, axis=(1, 2)))

    print(" Splitting LIST to COLUMNS...")
    # Import dates
    dates_file = join(dirname(pth_raster), f"dates_{suffix}" + basename(pth_raster)[10:-3] + "txt")
    # Create list of column names (dates of individual images)
    with open(dates_file) as file:
        dates = [suffix + "_" + x.rstrip('\n')[5:] for x in file]
    # Split columns into new dataframe
    tdf[dates] = gpd.GeoDataFrame(tdf.all_rstrs.tolist(), index=tdf.index)
    tdf = tdf.drop(columns=["all_rstrs", "ext_rst"])

    # Set filename for saving (just CSV)
    out_name = basename(pth_shp)[:-4] + "_" + suffix + ".csv"
    out_name = join(dir_out, out_name)

    # Save temp CSV
    print("Saving final CSV...")
    makedirs(dir_out, exist_ok=True)
    tdf = tdf.drop(columns=["geometry", "geom", "buffered"])
    tdf.to_csv(out_name, index=False)

    return "DONE!"


if __name__ == "__main__":
    # SELECT INDEX
    for suff in ["evi2", "ndvi", "ndwi", "savi"]:
        # Input rasters
        geotiff = f"d:\\susa_temp_19\\stack_{suff.upper()}_2019.tif"

        # Input shape file
        shapefile = "t:\\susa\\ARSKTRP_ZV_clean\\KMRS2019_d96tm_clean.shp"

        # Output folder and suffix to be added to SHP and CSV files
        output_loc = "t:\\susa\\ZV2019_pc075"

        print(f">>> Run for: {suff}")

        result = r2t(geotiff, shapefile, output_loc, suff)
        print(result)
