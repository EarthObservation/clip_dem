"""
Creates a clipped DEM from a VRT file and saves it as GeoTIFF. Inputs are extents (bbox), but can be quickly
changed to use polygons instead (from SHP file).

29/10/2021 ADDED:
- resample to 1m (needed for orthorectification)
- fill NaNs
"""

import rasterio
from rasterio.mask import mask
from pyproj import CRS
from shapely.geometry import box
import affine
from rasterio.warp import reproject, Resampling
import numpy as np
from rasterio.fill import fillnodata
import time

t1 = time.time()

# Define in and out paths
vrt_path = "l:\\Slovenija\\DEM_D96_05m_virtual_mosaic.vrt"
out_path = ".\\test_final.tif"

# Extents
dst_crs = CRS.from_epsg(3794)
# dst_bounds = 408000, 68000, 410000, 70000
dst_bounds = 399250, 61595, 418730, 82747
left, bottom, right, top = dst_bounds

# Set parameters for resampling
xres = yres = 1  # target resolution
resampled_transform = affine.Affine(xres, 0.0, left, 0.0, -yres, top)

with rasterio.open(vrt_path, "r") as vrt:

    # CLIP
    print("CLIP")
    polygon = [box(left, bottom, right, top)]
    out_image, out_transform = mask(vrt, polygon, crop=True)
    print(time.time() - t1)

    # SAVE META FOR OUTPUT
    out_profile = vrt.profile.copy()
    # read original resolution from metadata
    v_x, v_y = vrt.res

    # RESAMPLING
    x_res_ratio = v_x / xres
    y_res_ratio = v_y / yres
    # create empty array
    resampled_image = np.empty(
        shape=(
            out_image.shape[0],
            int(round(out_image.shape[1] * y_res_ratio)),
            int(round(out_image.shape[2] * x_res_ratio)))
    )
    # resample by using reproject
    print("RESAMPLE")
    reproject(
        out_image, resampled_image,
        src_transform=out_transform,
        dst_transform=resampled_transform,
        src_crs=dst_crs,
        dst_crs=dst_crs,
        resample=Resampling.bilinear
    )
    print(time.time() - t1)

    # FILL NANS
    if (resampled_image == out_profile["nodata"]).sum() > 0:
        print("FILL NANS")
        nan_mask = ~(resampled_image == out_profile["nodata"])
        result = fillnodata(resampled_image, mask=nan_mask)
        print(time.time() - t1)
    else:
        result = resampled_image

# Update metadata for final output
out_profile.update(
    driver="GTiff",
    compress="lzw",
    count=1,
    width=resampled_image.shape[2],
    height=resampled_image.shape[1],
    dtype=resampled_image.dtype,
    transform=resampled_transform,
    tiled=False,
    blockxsize=None,
    blockysize=None,
    crs=CRS.from_epsg(3794)
)

# Save geotiff
print("SAVING GEOTIFF")
with rasterio.open(out_path, "w", **out_profile, BIGTIFF="YES") as dst:
    dst.write(result)
print(time.time() - t1)

print("DONE")

