import glob
from hiresprojection.guide import GuiderImageProjection
import numpy as np
from scipy.interpolate import griddata
import argparse
import logging
import pathlib
import os


def plot_map(lon, lat, img, pixres=0.1):
    '''
        project the image onto a lat/lon grid
    '''
    gridlat = np.arange(-90., 90., pixres)
    gridlon = np.arange(0., 360., pixres)

    LAT, LON = np.meshgrid(gridlat, gridlon)

    lon_f = lon.flatten()
    lat_f = lat.flatten()
    img_f = img.flatten()

    mask = np.where(np.isfinite(lon_f) & np.isfinite(lat_f))[0]
    img = img_f[mask]
    lats = lat_f[mask]
    lons = lon_f[mask]

    # convert to east positive
    IMG = griddata((lons, lats), img, (LON, LAT), method='cubic').T

    IMG[np.isnan(IMG)] = 0.
    IMG[IMG < 0.] = 0.

    return IMG


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Limb fit a set of MAGIQ files in a folder")
parser.add_argument("-folder", "--folder", type=pathlib.Path, required=True)
parser.add_argument("-plotfolder", "--plotfolder", type=pathlib.Path, required=True)
args = parser.parse_args()

folder = args.folder
plotfolder = args.plotfolder

if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

files = sorted(glob.glob(os.path.join(folder, "*.fits")))

logger.info(f"Found {len(files)} FITS files")

for file in files:
    fname = os.path.basename(file)
    plotname = os.path.join(plotfolder, f"{fname}.npz")
    if os.path.exists(plotname):
        continue

    logger.info(f"Processing {fname}")

    projector = GuiderImageProjection(file)
    projector.project_to_lonlat()
    map = plot_map(projector.lonlat[:, :, 0], projector.lonlat[:, :, 1], projector.data / 65536)

    np.savez(plotname, lonlat=projector.lonlat, map=map)
