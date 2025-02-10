import glob
import os
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import numpy as np
import tqdm
from hiresprojection.project import project_and_save
import logging
import argparse
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Limb fit a set of MAGIQ files in a folder")
parser.add_argument("-folder", "--folder", type=pathlib.Path, required=True)
parser.add_argument("-guide_folder", "--guide_folder", type=pathlib.Path, required=True)
parser.add_argument("-navfolder", "--navfolder", type=pathlib.Path, required=True)
args = parser.parse_args()

folder = args.folder
guide_folder = args.guide_folder
navfolder = args.navfolder


guide_fits = sorted(glob.glob(os.path.join(guide_folder, "*.fits")))
guide_times = []
guide_RAs = []
for file in tqdm.tqdm(guide_fits, desc="Loading guide images"):
    with fits.open(file) as hdulist:
        header = hdulist[0].header
        time = Time(header['DATE-OBS'] + 'T' + header['UTC'])
        guide_times.append(time.mjd)
        # guide_times.append(float(hdulist[0].header['MJD-OBS']))
        guide_RAs.append(float(hdulist[0].header['RA']))
guide_RAs = np.asarray(guide_RAs)
guide_times = np.asarray(guide_times)

files = sorted(glob.glob(os.path.join(folder, "*.fits")))

logger.info(f"Found {len(files)} FITS files")

if not os.path.exists(navfolder):
    os.makedirs(navfolder)

for file in files:
    fname = os.path.basename(file)
    logger.info(f"Processing {fname}")

    with fits.open(file) as hdulist:
        header = hdulist[0].header
        time = Time(hdulist[0].header['DATE-OBS']).mjd

    ind = np.argmin((time - guide_times)**2.)
    guider_fits = guide_fits[ind]
    time_diff = (guide_times[ind] - time) * 24 * 3600
    if np.abs(time_diff) < 10:
        with fits.open(guider_fits) as guider_hdus:
            wcs = WCS(guider_hdus[0].header)
        radec0 = wcs.pixel_to_world([273], [240])
        navfile = os.path.join(navfolder, fname.replace('.fits', '_nav.fits'))
        project_and_save(file, navfile, radec0)
    else:
        logger.warning(f"Could not find close guide image for matching image center for {fname}. Closest time diffence was {time_diff} seconds")
