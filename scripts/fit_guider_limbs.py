import argparse
import glob
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

from hiresprojection.fit_utils import get_ellipse_params
from hiresprojection.guide import GuiderImageProjection, flip_north_south

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Limb fit a set of MAGIQ files in a folder"
)
parser.add_argument("-folder", "--folder", type=pathlib.Path, required=True)
parser.add_argument("-plotfolder", "--plotfolder", type=pathlib.Path, required=True)
args = parser.parse_args()

folder = args.folder
plotfolder = args.plotfolder

files = sorted(glob.glob(os.path.join(folder, "*.fits")))

if not os.path.exists(plotfolder):
    os.makedirs(plotfolder)

logger.info(f"Found {len(files)} FITS files")

for file in files:
    fname = os.path.basename(file)
    plotname = os.path.join(plotfolder, f"{fname}.png")
    if os.path.exists(plotname):
        continue

    logger.info(f"Processing {fname}")

    projector = GuiderImageProjection(file)
    contour = projector.detect_limb(gamma=0.1, threshold=0.45)

    wcs = projector.wcs
    wcs_fit = get_ellipse_params(
        contour, projector.limbRADec, projector.subpt, wcs, projector.data.shape
    )

    limbRADecSky = [
        SkyCoord(ra=radec[0] * u.radian, dec=radec[1] * u.radian, frame='fk5')
        for radec in projector.limbRADec
    ]
    limbpix = np.asarray([wcs_fit.world_to_pixel(radec) for radec in limbRADecSky])

    # we know that the north pole is to the left of the image
    if limbpix[0, 0] > limbpix[int(len(limbpix) / 2), 0]:
        flip_north_south(wcs_fit)
        limbRADecSky = [
            SkyCoord(ra=radec[0] * u.radian, dec=radec[1] * u.radian, frame='fk5')
            for radec in projector.limbRADec
        ]
        limbpix = np.asarray([wcs_fit.world_to_pixel(radec) for radec in limbRADecSky])

    projector.update_fits_wcs(wcs_fit)

    fig, ax = plt.subplots(1, 1, dpi=150)
    ax.imshow(projector.data, cmap='grey')
    ax.plot(*contour.T, 'gx', markersize=0.5)
    ax.plot(limbpix[:, 0], limbpix[:, 1], 'r-')
    plt.savefig(plotname)
    plt.close('all')
