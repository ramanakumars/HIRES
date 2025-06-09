import argparse
import glob
import logging
import os
import pathlib

import tqdm
from astropy.io import fits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Update missing header files in the reduced data from the raw headers"
)
parser.add_argument("-folder", "--folder", type=pathlib.Path, required=True)
parser.add_argument("-raw_folder", "--raw_folder", type=pathlib.Path, required=True)
args = parser.parse_args()

folder = args.folder
raw_folder = args.raw_folder

files = sorted(glob.glob(os.path.join(folder, "*.fits")))

logger.info(f"Found {len(files)} FITS files")

keys = ["ROTPOSN", "DATE", "DECKNAME", "TARGNAME"]

for file in tqdm.tqdm(files):
    fname = os.path.basename(file)
    raw_file = (
        str(file)
        .replace(str(folder), str(raw_folder))
        .replace("_reduced.fits", ".fits")
    )

    with fits.open(file, mode='update') as outhdu:
        with fits.open(raw_file) as inhdu:
            for key in keys:
                outhdu[0].header[key] = inhdu[0].header[key]
