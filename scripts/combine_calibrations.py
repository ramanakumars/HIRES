import argparse
import glob
import logging
import os
import pathlib

import numpy as np

from hiresprojection.calibration_utils import get_standard_error_of_mean

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Combine calibrations together"
)
parser.add_argument("-folder", "--folder", type=pathlib.Path, required=True)
parser.add_argument("-output", "--output", type=str, required=True, default="calibration.npz")
args = parser.parse_args()

calibration_files = sorted(glob.glob(os.path.join(args.folder, "*.npz")))

calibrations = []
calibrations_error = []
    
for calibration_file in calibration_files:
    file_data = np.load(calibration_file)
    data = file_data['calibration']
    calibrations.append(file_data['calibration'])
    calibrations_error.append(file_data['error'])

median_calibration = np.median(calibrations, axis=0)
median_error = 1.253 * get_standard_error_of_mean(np.asarray(calibrations), np.asarray(calibrations_error), axis=0)

np.savez(args.output, calibration=median_calibration, error=median_error)
