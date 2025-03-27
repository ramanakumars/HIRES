import numpy as np
import tqdm
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.time import Time
from scipy.signal import savgol_filter
from astropy import units as u
import glob
from astroquery.jplhorizons import Horizons
import os
import logging
import argparse
import pathlib
from calibration_utils import get_data_from_fits


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_solar_spectra(file: str, time: str) -> tuple[np.ndarray, np.ndarray]:
    AU = 1.496e11
    # sun_jup_dist = 4.958302537873 * AU
    jupiter = Horizons(id=599, location='T16', epochs=Time(time).mjd)
    sun_jup_dist = jupiter.ephemerides()['r'][0]

    with fits.open(file) as hdulist:
        data = np.asarray(hdulist[1].data[:])
        data = np.asarray([[dati[0], dati[1]] for dati in data])

    spectra = data[:, 1] * u.erg / u.s / (u.cm ** 2) / u.Angstrom  # in W/m^2/m
    spectra = spectra.to('W/m^2/nm')
    spectra_jupiter = spectra * (4 * np.pi * AU**2.) / (4 * np.pi * sun_jup_dist**2.)

    return data[:, 0], spectra_jupiter


parser = argparse.ArgumentParser(description="Get the I/F for a set of input FITS files using a calibration file")
parser.add_argument("-input_folder", "--input_folder", type=pathlib.Path, required=True)
parser.add_argument("-calibration_file", "--calibration_file", type=pathlib.Path, required=True)
parser.add_argument("-solar_spectra", "--solar_spectra", type=pathlib.Path, required=True)
parser.add_argument("-output_folder", "--output_folder", type=pathlib.Path, required=True)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

jupiter_fits = sorted(glob.glob(os.path.join(args.input_folder, "*.fits")))

calibration = np.load(args.calibration_file)

for i, file in enumerate(tqdm.tqdm(jupiter_fits)):
    filename = os.path.basename(file)
    header, datai, wavei = get_data_from_fits(file)
    calibrated_data = np.zeros_like(datai)

    # area of the pixel in steradians
    ster = float(header['SPASCALE']) * float(header['SPESCALE']) / (206265 * 206265)

    for pos in range(61):
        calibrated_data[:, pos] = savgol_filter(datai[:, pos, :], 201, 2, axis=-1) / calibration / ster / np.pi

    colors = plt.cm.tab20(np.linspace(0, 1, 61))
    sol = np.zeros_like(calibrated_data)

    solar_wavelength, solar_spectra = get_solar_spectra(args.solar_spectra, header['DATE-OBS'])

    for k in range(31):
        sol[k, :] = np.interp(wavei[k], solar_wavelength / 10, solar_spectra.value)

    if_jupiter = calibrated_data / sol

    with fits.open(file) as hdulist:
        hdu = hdulist[0]
        hdu.data = if_jupiter
        hdulist.writeto(os.path.join(args.output_folder, filename.replace(".fits", "_cal.fits")), overwrite=args.overwrite)
