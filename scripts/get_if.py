import argparse
import glob
import logging
import os
import pathlib

import numpy as np
import tqdm
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.signal import savgol_filter

from hiresprojection.io_utils import get_data_from_fits

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_solar_spectra(file: str, time: str) -> tuple[np.ndarray, np.ndarray]:
    AU = 1.496e11
    # sun_jup_dist = 4.958302537873 * AU
    jupiter = Horizons(id=599, location='T16', epochs=Time(time).mjd)
    sun_jup_dist = jupiter.ephemerides()['r'][0] * AU

    with fits.open(file) as hdulist:
        data = np.asarray(hdulist[1].data[:])
        data = np.asarray([[dati[0], dati[1]] for dati in data])

    spectra = data[:, 1] * u.erg / u.s / (u.cm**2) / u.Angstrom  # in W/m^2/m
    spectra = spectra.to('W/(m^2 nm)')
    spectra_jupiter = spectra * (4 * np.pi * AU**2.0) / (4 * np.pi * sun_jup_dist**2.0)

    return data[:, 0], spectra_jupiter


parser = argparse.ArgumentParser(
    description="Get the I/F for a set of input FITS files using a calibration file"
)
parser.add_argument("-input_folder", "--input_folder", type=pathlib.Path, required=True)
parser.add_argument(
    "-calibration_file", "--calibration_file", type=pathlib.Path, required=True
)
parser.add_argument(
    "-solar_spectra", "--solar_spectra", type=pathlib.Path, required=True
)
parser.add_argument(
    "-extinction_table", "--extinction_table", type=pathlib.Path, required=True
)
parser.add_argument(
    "-sky_background", "--sky_background", type=pathlib.Path, required=True
)
parser.add_argument(
    "-output_folder", "--output_folder", type=pathlib.Path, required=True
)
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

jupiter_fits = sorted(glob.glob(os.path.join(args.input_folder, "*.fits")))

calibration = np.load(args.calibration_file)
extinction = np.load(args.extinction_table)
sky_data = np.load(args.sky_background)

sky = savgol_filter(sky_data['data'], 201, 1, axis=-1)
sky_error = sky_data['error']

for i, file in enumerate(tqdm.tqdm(jupiter_fits)):
    filename = os.path.basename(file)
    header, datai, unci, wavei = get_data_from_fits(file)

    if 'jupiter' not in header['targname'].lower():
        continue

    calibrated_data = np.zeros_like(datai)
    calibrated_unc = np.zeros_like(datai)
    airmass = float(header['AIRMASS'])

    opacity = np.exp(-extinction['extinction'] * airmass)
    # d(exp(x)) = exp(x) dx
    opacity_error = opacity * extinction['error']

    opacity = np.repeat(opacity[:, np.newaxis], datai.shape[-1], axis=-1)
    opacity_error = np.repeat(opacity_error[:, np.newaxis], datai.shape[-1], axis=-1)

    # area of the pixel in steradians
    ster = float(header['SPASCALE']) * float(header['SPESCALE']) / (206265 * 206265)

    for pos in range(61):
        data_pos_i = savgol_filter(datai[:, pos, :], 201, 1, axis=-1)  # - sky
        unc_pos_i = np.sqrt(
            savgol_filter(unci[:, pos, :], 201, 1, axis=-1) + sky_error**2.0
        )
        calibrated_data[:, pos] = (
            data_pos_i / calibration['calibration'] / ster / np.pi * opacity
        )
        calibrated_unc[:, pos] = (
            np.sqrt(
                +((unc_pos_i / data_pos_i) ** 2.0)
                + (calibration['error'] / calibration['calibration']) ** 2.0
                + (opacity_error / opacity) ** 2.0
            )
            * calibrated_data[:, pos]
        )

    sol = np.zeros_like(calibrated_data)

    solar_wavelength, solar_spectra = get_solar_spectra(
        args.solar_spectra, header['DATE-OBS']
    )

    for k in range(31):
        sol[k, :] = np.interp(wavei[k], solar_wavelength / 10, solar_spectra.value)

    if_jupiter = calibrated_data / sol
    if_jupiter_error = calibrated_unc / sol

    with fits.open(file) as hdulist:
        hdulist[0].data = if_jupiter
        hdulist[1].data = if_jupiter_error
        # hdulist[3].data = wavei
        # hdulist[4].data = hdulist[4].data[:, :-50]
        hdulist.writeto(
            os.path.join(args.output_folder, filename.replace(".fits", "_cal.fits")),
            overwrite=args.overwrite,
        )
