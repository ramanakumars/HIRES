import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from scipy.optimize import dual_annealing
from scipy.signal import savgol_filter

from hiresprojection.calibration_utils import get_data_from_fits

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

    spectra = data[:, 1] * u.erg / u.s / (u.cm**2) / u.Angstrom  # in W/m^2/m
    spectra = spectra.to('W/m^2/nm')
    spectra_jupiter = spectra * (4 * np.pi * AU**2.0) / (4 * np.pi * sun_jup_dist**2.0)

    return data[:, 0], spectra_jupiter


parser = argparse.ArgumentParser(
    description="Fit the Minnaert coefficients for a Jupiter spectra"
)
parser.add_argument(
    "-input_files", "--input_files", type=pathlib.Path, nargs='+', required=True
)
parser.add_argument(
    "-calibration_file", "--calibration_file", type=pathlib.Path, required=True
)
parser.add_argument(
    "-solar_spectra", "--solar_spectra", type=pathlib.Path, required=True
)
parser.add_argument("-output_file", "--output_file", type=pathlib.Path, required=True)
args = parser.parse_args()

jupiter_fits = args.input_files

calibration = np.load(args.calibration_file)

mu = []
mu0 = []
ifs = []

for i, file in enumerate(jupiter_fits):
    with fits.open(file) as hdulist:
        latitude = hdulist[6].data[:]
        longitude = hdulist[5].data[:]
        incidence = hdulist[7].data[:]
        emission = hdulist[8].data[:]

    header, datai, wavei = get_data_from_fits(file)
    calibrated_data = np.zeros_like(datai)

    # area of the pixel in steradians
    ster = float(header['SPASCALE']) * float(header['SPESCALE']) / (206265 * 206265)

    for pos in range(61):
        calibrated_data[:, pos] = (
            savgol_filter(datai[:, pos, :], 201, 2, axis=-1)
            / calibration
            / ster
            / np.pi
        )

    colors = plt.cm.tab20(np.linspace(0, 1, 61))
    sol = np.zeros_like(calibrated_data)

    solar_wavelength, solar_spectra = get_solar_spectra(
        args.solar_spectra, header['DATE-OBS']
    )

    for k in range(31):
        sol[k, :] = np.interp(wavei[k], solar_wavelength / 10, solar_spectra.value)

    if_jupiter = calibrated_data / sol
    if_jupiter_max = if_jupiter[:, np.argmin(incidence), :]
    if_jupiter = np.nanmedian(
        if_jupiter / np.repeat(if_jupiter_max[:, np.newaxis, :], 61, axis=1), axis=-1
    )

    mu0.append(np.cos(np.radians(incidence))[:-3])
    mu.append(np.cos(np.radians(emission))[:-3])
    ifs.append(if_jupiter[:, :-3])

mu0 = np.asarray(mu0)
mu = np.asarray(mu)
ifs = np.asarray(ifs)

print(ifs.shape)

minnaert_k = np.zeros(ifs.shape[1])
minnaert_b = np.zeros(ifs.shape[1])

for k in range(ifs.shape[1]):
    mu0_k = mu0.flatten()
    mu_k = mu.flatten()
    ifs_k = ifs[:, k].flatten()

    print(ifs_k.min(), ifs_k.max(), mu0_k.min(), mu0_k.max(), mu_k.min(), mu_k.max())

    def func(x):
        ki, bi = x
        return np.mean(
            np.abs(ki * np.log(mu0_k) + (ki - 1) * np.log(mu_k) + bi - np.log(ifs_k))
        )

    bounds = [[0, 2], [-2, 2]]
    try:
        output = dual_annealing(func, bounds)

        minnaert_k[k] = output.x[0]
        minnaert_b[k] = np.exp(output.x[1])

        print(output.x[0])
    except ValueError as e:
        print(e)
        continue

np.savez(args.output_file, minnaert_k=minnaert_k, minnaert_b=minnaert_b)
