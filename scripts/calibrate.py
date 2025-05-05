import glob
import os
import logging
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import savgol_filter
from astropy import units as u
import pandas as pd
from calibration_utils import star_spectra, get_data_from_fits, get_calibration, get_coarse_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Calibrate the HIRES spectra spectra with a reference star")
parser.add_argument("-star_folder", "--star_folder", type=pathlib.Path, required=True)
parser.add_argument("-target", "--target", type=str, required=True, default="HR8634")
args = parser.parse_args()

STAR_TARGET = args.target

all_star_fits = sorted(glob.glob(os.path.join(args.star_folder, "*.fits")))
star_fits = []
for i, star in enumerate(all_star_fits):
    with fits.open(star) as hdulist:
        header = hdulist[0].header
    if header['targname'].strip() in star_spectra:
        star_fits.append({'target': header['targname'], 'file': star})

logger.info(f"Found {len(star_fits)} files for stars: {set([file['target'] for file in star_fits])}")

if not os.path.exists(STAR_TARGET):
    os.makedirs(STAR_TARGET)

for i, star in enumerate(star_fits):
    if star['target'] == STAR_TARGET:
        _, _, calibration, wavelength = get_calibration(star_fits, star, STAR_TARGET)
        np.save(f'{STAR_TARGET}/calibration_{STAR_TARGET}_{i}.npy', calibration)

cal_files = sorted(glob.glob(f'{STAR_TARGET}/*.npy'))
colors = plt.cm.tab10(np.linspace(0, 1, len(cal_files)))

least_squares = np.zeros((len(cal_files), len(star_spectra)))

for j, (star_name, star_spectrum) in enumerate(star_spectra.items()):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)

    try:
        target = [star for star in star_fits if star['target'] == star_name][0]
    except IndexError:
        print(f"No observations found for {star_name}")
        break
    _, data0_coarse = get_coarse_data(target['file'])
    scaled_star_data = []
    for star in star_fits:
        if star['target'] == star_name:
            file = star['file']

            _, data, wavelength = get_data_from_fits(file)

            peak = 31  # np.argmax(data.mean(axis=(0, 2)))

            object_mask = np.abs(np.arange(61) - peak + 1) < 8
            background_mask = (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)

            wave_coarse, data_coarse = get_coarse_data(file)

            scaling = data0_coarse / data_coarse

            scaled_datai = np.zeros_like(data[:, 0])

            for k in range(31):
                object_spectra = data[k, object_mask, :].sum(axis=0)
                sky = np.median(data[k, background_mask, :], axis=0)
                sky_subtracted = savgol_filter(object_spectra - sky, 201, 1, axis=-1)

                wavei = wave_coarse[k]
                scalei = scaling[k]
                scalei_fine = np.interp(wavelength[k], wavei, scalei)
                scaled_datai[k] = sky_subtracted  # * scalei_fine

            scaled_star_data.append(scaled_datai)

    scaled_star_data = np.asarray(scaled_star_data)
    median_star_data = np.median(scaled_star_data, 0)

    star_wave = star_spectra[star_name].spectral_axis.to('nanometer')
    star_flux = star_spectra[star_name].flux.to('W/(m^2 nm)', equivalencies=u.spectral_density(star_spectra[star_name].spectral_axis))
    ax.plot(star_wave, star_flux, 'r-', linewidth=0.5)
    for i, cal_file in enumerate(cal_files):
        calibration = np.load(cal_file)
        calibrated_star = median_star_data / calibration
        calibrated_star[~np.isfinite(calibrated_star)] = 0.

        least_squares_i = 0.
        for k in range(31):
            ax.plot(wavelength[k], calibrated_star[k], '-', linewidth=0.8, color=colors[i], label=cal_file if k == 0 else None)
            star_flux_interp = np.interp(wavelength[k], star_wave.value, star_flux.value)
            least_squares_i += np.sum((calibrated_star[k] - star_flux_interp)**2.)
        least_squares[i, j] = least_squares_i / wavelength.size

    # ax.set_ylim((0, 1e5))
    ax.legend(loc='upper right', ncols=2, fontsize=8)
    ax.set_ylabel(r"Flux [W/m$^2$/nm]")
    ax.set_xlabel(r"Wavelength [nm]")
    ax.set_title(f"Star: {star_name} using {STAR_TARGET} for calibration")
    plt.tight_layout()

    plt.savefig(f"{STAR_TARGET}/calibration_{star_name}.png")

least_squares = pd.DataFrame(least_squares, index=[os.path.basename(cal_file) for cal_file in cal_files],
                             columns=list(star_spectra.keys()))
least_squares['total'] = least_squares[list(least_squares.columns)].sum(axis=1)
print(least_squares.sort_values('total', ascending=True))
