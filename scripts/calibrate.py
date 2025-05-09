import glob
import os
import logging
import argparse
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
import pandas as pd
from calibration_utils import star_spectra, get_data_from_fits, get_calibration, get_coarse_data, subtract_sky

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Calibrate the HIRES spectra spectra with a reference star")
parser.add_argument("-star_folder", "--star_folder", type=pathlib.Path, required=True)
parser.add_argument("-target", "--target", type=str, required=True, default="HR8634")
args = parser.parse_args()

STAR_TARGET = args.target

# find all the files in the folder that correspond to the known stars
# star_spectra is set in calibration_utils and is just the list of stars that have known spectra
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
    # for the given star target get the calibration across all files of this target
    if star['target'] == STAR_TARGET:
        scaled_data, median_star_data, median_star_error, calibration, calibration_error, wavelength = get_calibration(star_fits, star, STAR_TARGET)
        np.savez(f'{STAR_TARGET}/median_{STAR_TARGET}_{i}.npz', median=median_star_data, error=median_star_error, scaled_data=scaled_data)
        np.savez(f'{STAR_TARGET}/calibration_{STAR_TARGET}_{i}.npz', calibration=calibration, error=calibration_error)

cal_files = sorted(glob.glob(f'{STAR_TARGET}/calibration*.npz'))
colors = plt.cm.tab10(np.linspace(0, 1, len(cal_files)))

least_squares = np.zeros((len(cal_files), len(star_spectra)))

# now we loop through all the files and check and plot the calibration
for j, (star_name, star_spectrum) in enumerate(star_spectra.items()):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)

    try:
        target = [star for star in star_fits if star['target'] == star_name][0]
    except IndexError:
        print(f"No observations found for {star_name}")
        break

    # first get the normalization target data
    _, _, unc0_fine, _ = get_data_from_fits(target['file'])
    _, data0_coarse, unc0_coarse = get_coarse_data(target['file'])

    scaled_star_data = []
    scaled_star_error = []
    
    peak = 31  # np.argmax(data.mean(axis=(0, 2)))

    # get the object mask (a uniform window of 8 pixels from the center)
    object_mask = np.abs(np.arange(61) - peak + 1) < 8
    # get the background (all pixels that are 12 pixels away from the center and 2 pixels from the edges)
    background_mask = (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)

    # for each star observation, load the data and find the normalization between this
    # file and the target
    for star in star_fits:
        if star['target'] == star_name:
            file = star['file']

            _, data, error, wavelength = get_data_from_fits(file)

            # get the coarsened data for calibration
            wave_coarse, data_coarse, unc_coarse = get_coarse_data(file)

            scaling = data_coarse / data0_coarse

            scaled_datai = np.zeros_like(data[:, 0])
            scaled_errori = np.zeros_like(data[:, 0])
            sky_subtracted, sky_error = subtract_sky(data, error, object_mask, background_mask)
            
            for k in range(31):
                wavei = wave_coarse[k]
                scalei = scaling[k]
                scalei_fine = np.interp(wavelength[k], wavei, scalei)
                scaled_datai[k] = sky_subtracted[k] / scalei_fine
                scaled_errori[k] = sky_error[k] / scalei_fine

            scaled_star_data.append(scaled_datai)
            scaled_star_error.append(scaled_errori)

    scaled_star_data = np.asarray(scaled_star_data)
    scaled_star_error = np.asarray(scaled_star_error)

    scaled_star_data = np.asarray(scaled_star_data)
    median_star_data = np.median(scaled_star_data, 0)

    star_wave = star_spectra[star_name].spectral_axis.to('nanometer')
    star_flux = star_spectra[star_name].flux.to('W/(m^2 nm)', equivalencies=u.spectral_density(star_spectra[star_name].spectral_axis))
    ax.plot(star_wave, star_flux, 'r-', linewidth=0.5)
    for i, cal_file in enumerate(cal_files):
        calibration_data = np.load(cal_file)
        calibration = calibration_data['calibration']
        calibration_error = calibration_data['error']
        calibrated_star = median_star_data / calibration
        calibrated_star_error = calibration_error / calibration * calibrated_star
        calibrated_star[~np.isfinite(calibrated_star)] = 0.
        calibrated_star_error[~np.isfinite(calibrated_star_error)] = 0.

        least_squares_i = 0.
        for k in range(31):
            ax.fill_between(wavelength[k], calibrated_star[k] + calibrated_star_error[k], calibrated_star[k] - calibrated_star_error[k], color=colors[i], alpha=0.1)
            ax.plot(wavelength[k], calibrated_star[k], '-', linewidth=0.8, color=colors[i], label=cal_file if k == 0 else None)
            star_flux_interp = np.interp(wavelength[k], star_wave.value, star_flux.value)
            least_squares_i += np.sum((calibrated_star[k] - star_flux_interp)**2. + calibrated_star_error[k] ** 2.)
        least_squares[i, j] = least_squares_i / wavelength.size

    # ax.set_ylim((0, 1e5))
    ax.legend(loc='upper right', ncols=2, fontsize=8)
    ax.set_ylabel(r"Flux [W/m$^2$/nm]")
    ax.set_xlabel(r"Wavelength [nm]")
    ax.set_ylim((0, 1.2 * star_flux.value.max()))
    ax.set_title(f"Star: {star_name} using {STAR_TARGET} for calibration")
    plt.tight_layout()

    plt.savefig(f"{STAR_TARGET}/calibration_{star_name}.png")

least_squares = pd.DataFrame(least_squares, index=[os.path.basename(cal_file) for cal_file in cal_files],
                             columns=list(star_spectra.keys()))
least_squares['total'] = least_squares[list(least_squares.columns)].sum(axis=1)
print(least_squares.sort_values('total', ascending=True))
