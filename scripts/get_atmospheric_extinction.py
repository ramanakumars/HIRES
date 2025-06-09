import argparse
import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from astropy import units as u

from hiresprojection.calibration_utils import get_sky
from hiresprojection.io_utils import (
    get_coarse_data,
    get_data_from_fits,
)
from hiresprojection.star import star_spectra

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
    description="Calibrate the HIRES spectra spectra with a reference star"
)
parser.add_argument("-star_folder", "--star_folder", type=str, required=True)
parser.add_argument("-output_file", "--output_file", type=str, required=True)
args = parser.parse_args()

# find all the files in the folder that correspond to the known stars
# star_spectra is set in calibration_utils and is just the list of stars that have known spectra
star_fits = sorted(glob.glob(os.path.join(args.star_folder, "*.fits")))

peak = 31  # np.argmax(data.mean(axis=(0, 2)))

# get the object mask (a uniform window of 6 pixels from the center)
object_mask = np.abs(np.arange(61) - peak + 1) < 6
# get the background (all pixels that are 12 pixels away from the center and 2 pixels from the edges)
background_mask = (
    (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)
)

sky_spectra = []
sky_error = []
airmass = []

colors = plt.cm.tab20(np.linspace(0, 1, 20))

# loop through the files and load the spectra
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(10, 3))
for i, file in enumerate(tqdm.tqdm(star_fits)):
    header, data, error, wavelength = get_data_from_fits(file)
    wave_coarse, data_coarse, error_coarse = get_coarse_data(file)
    target = header['targname']

    if target not in star_spectra.keys():
        continue

    airmass.append(float(header['AIRMASS']))
    spectra = star_spectra[target]
    spectra_interp = np.interp(
        wave_coarse,
        spectra.spectral_axis.to('nm').value,
        spectra.flux.to(
            'W/(m^2 nm)', equivalencies=u.spectral_density(spectra.spectral_axis)
        ).value,
    )

    sky, error = get_sky(data_coarse, error_coarse, object_mask)

    # we will scale the spectra by the corresponding star so we can now calculate the contribution from only the opacity
    sky = sky / spectra_interp
    sky_spectra.append(sky)
    sky_error.append(error / spectra_interp)

    # plot this out to check later
    for k in range(31):
        ax.fill_between(
            wave_coarse[k, :-1],
            sky[k, :-1] - error[k, :-1],
            sky[k, :-1] + error[k, :-1],
            color=colors[0],
            alpha=0.5,
        )
        if k == 0:
            ax.plot(
                wave_coarse[k, :-1],
                sky[k, :-1],
                '-',
                color=colors[0],
                linewidth=0.5,
                label=airmass[-1],
            )
        else:
            ax.plot(
                wave_coarse[k, :-1], sky[k, :-1], '-', color=colors[0], linewidth=0.5
            )
    colors = np.roll(colors, 1)
ax.legend(loc='upper left', ncols=3, fontsize=8)
plt.savefig('star_spectra_airmass.png')

sky_spectra = np.asarray(sky_spectra)
sky_error = np.asarray(sky_error)
airmass = np.asarray(airmass, dtype=float)

# now fit the airmass-extinction relation (we will assume a linear relationship)
ext_fits = np.zeros((31, 2))
ext_fits_error = np.zeros((31, 2))
for k in range(31):
    mask = airmass < 2  # np.log(sky_spectra[:, k].mean(-1)) < 3.8
    errori = np.mean(sky_error[:, k] / sky_spectra[:, k], axis=-1)
    try:
        p, V = np.polyfit(
            airmass[mask],
            np.log(sky_spectra[mask, k].mean(-1)),
            1,
            full=False,
            w=1 / errori,
            rcond=1,
            cov=True,
        )
        ext_fits[k, :] = p
        ext_fits_error[k, :] = np.diag(V)
    except Exception:
        continue

# save this out to a file. the absorption coefficient is the slope in the airmass space
np.savez(args.output_file, extinction=ext_fits[:, 0], error=ext_fits_error[:, 0])

# plot out the extinction corrected spectra
fig, ax = plt.subplots(1, 1, dpi=150, figsize=(10, 3))
colors = plt.cm.tab20(np.linspace(0, 1, 20))

for i, (file, air) in tqdm.tqdm(
    enumerate(zip(star_fits, airmass)), total=len(star_fits)
):
    header, data, error, wavelength = get_data_from_fits(file)
    wave_coarse, data_coarse, error_coarse = get_coarse_data(file)
    sky, _ = get_sky(data_coarse, error_coarse, object_mask)
    for k in range(31):
        # extinction is just exp(-tau * X)
        opacity = np.exp(-ext_fits[k, 0] * air)
        if k == 0:
            ax.plot(
                wave_coarse[k, :-1],
                sky[k, :-1] * opacity,
                '-',
                color=colors[0],
                linewidth=0.5,
                label=air,
            )
        else:
            ax.plot(
                wave_coarse[k, :-1],
                sky[k, :-1] * opacity,
                '-',
                color=colors[0],
                linewidth=0.5,
            )
    colors = np.roll(colors, 1)
ax.legend(loc='upper right', ncols=3, fontsize=8)

plt.savefig("star_spectra_extinction_corrected.png")
