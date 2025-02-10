import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import savgol_filter
from astropy import units as u
import glob
import os
from specreduce.calibration_data import load_onedstds

NWAVE_COARSE = 20

STAR_TARGET = 'HR8634'

star_spectra = {
    'HR8634': load_onedstds("eso", "ctiostan/hr8634.dat"),
    'HR7950': load_onedstds("eso", "ctiostan/hr7950.dat")
}


def get_data_from_fits(file):
    with fits.open(file) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data[:, :, :-50]
        wavelength = hdulist[3].data[:, :-50]
    return header, data, wavelength


def get_coarse_data(file):
    _, data, wavelength = get_data_from_fits(file)
    peak = 31  # np.argmax(data.mean(axis=(0, 2)))

    object_mask = np.abs(np.arange(61) - peak + 1) < 8
    background_mask = (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)

    data_coarse = np.zeros((31, NWAVE_COARSE))
    wave_coarse = np.zeros((31, NWAVE_COARSE))
    for k in range(31):
        wavei = np.linspace(wavelength[k].min(), wavelength[k].max(), NWAVE_COARSE)
        object_spectra = data[k, object_mask, :].sum(axis=0)
        sky = np.median(data[k, background_mask, :], axis=0)

        sky_subtracted = savgol_filter(object_spectra - sky, 201, 1, axis=-1)
        wave_coarse[k] = wavei
        data_coarse[k] = np.interp(wavei, wavelength[k], sky_subtracted)

    return wave_coarse, data_coarse


def get_calibration(star_fits, target, target_name):
    _, data0_coarse = get_coarse_data(target['file'])

    scaled_star_data = []
    for star in star_fits:
        if star['target'] == target_name:
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
                scaled_datai[k] = sky_subtracted * scalei_fine

            scaled_star_data.append(scaled_datai)

    scaled_star_data = np.asarray(scaled_star_data)

    median_star_data = np.median(scaled_star_data, 0)

    calibration = np.zeros_like(median_star_data)

    for k in range(31):
        wavei = wave_coarse[k]
        scalei = scaling[k]
        scalei_fine = np.interp(wavelength[k], wavei, scalei)
        scaled_datai[k] = sky_subtracted * scalei_fine
        star_interp = np.interp(wavei,
                                star_spectra[target_name].spectral_axis.to('nanometer').value,
                                star_spectra[target_name].flux.to('W/m^2/nm', equivalencies=u.spectral_density(star_spectra[target_name].spectral_axis)).value
                                )
        calibration_coarse = np.interp(wavei, wavelength[k], median_star_data[k]) / star_interp
        calibration[k] = np.interp(wavelength[k], wavei, calibration_coarse)

    return scaled_star_data, median_star_data, calibration, wavelength


all_star_fits = sorted(glob.glob('../data_reduced/2022Aug15/HIRES/spec/star/hires*.fits'))
star_fits = []
for i, star in enumerate(all_star_fits):
    with fits.open(star.replace('data_reduced/2022Aug15/HIRES/spec/star', 'data/2022Aug15/HIRES/SPEC/')) as hdulist:
        header = hdulist[0].header
    if header['targname'].strip() in star_spectra.keys():
        star_fits.append({'target': header['targname'], 'file': star})

if not os.path.exists(STAR_TARGET):
    os.makedirs(STAR_TARGET)

for i, star in enumerate(star_fits):
    if star['target'] == STAR_TARGET:
        _, _, calibration, wavelength = get_calibration(star_fits, star, STAR_TARGET)
        np.save(f'{STAR_TARGET}/calibration_HR8634_{i}.npy', calibration)

colors = plt.cm.tab10(np.linspace(0, 1, len(star_fits)))

fig, ax = plt.subplots(1, 1, figsize=(8, 4), dpi=150)

cal_files = sorted(glob.glob(f'{STAR_TARGET}/*.npy'))

colors = plt.cm.tab10(np.linspace(0, 1, len(cal_files)))

target = star_fits[6]
_, data0_coarse = get_coarse_data(target['file'])
scaled_star_data = []
for star in star_fits:
    if star['target'] == STAR_TARGET:
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
            scaled_datai[k] = sky_subtracted * scalei_fine

        scaled_star_data.append(scaled_datai)

scaled_star_data = np.asarray(scaled_star_data)
median_star_data = np.median(scaled_star_data, 0)

ax.plot(star_spectra[STAR_TARGET].spectral_axis.to('nanometer'), star_spectra[STAR_TARGET].flux.to('W/m^2/nm', equivalencies=u.spectral_density(star_spectra[STAR_TARGET].spectral_axis)), 'r-', linewidth=0.5)
for i, cal_file in enumerate(cal_files):
    calibration = np.load(cal_file)
    calibrated_star = median_star_data / calibration

    for k in range(31):
        ax.plot(wavelength[k], calibrated_star[k], '-', linewidth=0.8, color=colors[i], label=cal_file if k == 0 else None)
# ax.set_ylim((0, 1e5))
ax.legend(loc='upper right', ncols=2, fontsize=8)
plt.tight_layout()

plt.savefig(f"calibration_{STAR_TARGET}.png")
