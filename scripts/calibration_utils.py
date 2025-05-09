import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
from astropy import units as u
from specreduce.calibration_data import load_onedstds

NWAVE_COARSE = 20

star_spectra = {
    'HR8634': load_onedstds("eso", "ctiostan/hr8634.dat"),
    'HR7950': load_onedstds("eso", "ctiostan/hr7950.dat")
}


def get_data_from_fits(file: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Retrieve data from the HIRES FITS file (header, spectra, wavelength)

    :param file: path to the FITS file

    :returns: - header information
              - spectra (shape: n_slit, n_wavelengths, n_echelle_order)
              - wavelength (shape: n_wavelengths, n_echelle_order)
    """
    with fits.open(file) as hdulist:
        header = hdulist[0].header
        data = hdulist[0].data[:, :, :-50]
        uncertainty = hdulist[1].data[:, :, :-50]
        wavelength = hdulist[3].data[:, :-50]
    data[np.isnan(data)] = 0.
    return header, data, uncertainty, wavelength


def get_coarse_data(file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve data from a FITS file and get the coarse-resolution, sky-subtracted, savgol-filtered spectrum

    :param file: path to the FITS file

    :returns: - wavelength (shape: NWAVE_COARSE, n_echelle_order)
              - spectra (shape: n_slit, NWAVE_COARSE, n_echelle_order)
              - uncertainty (shape: n_slit, NWAVE_COARSE, n_echelle_order)
    """
    _, data, uncertainty, wavelength = get_data_from_fits(file)
    peak = 31  # np.argmax(data.mean(axis=(0, 2)))

    object_mask = np.abs(np.arange(61) - peak + 1) < 8
    background_mask = (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)

    data_coarse = np.zeros((31, NWAVE_COARSE))
    unc_coarse = np.zeros((31, NWAVE_COARSE))
    wave_coarse = np.zeros((31, NWAVE_COARSE))

    sky_subtracted, sky_subtracted_error = subtract_sky(data, uncertainty, object_mask, background_mask)

    for k in range(31):
        wavei = np.linspace(wavelength[k].min(), wavelength[k].max(), NWAVE_COARSE)
        wave_coarse[k] = wavei
        data_coarse[k] = np.interp(wavei, wavelength[k], sky_subtracted[k])
        unc_coarse[k] = np.interp(wavei, wavelength[k], sky_subtracted_error[k], 0)

    return wave_coarse, data_coarse, unc_coarse


def get_standard_error_of_mean(data: np.ndarray, error: np.ndarray, axis: int=0) -> np.ndarray:
    """
    Standard error in the mean of the measurements. This is just propogating errors from the idea of mean:
       <x> = (x_1 + x_2 + .... ) / N
    so
       d(<x>) = sqrt(d(x_1)^2 + d(x_2)^2 + .... )) / N

    :param error: the error of the individual measurements
    :param axis: the axis along which to apply the mean

    :return: the error in the mean
    """
    mean = np.mean(data, axis=axis)
    cov = np.sum((data - mean) ** 2., axis=0) / data.shape[axis]
    return np.sqrt(np.sum(error ** 2., axis=0) + cov) / error.shape[axis]

def subtract_sky(data: np.ndarray, error: np.ndarray, object_mask: np.ndarray, background_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sky_subtracted = np.zeros_like(data[:, 0])
    sky_subtracted_error = np.zeros_like(data[:, 0])
    # loop over all echelle orders, subtract the background and coarsen the data
    for k in range(31):
        object_spectra = data[k, object_mask, :]
        sky = np.median(data[k, background_mask, :], axis=0)
        sky_error = get_standard_error_of_mean(data[k, background_mask, :], error[k, background_mask, :]) * 1.253
        sky_subtracted[k] = savgol_filter((object_spectra - sky).sum(0), 201, 1, axis=-1)
        sky_subtracted_error[k] = np.sqrt(np.sum(error[k, object_mask, :] ** 2., axis=0) + sky_error ** 2.)

    return sky_subtracted, sky_subtracted_error


def get_calibration(star_fits: str, target: dict, target_name: str) -> tuple[np.ndarray]:
    """
    Get the calibration of an observed star from a FITS file using a corresponding ground truth

    :param star_fits: the path to the FITS file
    :param target: the target FITS file to normalize against
    :param target_name: the name of the calibration star

    :returns:
        - normalized star data
        - the median star data across all observations
        - the calibration (ratio of observation to the ground truth)
        - wavelength of the observations
    """
    peak = 31  # np.argmax(data.mean(axis=(0, 2)))

    # get the object mask (a uniform window of 8 pixels from the center)
    object_mask = np.abs(np.arange(61) - peak + 1) < 8
    # get the background (all pixels that are 12 pixels away from the center and 2 pixels from the edges)
    background_mask = (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)

    # first get the normalization target data
    _, _, unc0_fine, _ = get_data_from_fits(target['file'])
    _, data0_coarse, unc0_coarse = get_coarse_data(target['file'])

    scaled_star_data = []
    scaled_star_error = []

    # for each star observation, load the data and find the normalization between this
    # file and the target
    for star in star_fits:
        if star['target'] == target_name:
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
                scaled_errori[k] = np.sqrt(np.sum(sky_error[k] ** 2., axis=0)) / scalei_fine

            scaled_star_data.append(scaled_datai)
            scaled_star_error.append(scaled_errori)

    scaled_star_data = np.asarray(scaled_star_data)
    scaled_star_error = np.asarray(scaled_star_error)
    median_star_data = np.median(scaled_star_data, 0)

    # error in the median data is 1.253 * std as before
    median_star_error = get_standard_error_of_mean(scaled_star_data, scaled_star_error) * 1.253
    calibration = np.zeros_like(median_star_data)
    calibration_error = np.zeros_like(median_star_data)

    # now loop over all echelle orders and find the calibration parameters
    for k in range(31):
        wavei = wave_coarse[k]
        star_interp = np.interp(wavei,
                                star_spectra[target_name].spectral_axis.to('nanometer').value,
                                star_spectra[target_name].flux.to('W/(m^2 nm)', equivalencies=u.spectral_density(star_spectra[target_name].spectral_axis)).value
                                )
        calibration_coarse = np.interp(wavei, wavelength[k], median_star_data[k]) / star_interp
        calibration[k] = np.interp(wavelength[k], wavei, calibration_coarse)
        calibration_error[k] = np.interp(wavelength[k], wavei, np.interp(wavei, wavelength[k], median_star_error[k]) / star_interp)

    return scaled_star_data, median_star_data, median_star_error, calibration, calibration_error, wavelength
