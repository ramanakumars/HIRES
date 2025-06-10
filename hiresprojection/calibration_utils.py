import numpy as np
from astropy import units as u
from scipy.signal import savgol_filter

from .io_utils import get_coarse_data, get_data_from_fits
from .star import star_spectra


def get_standard_error_of_mean(
    data: np.ndarray, error: np.ndarray, axis: int = 0
) -> np.ndarray:
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
    dmean = np.sqrt(np.sum(error**2.0, axis=axis)) / (data.shape[axis])

    # add the covariance
    cov = np.sum((data - mean) ** 2.0, axis=axis) / data.shape[axis]
    return  np.sqrt(dmean ** 2. + cov)


def get_sky(
    data: np.ndarray, error: np.ndarray, background_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the sky background spectra and error for a given observation. `background_mask` defines the positions
    along the slit which defines the sky, which is median combined to give the sky spectra.

    :param data: input spectra (shape: [nechelle, nslit, nwavelength])
    :param error: the error in the spectra (same shape as data)
    :param background_mask: a mask of the slit positions which correspond to the sky (shape: nslit)

    :returns:
        - the sky background spectra (shape: [nechelle, nwavelength])
        - the error in the sky spectra (same shape as the spectra).
    """
    sky = np.zeros_like(data[:, 0])
    sky_error = np.zeros_like(data[:, 0])
    # loop over all echelle orders, subtract the background and coarsen the data
    for k in range(31):
        sky[k] = np.median(data[k, background_mask, :], axis=0)
        sky_error[k] = (
            get_standard_error_of_mean(
                data[k, background_mask, :], error[k, background_mask, :]
            )
            * 1.253
        )

    return sky, sky_error


def subtract_sky(
    data: np.ndarray,
    error: np.ndarray,
    object_mask: np.ndarray,
    background_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the sky-subtracted spectra and error for a given observation assuming a point-source.
    `background_mask` defines the position along the slit which defines the sky, which is
    median combined to give the sky spectra. The `object_mask` defines the positions along
    the slit which contain the point source. The final spectra is summed along `object_mask`

    :param data: input spectra (shape: [nechelle, nslit, nwavelength])
    :param error: the error in the spectra (same shape as data)
    :param object_mask: a mask of the slit positions which correspond to the object (shape: nslit)
    :param background_mask: a mask of the slit positions which correspond to the sky (shape: nslit)

    :returns:
        - the sky-subtracted spectra (shape: [nechelle, nwavelength])
        - the error in the sky spectra (same shape as the spectra).
    """
    sky_subtracted = np.zeros_like(data[:, 0])
    sky_subtracted_error = np.zeros_like(data[:, 0])
    sky, sky_error = get_sky(data, error, background_mask)

    # loop over all echelle orders, subtract the background and coarsen the data
    for k in range(31):
        object_spectra = data[k, object_mask, :]
        sky_subtracted[k] = savgol_filter(
            (object_spectra - sky[k]).sum(0), 201, 1, axis=-1
        )
        sky_subtracted_error[k] = np.sqrt(
            np.sum(error[k, object_mask, :] ** 2.0, axis=0) + sky_error[k] ** 2.0
        )

    return sky_subtracted, sky_subtracted_error


def get_calibration(
    star_fits: str, target: dict, target_name: str, extinction: np.ndarray
) -> tuple[np.ndarray]:
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

    # first get the normalization target data
    header, data0, unc0_fine, wavelength = get_data_from_fits(target['file'])

    peak = np.argmax(data0.mean(axis=(0, 2)))

    # get the object mask (a uniform window of 8 pixels from the center)
    object_mask = np.abs(np.arange(61) - peak + 1) < 8
    # get the background (all pixels that are 15 pixels away from the center and 2 pixels from the edges)
    background_mask = (
        (np.abs(np.arange(61) - peak) > 12) & (np.arange(61) > 2) & (np.arange(61) < 59)
    )

    wave_coarse, data0_coarse, unc0_coarse = get_coarse_data(target['file'])
    sky0, _ = get_sky(
        savgol_filter(data0, 201, 1, axis=-1),
        savgol_filter(unc0_fine, 201, 1, axis=-1),
        background_mask,
    )

    scaled_star_data = []
    scaled_star_error = []

    # for each star observation, load the data and find the normalization between this
    # file and the target
    for star in star_fits:
        if star['target'] == target_name:
            file = star['file']

            header, data, error, wavelength = get_data_from_fits(file)
            airmass = float(header['AIRMASS'])
            wave_coarse, data_coarse, error_coarse = get_coarse_data(file)

            scaled_datai = np.zeros_like(data[:, 0])
            scaled_errori = np.zeros_like(data[:, 0])
            sky_subtracted, sky_error = subtract_sky(
                data, error, object_mask, background_mask
            )
            opacity = np.exp(-extinction['extinction'] * airmass)

            # d(exp(x)) = exp(x) dx
            opacity_error = opacity * extinction['error']

            for k in range(31):
                wavei = wave_coarse[k]
                scaled_datai[k] = sky_subtracted[k] * opacity[k]
                scaled_errori[k] = (
                    np.sqrt(
                        (sky_error[k] / sky_subtracted[k]) ** 2
                        + (opacity_error[k] / opacity[k]) ** 2
                    )
                    * scaled_datai[k]
                )

            scaled_star_data.append(scaled_datai)
            scaled_star_error.append(scaled_errori)

    scaled_star_data = np.asarray(scaled_star_data)
    scaled_star_error = np.asarray(scaled_star_error)
    median_star_data = np.median(scaled_star_data, 0)

    # error in the median data is 1.253 * std as before
    median_star_error = (
        get_standard_error_of_mean(scaled_star_data, scaled_star_error) * 1.253
    )
    calibration = np.zeros_like(median_star_data)
    calibration_error = np.zeros_like(median_star_data)

    # now loop over all echelle orders and find the calibration parameters
    for k in range(31):
        wavei = wave_coarse[k]
        star_interp = np.interp(
            wavei,
            star_spectra[target_name].spectral_axis.to('nanometer').value,
            star_spectra[target_name]
            .flux.to(
                'W/(m^2 nm)',
                equivalencies=u.spectral_density(
                    star_spectra[target_name].spectral_axis
                ),
            )
            .value,
        )
        calibration_coarse = (
            np.interp(wavei, wavelength[k], median_star_data[k]) / star_interp
        )
        calibration[k] = np.interp(wavelength[k], wavei, calibration_coarse)
        calibration_error[k] = np.interp(
            wavelength[k],
            wavei,
            np.interp(wavei, wavelength[k], median_star_error[k]) / star_interp,
        )

    return (
        scaled_star_data,
        median_star_data,
        median_star_error,
        calibration,
        calibration_error,
        wavelength,
    )
