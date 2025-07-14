import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter

NWAVE_COARSE = 50


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
    data[np.isnan(data)] = 0.0
    uncertainty[np.isnan(data)] = 0.0
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

    data = savgol_filter(data, 201, 1, axis=-1)
    uncertainty = savgol_filter(uncertainty, 201, 1, axis=-1)

    data_coarse = np.zeros((31, 61, NWAVE_COARSE))
    unc_coarse = np.zeros((31, 61, NWAVE_COARSE))
    wave_coarse = np.zeros((31, NWAVE_COARSE))

    for k in range(31):
        wavei = np.linspace(wavelength[k].min(), wavelength[k].max(), NWAVE_COARSE)
        wave_coarse[k] = wavei
        for j in range(61):
            data_coarse[k, j] = np.interp(wavei, wavelength[k], data[k, j])
            unc_coarse[k, j] = np.interp(wavei, wavelength[k], uncertainty[k, j])

    return wave_coarse, data_coarse, unc_coarse
