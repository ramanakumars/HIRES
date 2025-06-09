from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits
from einops import rearrange


@dataclass
class EchelleOrder:
    raw_data: np.ndarray
    raw_wavelength: np.ndarray
    order: int
    trim: int = 150
    data: np.ndarray = field(init=False)
    wavelength: np.ndarray = field(init=False)

    def __post_init__(self):
        self.data = self.raw_data[:, : -self.trim]
        self.wavelength = self.raw_wavelength[: -self.trim]


class HIRESSpectra:
    data: list[EchelleOrder]

    def __init__(self, fname, trim=60):
        with fits.open(fname) as hdulist:
            self.n_echelle, self.n_slit, self.n_wavelength = hdulist[0].data.shape
            self.orders = [
                EchelleOrder(data, wavelength, order, trim)
                for order, (data, wavelength) in enumerate(
                    zip(hdulist[0].data, hdulist[3].data)
                )
            ]

    @property
    def spectrum(self):
        return rearrange(
            np.asarray([order.data / order.calibration for order in self.orders]),
            "o s w -> s (o w)",
        )

    @property
    def wavelength(self):
        return np.ravel([order.wavelength for order in self.orders])
