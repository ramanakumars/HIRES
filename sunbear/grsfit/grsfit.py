import os
import sys

sys.path.append(os.environ['BEAR_RT_PATH'])
import logging

import numpy as np

from sunbear import SUNBEAR, MCMCSolver

logger = logging.getLogger(__name__)


class GRSFit:
    def __init__(self, datafile: str, root_folder: str):
        self.datafile = datafile
        self.root_folder = root_folder

        if not os.path.exists(root_folder):
            os.makedirs(self.root_folder)

        self.data = np.load(self.datafile)

        self.albedo = self.data['albedo']
        self.error = self.data['error']
        self.wavelength = self.data['wavelength'][:, :-50]
        self.incidence = self.data['incidence']
        self.emission = self.data['emission']
        self.latitude = self.data['latitude']
        self.longitude = self.data['longitude']

    def get_spectrum(self, ind: int) -> tuple[np.ndarray, np.ndarray, float]:
        spectrum = self.albedo[ind]
        spectrum_error = self.error[ind]

        mu = np.cos(np.radians(self.incidence[ind]))
        spectrum[(spectrum <= 0.2) & (self.wavelength < 850)] = np.nan
        spectrum[spectrum > 0.8] = np.nan
        spectrum[~np.isfinite(spectrum_error)] = np.nan

        # mask out telluric regions
        spectrum[(self.wavelength > 755) & (self.wavelength < 770)] = np.nan
        spectrum[(self.wavelength > 820) & (self.wavelength < 840)] = np.nan

        return spectrum, spectrum_error, mu

    def fit_model(
        self,
        base_config: str,
        get_model: callable,
        parameters: dict,
        n_walkers: int = 48,
        n_epochs: int = 256,
        threads: int = 28,
    ):
        bear = SUNBEAR(base_config)
        fit_error = np.zeros((self.albedo.shape[0], n_walkers))

        for ind in range(self.albedo.shape[0]):
            logger.info(
                f"Running index {ind} => latitude: {self.latitude[ind]} longitude: {self.longitude[ind]}"
            )
            spectrum, error, mu = self.get_spectrum(ind)
            mask = np.isfinite(spectrum.flatten())

            bear.config['mu'] = mu
            bear.config['mu0'] = mu

            solver = MCMCSolver(
                bear, get_model, os.path.join(self.root_folder, f"run_{ind:03d}.h5")
            )
            solver.set_spectrum(
                self.wavelength.flatten()[mask] / 1000,
                spectrum.flatten()[mask],
                error.flatten()[mask],
            )
            solver.set_parameters(parameters)

            guess = np.zeros((n_walkers, len(parameters.keys())))

            for i, parameter in enumerate(parameters):
                guess[:, i] = np.random.uniform(
                    parameters[parameter]['min'],
                    parameters[parameter]['max'],
                    (n_walkers),
                )

            sampler = solver.run_mcmc(
                guess, n_epochs, n_walkers=n_walkers, threads=threads, verbose=True
            )

            fit_error[ind, :] = sampler.get_log_prob()[-1]

        return fit_error
