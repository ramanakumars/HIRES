import logging

import numpy as np
from grsfit.grsfit import GRSFit

# turn off logging here so we don't get logging outputs from the emcee caller
SUNBEARlogger = logging.getLogger('sunbear')
SUNBEARlogger.setLevel(logging.ERROR)


def get_model(config, n1, n2, tau_1, tau_2, tau_3, p1, pfrac1, p2, pfrac2, p3, pfrac3):
    config['aerosols'][0]['scattering']['refractive_index'] = [1.4, -abs(10**n1)]
    config['aerosols'][1]['scattering']['refractive_index'] = [1.4, -abs(10**n2)]

    config['aerosols'][0]['tau'] = 10**tau_1
    config['aerosols'][1]['tau'] = 10**tau_2
    config['aerosols'][2]['tau'] = 10**tau_3

    # config['aerosols'][0]['hfrac'] = 10**h1
    # config['aerosols'][1]['hfrac'] = 10**h2
    # config['aerosols'][2]['hfrac'] = 10**h3

    pmax1 = 10**p1
    pmax2 = 10**p2
    pmax3 = 10**p3

    config['aerosols'][0]['pmin'] = 10**pfrac1 * pmax1
    config['aerosols'][1]['pmin'] = 10**pfrac2 * pmax2
    config['aerosols'][2]['pmin'] = 10**pfrac3 * pmax3

    config['aerosols'][0]['pmax'] = pmax1
    config['aerosols'][1]['pmax'] = pmax2
    config['aerosols'][2]['pmax'] = pmax3

    config['composition'][3]['val'] = 2.61e-4

    config['composition'][4]['parameters']['xnh3_strat'] = 1.059e-07  # 10**xNH3_strat
    config['composition'][4]['parameters']['xnh3_trop'] = 1.077e-07
    config['composition'][4]['parameters']['xnh3_deep'] = 8.931e-05  # 6.733e-05

    return config


parameters = {
    'n1': {'min': -4, 'max': -1},
    'n2': {'min': -4, 'max': -1},
    'tau_1': {'min': -3, 'max': 2},
    'tau_2': {'min': -3, 'max': 2},
    'tau_3': {'min': -3, 'max': 3},
    'p1': {'min': -1.3, 'max': -1},
    'pfrac1': {'min': -1, 'max': 0},
    'p2': {'min': -1, 'max': 0},
    'pfrac2': {'min': -1, 'max': 0},
    'p3': {'min': 0, 'max': 1},
    'pfrac3': {'min': -1, 'max': 0},
}


fitter = GRSFit('GRS_spectra.npz', 'fits/')

fit_error = fitter.fit_model('jupiter_test.json', get_model, parameters)

np.save("fit_error.npy", fit_error)
