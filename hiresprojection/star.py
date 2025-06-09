import pysynphot
from astropy import units as u
from specreduce.calibration_data import Spectrum1D, load_onedstds

R_SUN = 696_340_000  # m
LY = 9.4607e15  # light year in m


# Load the data for HR8518 and HR0804 from the Castelli-Kurucz Atlas
HR8518_spectrum = pysynphot.Icat('ck04models/', 10500, 0, 4.3)
HR8518_wave = HR8518_spectrum.wave * u.AA
HR8518_flux = (
    HR8518_spectrum.flux
    * ((2.7 * R_SUN) / (164 * LY)) ** 2.0
    * u.erg
    / (u.s * u.cm * u.cm * u.AA)
)
HR8518_mask = (HR8518_wave < 10000 * u.AA) & (HR8518_wave > 350 * u.AA)

HR0804_spectrum = pysynphot.Icat('ck04models/', 8900, 0, 4.3)
HR0804_wave = HR0804_spectrum.wave * u.AA
HR0804_flux = (
    HR0804_spectrum.flux
    * ((1.9 * R_SUN) / (80 * LY)) ** 2.0
    * u.erg
    / (u.s * u.cm * u.cm * u.AA)
)
HR0804_mask = (HR0804_wave < 10000 * u.AA) & (HR0804_wave > 350 * u.AA)

star_spectra = {
    'HR8634': load_onedstds("eso", "ctiostan/hr8634.dat"),
    'HR7950': load_onedstds("eso", "ctiostan/hr7950.dat"),
    'HR0804': Spectrum1D(
        flux=HR0804_flux[HR0804_mask], spectral_axis=HR0804_wave[HR0804_mask]
    ),
    'HR8518': Spectrum1D(
        flux=HR8518_flux[HR8518_mask], spectral_axis=HR8518_wave[HR8518_mask]
    ),
}
