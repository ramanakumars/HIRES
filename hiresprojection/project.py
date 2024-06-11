import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from .spice_utils import get_kernels
import spiceypy as spice
import os

slit_widths = {
    'D1': 14 * u.arcsec
}

KERNEL_DATAFOLDER = './kernels/'


class SpectralProjection:
    def __init__(self, filename, target='JUPITER'):
        kernels = get_kernels(KERNEL_DATAFOLDER, 'jupiter')
        for kernel in kernels:
            try:
                spice.kinfo(kernel)
            except spice.NotFoundError:
                spice.furnsh(kernel)

        kernels.append(os.path.join(KERNEL_DATAFOLDER, "keck.bsp"))
        for kernel in kernels:
            spice.furnsh(kernel)

        spice.boddef("KECK", 1001)

        with fits.open(filename) as hdulist:
            primaryhdu = hdulist[0]
            header = primaryhdu.header
            self.obstime = Time(header['DATE-OBS'])

            # rotate positional angle so that it is clockwise from top
            self.position_angle = (270 + float(header['ROTPOSN'])) * u.deg
            self.slit_width = slit_widths[header['DECKNAME'].strip()]
            self.slit_center = SkyCoord(f'{header["RA"]} {header["DEC"]}', frame="fk5", unit=(u.hourangle, u.deg), obstime=self.obstime)

            self.data = primaryhdu.data

            self.n_wavelengths, self.n_pos, self.n_orders = self.data.shape

            self.et = spice.utc2et(self.obstime.to_datetime().isoformat())

            # calculate target information
            self.target = target
            self.target_frame = 'IAU_' + target
            self.radii = spice.bodvar(spice.bodn2c(target), "RADII", 3)
            self.flattening = (self.radii[0] - self.radii[2]) / self.radii[0]

    def project_to_lonlat(self):
        half_width = self.slit_width.to(u.arcsec).value / 2.
        separations = np.linspace(-half_width, half_width, self.n_pos, endpoint=True) * u.arcsec

        points = [self.slit_center.directional_offset_by(self.position_angle, sep) for sep in separations]

        self.lonlat = np.zeros((self.n_pos, 2))

        for i, point in enumerate(points):
            veci = spice.radrec(1., point.ra.to(u.radian).value, point.dec.to(u.radian).value)

            # check for the intercept
            try:
                spoint, ep, srfvec = spice.sincpt("Ellipsoid", self.target, self.et,
                                                  self.target_frame, "CN+S", "KECK", "J2000", veci)
            except Exception:
                continue

            # if the intercept works, determine the planetographic
            # lat/lon values
            loni, lati, alt = spice.recpgr(self.target, spoint, self.radii[0], self.flattening)

            self.lonlat[i, :] = np.degrees(loni), np.degrees(lati)
