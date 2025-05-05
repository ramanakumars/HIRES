import numpy as np
import copy
from einops import rearrange
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


def project_and_save(filename, outfile, slit_center=None, target='JUPITER'):
    kernels = get_kernels(KERNEL_DATAFOLDER, target)
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
        obstime = Time(header['DATE-OBS'])

        # rotate positional angle so that it is clockwise from top
        position_angle = (90 + float(header['ROTPOSN'])) * u.deg
        slit_width = slit_widths[header['DECKNAME'].strip()]
        if slit_center is None:
            slit_center = SkyCoord(f'{header["RA"]} {header["DEC"]}', frame="fk5", unit=(u.hourangle, u.deg), obstime=obstime)

        data = primaryhdu.data

        n_wave, n_pos, n_orders = data.shape

        # get the data and reshape so that it's flattened along the wavelength axis
        # i.e. combine the echelle order with the spectral axis
        data = rearrange(data, "e p w -> p (e w)")

        et = spice.utc2et(obstime.to_datetime().isoformat())

        # calculate target information
        target = target
        target_frame = 'IAU_' + target
        radii = spice.bodvar(spice.bodn2c(target), "RADII", 3)
        flattening = (radii[0] - radii[2]) / radii[0]

        half_width = slit_width.to(u.arcsec).value / 2.
        separations = np.linspace(-half_width, half_width, n_pos, endpoint=True) * u.arcsec

        points = [slit_center.directional_offset_by(position_angle, sep) for sep in separations]

        lon = np.nan * np.zeros(n_pos)
        lat = np.nan * np.zeros(n_pos)
        emis = np.nan * np.zeros(n_pos)
        inci = np.nan * np.zeros(n_pos)
        phase = np.nan * np.zeros(n_pos)

        for i, point in enumerate(points):
            veci = spice.radrec(1., point.ra.to(u.radian).value, point.dec.to(u.radian).value)

            # check for the intercept
            try:
                spoint, _, _ = spice.sincpt("Ellipsoid", target, et,
                                            target_frame, "CN+S", "KECK", "J2000", veci)
            except Exception:
                continue

            # if the intercept works, determine the planetographic
            # lat/lon values
            loni, lati, _ = spice.recpgr(target, spoint, radii[0], flattening)
            lon[i], lat[i] = np.degrees(loni), np.degrees(lati)

            _, _, phase[i], inci[i], emis[i] = spice.ilumin("ELLIPSOID", target, et, target_frame, "CN+S", "KECK", spoint)

        lonHeader = fits.Header()
        lonHeader['EXTNAME'] = "SYS_III_LONGITUDE"
        lonHDU = fits.PrimaryHDU(lon, header=lonHeader)

        latHeader = fits.Header()
        latHeader['EXTNAME'] = "GRAPHIC_LATITUDE"
        latHDU = fits.PrimaryHDU(lat, header=latHeader)

        incHeader = fits.Header()
        incHeader['EXTNAME'] = "INCIDENCE_ANGLE"
        incHDU = fits.PrimaryHDU(np.degrees(inci), header=incHeader)

        emisHeader = fits.Header()
        emisHeader['EXTNAME'] = "EMISSION_ANGLE"
        emisHDU = fits.PrimaryHDU(np.degrees(emis), header=emisHeader)

        phaseHeader = fits.Header()
        phaseHeader['EXTNAME'] = "PHASE_ANGLE"
        phaseHDU = fits.PrimaryHDU(np.degrees(phase), header=phaseHeader)

        outhdu = copy.deepcopy(hdulist)
        outhdu.append(lonHDU)
        outhdu.append(latHDU)
        outhdu.append(incHDU)
        outhdu.append(emisHDU)
        outhdu.append(phaseHDU)
        outhdu.writeto(outfile, overwrite=True)
