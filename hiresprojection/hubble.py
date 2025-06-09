import numpy as np
import spiceypy as spice
import tqdm
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from .spice_utils import (
    BASEURL,
    check_and_download_kernels,
    fetch_kernels_from_https,
    get_kernels,
)

c_light = 3.0e5

KERNEL_DATAFOLDER = './kernels/'

KECK_LOCATION = (np.radians(19.8263), -np.radians(155.47441))


class HubbleImageProjection:
    def __init__(self, filename, target='JUPITER'):
        with fits.open(filename) as hdulist:
            self.header = hdulist[0].header
            timestart = float(self.header['EXPSTART'])
            timeend = float(self.header['EXPEND'])
            self.obstime = Time(0.5 * (timestart + timeend), format='mjd')

            self.wcs = WCS(hdulist[1].header)
            self.data = hdulist[1].data

        kernels = get_kernels(KERNEL_DATAFOLDER, 'jupiter')
        kernels.extend(
            check_and_download_kernels(
                fetch_kernels_from_https(BASEURL + "HST/kernels/spk/", "hst.bsp"),
                KERNEL_DATAFOLDER,
            )
        )
        for kernel in kernels:
            print(kernel)
            spice.furnsh(kernel)

        self.et = spice.utc2et(self.obstime.to_datetime().isoformat())

        print(spice.et2utc(self.et, "C", 2))

        # calculate target information
        self.target = target
        self.target_frame = 'IAU_' + target
        self.radii = spice.bodvar(spice.bodn2c(target), "RADII", 3)
        self.flattening = (self.radii[0] - self.radii[2]) / self.radii[0]

    def get_limb(self):
        pos, lt = spice.spkpos(self.target, self.et, 'J2000', 'CN+S', "HST")
        # pos = pos - self.keck_j2000
        dist, targRA, targDec = spice.recrad(pos)
        print(f"RA: {np.degrees(targRA):.4f} DEC: {np.degrees(targDec):.4f}")
        rolstep = np.radians(5)
        ncuts = int(2.0 * np.pi / rolstep)
        _, limbs, eplimb, vecs = spice.limbpt(
            'TANGENT/ELLIPSOID',
            self.target,
            self.et,
            self.target_frame,
            'CN+S',
            "ELLIPSOID LIMB",
            "HST",
            np.asarray([0, 0, 1]),
            rolstep,
            ncuts,
            1e-4,
            1e-7,
            ncuts,
        )

        limbJ2000 = np.zeros_like(vecs)
        limbRADec = np.zeros((ncuts, 2))
        limbdist = np.zeros(ncuts)
        for i in range(ncuts):
            # get the transformation of the limb points to the J2000 frame
            pxi = spice.pxfrm2(self.target_frame, 'J2000', eplimb[i], self.et)

            # transform the vectors from the observer to the limb to J2000
            limbJ2000[i, :] = np.matmul(pxi, vecs[i, :])

            # also convert to RA/Dec
            limbdist[i], limbRADec[i, 0], limbRADec[i, 1] = spice.recrad(
                limbJ2000[i, :]
            )

        return limbRADec

    def project_to_lonlat(self):
        ny, nx = self.data.shape

        self.lonlat = np.nan * np.zeros((*self.data.shape, 2))

        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

        radecs = self.wcs.pixel_to_world(X.flatten(), Y.flatten())

        for n, (i, j) in enumerate(
            tqdm.tqdm(zip(X.flatten(), Y.flatten()), total=X.size)
        ):
            ra = radecs[n].ra - self.raoff
            dec = radecs[n].dec - self.decoff
            veci = spice.radrec(1.0, ra.radian, dec.radian) + self.keck_j2000

            # check for the intercept
            try:
                spoint, ep, srfvec = spice.sincpt(
                    "Ellipsoid",
                    self.target,
                    self.et,
                    self.target_frame,
                    "CN",
                    "EARTH",
                    "J2000",
                    veci,
                )
            except Exception:
                continue

            # if the intercept works, determine the planetographic
            # lat/lon values
            loni, lati, alt = spice.recpgr(
                self.target, spoint, self.radii[0], self.flattening
            )

            self.lonlat[j, i, :] = np.degrees(loni), np.degrees(lati)
