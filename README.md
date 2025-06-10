# HIRES

Projection functions for HIRES data

## Projection and calibration pipeline

These are the steps to project and calibrate HIRES Jupiter data.

### Extinction and flux calibration

The first step in flux calibration is to obtain the atmospheric extinction profile and calibrate against the standard stars.

#### Getting the atmospheric extinction

The atmospheric extinction is determined by fitting the stellar spectra at different airmass. Since the absolute flux calibration
is not necessary here, we will simply divide each observation by the reference star spectra to get an arbitrary calibration. Then,
we can fit this arbitrary calibration as a function of airmass to fetch the atmospheric extinction as a function of wavelength.

To do this, run the following command from the main folder:

```bash
python3 scripts/get_atmospheric_extinction.py --star_folder [path to folder with star FITS] --output_file [path to output .npz file]
```

Note that this requires the reference spectra for stars to be defined, which is done in `hiresprojection/star.py`.
Modify `stellar_spectra` as needed to change the star list.

#### Flux calibrating against the stars

With the atmospheric extinction determined, we can calibrate against the reference spectra by first determining a pre-atmospheric flux
and then dividing that by the reference spectrum. This is done per star (i.e., we will calibrate against an individual
star spectra and test against the others). We can then median combine all these observations together later on.

To do this, run:

```bash
python3 scripts/calibrate.py --star_folder [path to folder with star FITS] --opacity_table [path to extinction.npz file] --target [name of reference star]
```

Here, `star_folder` points to the folder with all the observations, `opacity_table` is the extinction file generated in the previous step
and `target` is the reference star we will use for this calibration. This command will create a folder with the star name and place
the calibration files (calibration_xxxx.npz) and also the median combined calibrated star (median_xxxx.npz) into the folder.
It will also generate a series of plots for all unique stars in the `stellar_spectra` list (see `hiresprojection/star.py`) to visually
compare the calibration. The calibration and median files are done per observation so that we can remove specific observations which have
issues.

#### Median combine the calibration

Once the calibrations are determined for all the standard stars, and noisy calibration files are removed, we can median combine the
calibrations together to create a global calibration file.

```bash
python3 scripts/combine_calibrations.py --folder [path to folder containing all calibration.npz files] --output [path to new .npz file]
```

This will create a `output.npz` file with the `calibration` key having the calibration in W/(m^2 nm)/(e-/s)
and the corresponding uncertainty in the `error` key.

### Getting the planet spectra

#### Project the guider images

First, we need to register the guider images to find the astrometric position of the slit. Since the guider camera's navigation
has fairly large errors, we need to fit the limb and calibrate the plate.

To calibrate the guider camera, assuming that the images are in `data/HIRES/GUIDE/`, run the following command:

```bash
python3 scripts/fit_guider_limbs.py --folder data/HIRES/GUIDE --plotfolder guider_images
```

The guider images' WCS will be updated in place and a plot of each guider image with the observed and fitted limb will be saved in the `guider_images` folder.

#### Navigate and project the slit

With the guider images projected, we can now find the coordinate positions of the slit for each image. We will do this by matching the observed image
with a corresponding guider image which is closest in time to the observation. To calibrate the slit run the following command:

```bash
python3 scripts/project_slit_lonlat.py --folder [path to FITS files] --navfolder [path to output folder] --guide_folder [path to calibrated GUIDE images]
```

#### Calibrate the data

Once the backplane is determined, we can convert the input spectra to I/F values as follows:

```bash
python3 scripts/get_if.py --input_folder [path to the input _nav files] --calibration_file [path to the calibration.npz] --solar_spectra [path to solar spectra FITS file] --output_folder [path to output cal/ folder]  [--overwrite]
```

This will generate a series of \_nav_cal.fits in the `output_folder` which will contain the same FITS structure as the \_nav files but with the PrimaryHDU containing I/F instead of flux.

### Utilities

#### Project guider image to map

Use the `project_guider_maps.py` script to project a folder containing navigated guider images to a cylindrical map

#### Copy header data from raw FITS

The calibrated FITS sometimes doesn't contain all the necessary header. Run the `copy_headers.py` script to copy specific header
data over (edit the code to change the required keys).
