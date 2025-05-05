# HIRES
Projection functions for HIRES data

## Projection and calibration pipeline

These are the steps to project and calibrate HIRES Jupiter data.

### Project the guider images
First, we need to register the guider images to find the astrometric position of the slit. Since the guider camera's navigation 
has fairly large errors, we need to fit the limb and calibrate the plate.

To calibrate the guider camera, assuming that the images are in `data/HIRES/GUIDE/`, run the following command:

```bash
python3 scripts/fit_guider_limbs.py --folder data/HIRES/GUIDE --plotfolder guider_images
```

The guider images' WCS will be updated in place and a plot of each guider image with the observed and fitted limb will be saved in the `guider_images` folder.

### Navigate and project the slit
With the guider images projected, we can now find the coordinate positions of the slit for each image. We will do this by matching the observed image
with a corresponding guider image which is closest in time to the observation. To calibrate the slit run the following command:

```bash
python3 scripts/project_slit_lonlat.py --folder [path to FITS files] --navfolder [path to output folder] --guide_folder [path to calibrated GUIDE images]
```

### Calibrate the data


### Utilities

#### Project guider image to map
Use the `project_guider_maps.py` script to project a folder containing navigated guider images to a cylindrical map

#### Copy header data from raw FITS
The calibrated FITS sometimes doesn't contain all the necessary header. Run the `copy_headers.py` script to copy specific header
data over (edit the code to change the required keys).
