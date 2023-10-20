# glq_mpia
## Searching for Gravitationally-Lensed High-Redshift Quasars using Unsupervised Machine Learning

This is code and files for a Summer Internship I did at MPIA with Romain Meyer, in search of gravitationally lensed quasars.

This repo is for reproducibility's sake (mostly by myself!)

The project proceeds in the following order:

## Selecting Data
### Select Quasi-Quasar Objects from the Dark Energy Survey
Run the SQL query in `data/loose_cuts.txt` at [https://des.ncsa.illinois.edu/desaccess/db-access] (login required).
These selection criteria are similar to those often used to select high-redshift quasars, but are somewhat looser in order not to exclude lensed quasars.
The result is a ~32MB .csv file, containing data on 218 241 objects; this file is `data/external/cut_des.csv`

### Cross-match to the WISE and VHS Surveys
Install and open [TOPCAT](https://www.star.bris.ac.uk/~mbt/topcat/), and upload the above .csv file.

The CDS X-match service is accessed by pressing an X-shaped button on the main taskbar.
For the Remote Table, in the "VizieR Table ID/Alias" text box, select AllWISE.
For the Local Table, select the csv file just uploaded.
The RA and DEC columns should be automatically identified.
Under Match Parameters, choose Radius: 1.0 arcsec; Find mode: Best; Rename columns: Duplicates, Suffix: _x; Block size: 50000.
Hit Go.
A new table, for me called "1xAllWISE", is created after about a minute, with 145 567 entries.

We then crossmatch *this* table to VHS.
Again, open the CDS X-match service; select VHS DR5 as the Remote Table.
Select the result of the AllWISE crossmatch as the Local Table, and choose RA and DEC as the RA and Dec columns; the default choice may be different now.
Use the same Match Parameters as for the first X-match.
Hit Go.
A third table, for me called "2xVHS DR5", is created after another minute, containing 116 499 entries.
This table is `/data/interim/des_wise_vhs_objects.csv`.

The table `/data/external/all_hzqs.csv` contains data on 406 known high-redshift quasars.
Cross-matching to the `/data/interim/des_wise_vhs_objects.csv` file just created yields `/data/external/processed/known_hzqs.csv`, which contains 12 objects which are thus in the DESxVHSxWISE footprint.

### Perform cuts in WISE data

Run the program `processed_xmatched_data.py`, which calculates important data fields (e.g. flux, flux errors) and performs cuts to the data (particularly in WISE), removing non-detections ($2\sigma$) and many contaminating dwarf stars. This will generate a .csv file (`/data/processed/cut_crossmatched_objects.csv`) containing 7438 objects.

## Downloading Data
### Grab URLs
Run the program `fetch_urls.py`, which uses the coordinates in `/data/processed/cut_crossmatched_objects.csv` to find download URLs from the SIA service at [https://datalab.noirlab.edu/sia/des_dr2].
If all 5 bands are all there and there are no other problems with an object (e.g. on the boundary between tiles), the URLs are saved in `./data/external/img_url_list.txt`.
This will probably take a couple of hours.
URLs for 7018 objects are here; looks like about 6% had some problem.

### Download Images
Run the script `compile_images.py`, which reads from the `img_url_list.txt` file, uses a wget command to download the fits files for each object, extracts the image data from them, and then saves the resulting data in a big 7016x28x28x5 array (two objects seem to have had their images unavailable somehow).
The cache may get quite large for this, and this program takes a day or so to run.
The successful coadd object ids are stored in `data/processed/ids.npz`, and the corresponding images are stored in `data/processed/imgs.npz`.
This file is 117MB, which is too large for Github so they can be split across two files using the `/data/processed/split_imgs.py` into the two files `data/processed/imgs{1,2}.npz`.
The `.npz` files can be combined with `data/processed/assemble_imgs.py`.

## Clustering Images using Contrastive Learning

The notebook `contrastive_learning.ipynb` uses an unsupervised machine learning technique called contrastive learning to separate the objects into groups based on their imaging.
Machine learning is easiest with GPUs, and if you don't have one then the cloud-based GPUs available for free on Google Colab.
If not using Colab, the notebook can be run as-is.
If using Colab, this notebook should therefore be uploaded to Google Drive in a folder called `glq_mpia`, along with the following files:
- `glq_mpia/contrastive_utils.py` (ML nuts and bolts)
- `glq_mpia/data/processed/{ids, imgs}.npz` (Image files with corresponding IDs)
Running the notebook then trains a neural network to separate out the images in a 512 space.
Each of the 6536 objects is assigned a point in this space, saved as a 6536x512 array in `data/processed/encoded_imgs.npz`

Clustering and visualisation are both much easier in 2D than in 512.
We use t-distributed Stochastic Neighbor Embedding (t-SNE) to embed the 512 points into a 2D space while preserving as well as possible the distances between all of the points - and hence the groupings identified by the neural network.
The end of the notebook `contrastive_learning.ipynb` carries out this embedding, saving the result to the file `data/processed/embedding.npz`.
There is a small but distinct cluster of 12 objects -- stored in `data/processed/quasar_ids.npz` -- which contains 
#10 known high-redshift quasars.
#The final two objects turned out to be J0603--3923 and J0109--5424.


## SED fitting

The program `quasar_galaxy_fit.py` fits models of quasars, galaxies, and lensed quasars to the photometry of the objects in the group identified by the neural network