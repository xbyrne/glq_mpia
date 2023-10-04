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

### Perform cuts in WISE data

Run the program `processed_xmatched_data.py`, which calculates important data fields (e.g. flux, flux errors) and performs cuts to the data (particularly in WISE), removing non-detections and many contaminating dwarf stars. This will generate a .csv file (`/data/processed/cut_crossmatched_objects.csv`) containing 1996 objects.

The table `data/external/known_hzqs.csv` contains data on 11 objects in the DES footprint which are known to be high-redshift quasars.

## Downloading Data
### Grab URLs
Run the program `fetch_urls.py`, which uses the coordinates in `/data/processed/cut_crossmatched_objects.csv` to find download URLs from the SIA service at [https://datalab.noirlab.edu/sia/des_dr2].
If all 5 bands are all there and there are no other problems with an object (e.g. on the boundary between tiles), the URLs are saved in `./data/external/img_url_list.txt`.
URLs for 1880 objects are here; looks like about 6% had some problem.

### Download Images
Run the script `compile_images.py`, which reads from the `img_url_list.txt` file, uses a wget command to download the fits files for each object, extracts the image data from them, and then saves the resulting data in a big 1880x28x28x5 array.
The cache may get quite large for this, and this program takes several hours to run.
The successful coadd object ids, and the corresponding images, are stored in `data/processed/ids_images_{1,2}.npz`.
[Together the file would be 104MB, which as it is bigger than 100MB would require Git LFS which I can't be bothered to work out]
These are best compiled into one npz file, using the short program `combine_img_files.py`.

## Clustering Images using Contrastive Learning

The notebook `contrastive_learning.ipynb` uses an unsupervised machine learning technique called contrastive learning to separate the objects into groups based on their imaging.
Machine learning is easiest with GPUs, and as I don't have one, I leveraged the cloud-based GPUs available for free on Google Colab.
This notebook should therefore be uploaded to Google Drive in a folder called `glq_mpia`, along with the following files:
- `glq_mpia/contrastive_utils.py` (ML nuts and bolts)
- `glq_mpia/data/processed/ids_images.npz` (Image files)
Running the notebook then trains a neural network to separate out the images in a 1024D space.
Each of the 6168 objects is assigned a point in this space, saved as a 6128x1024 array in `data/processed/encoded_imgs.npz`

Clustering and visualisation are both much easier in 2D than in 1024D.
We use t-distributed Stochastic Neighbor Embedding (t-SNE) to embed the 1024D points into a 2D space while preserving as well as possible the distances between all of the points - and hence the groupings identified by the neural network.
The short program `embed_objects.py` implements this embedding using `sklearn.manifold.TSNE`, saving the embedding to the file `data/processed/embedding.npz`.
There is a small but distinct cluster of 12 objects -- stored in `data/processed/quasar_ids.npz` -- which contains 10 known high-redshift quasars.
The final two objects turned out to be J0603--3923 and J0109--5424.


## SED fitting

The program `quasar_galaxy_fit.py` fits models of quasars, galaxies, and lensed quasars to the spectra of the 