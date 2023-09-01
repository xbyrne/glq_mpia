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
For the Remote Table, in the "VizieR Table ID/Alias" dropdown, select UnWISE.
For the Local Table, select the csv file just uploaded.
The RA and DEC columns should be automatically identified.
Under Match Parameters, choose Radius: 1.0 arcsec; Find mode: Best; Rename columns: Duplicates, Suffix: _x; Block size: 50000.
Hit Go.
A new table, for me called "1xUnWISE", is created after about a minute, with 190 049 entries.

We then crossmatch *this* table to VHS.
Again, open the CDS X-match service; select VHS DR5 as the Remote Table.
Select the result of the UnWISE crossmatch as the Local Table, and choose RA and DEC as the RA and Dec columns; the default choice may be different now.
Use the same Match Parameters as for the first X-match.
Hit Go.
A third table, for me called "2xVHS DR5", is created after another minute, containing 151 629 entries.
This table is `/data/interim/des_wise_vhs_objects.csv`.
Said file is 109MB which is too large for Git and I can't be bothered to figure out LFS so it's stored compressed as a `.csv.gz` file.

### Perform cuts in WISE data

Run the program `processed_xmatched_data.py`, which calculates important data fields (e.g. flux, flux errors) and performs cuts to the data (particularly in WISE), removing non-detections and many contaminating dwarf stars. This will generate a .csv file (`/data/processed/cut_crossmatched_objects.csv`) containing 6566 objects.

The table `data/external/known_hzqs.csv` contains data on 11 objects in this table which are known to be high-redshift quasars.

## Downloading Data
### Grab URLs
Run the program `fetch_urls.py`, which uses the coordinates in `/data/processed/cut_crossmatched_objects.csv` to find download URLs from the SIA service at [https://datalab.noirlab.edu/sia/des_dr2].
If all 5 bands are all there and there are no other problems with an object (e.g. on the boundary between tiles), the URLs are saved in `./data/external/img_url_list.txt`.
URLs for 6171 objects are here; looks like about 6% had some problem.

### Download Images
Run the script `compile_images.py`, which reads from the `img_url_list.txt` file, uses a wget command to download the fits files for each object, extracts the image data from them, and then saves the resulting data in a big 6168x28x28x5 array, (3 objects failed to completely download).
The cache may get quite large for this, and this program takes about a day to run.
The successful coadd object ids, and the corresponding images, are stored in `data/processed/ids_images_{1,2}.npz`.
[Together the file would be 104MB, which as it is bigger than 100MB would require Git LFS which I can't be bothered to work out]
These are best compiled into one npz file, using the short program `combine_img_files.py`.

## Clustering Images using Contrastive Learning

Run the program `contrastive_learning.py`, which, using the functions defined in `contrastive_utils.py`, trains a neural network using contrastive learning, to separate the images into similar-looking groups.
The UMAP procedure is then used to aid separation in 2D.
The UMAP embedding is saved in `/data/processed/umap_embedding.npz`


<!-- The notebook `contrastive_learning/trainer.ipynb` trains separates out the remaining objects into clusters.##
Due to the intensive tensorflow calculations required, it is only tractable to run this on a GPU.
I only have access to one via Google Colab, so the notebook is written for that (if you are using a local GPU, you'll have to tweak things a bit).
On Google Drive, upload both the images.npz file (which is quite large so may take time) and objs_7102.csv.
The result of this notebook is a UMAP embedding - a list of points in 2D space that are representative of the separations of the images in the 1024-dimensional output of the encoder of the neural network.
This embedding is stored in `embedding.npz`, along with the respective IDs.
The notebook also gives a list of 12 IDs of objects on a particular island, which contains 8 known high-redshift quasars! -->

## SED Fitting
We now have a set of interesting objects and their photometry.
We now attempt to fit their photometry to SED models for stars, galaxies, quasars, and of course lensed quasars (i.e. a galaxy + a quasar). 
This is done using two different pieces of software: LePHARE (downloaded from [https://www.cfht.hawaii.edu/~arnouts/LEPHARE/lephare.html]) and BAGPIPES (a Python package that can be pip'd).
