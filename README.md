# glq_mpia
## Searching for Gravitationally-Lensed High-Redshift Quasars using Unsupervised Machine Learning

This is code for a Summer Internship I did at MPIA with Romain Meyer, in search of gravitationally lensed quasars.

This repo is for reproducibility's sake (mostly by myself!)

The project proceeds in the following order:

## Selecting Data
### Select Quasi-Quasar Objects from the Dark Energy Survey
Run the SQL query in selecting_data/loose_cuts.txt at https://des.ncsa.illinois.edu/desaccess/db-access (login required). These selection criteria are similar to those often used to select high-redshift quasars, but are somewhat looser in order not to exclude lensed quasars.
The result is a ~33MB .csv file, containing data on 218241 objects; this file is `selecting_data/objs_2e5.csv`
### Cross-match to the WISE Survey
Visit https://datalab.noirlab.edu/xmatch.php. Under the "Table Management" tab, upload the above .csv file. After a few minutes, it will be uploaded and will then appear in the first drop-down menu on the "Xmatch" tab. Select this file, then select the ra and dec columns in the next dropdowns. In the "select output columns", select All.
From the 2nd table dropdowns, select "unwise_dr1", "unwise_dr1.object" "ra" and "dec". In the "select output columns" dropdown, select 
[coadd_id, dec, dfluxlbs_w1, dfluxlbs_w2, flux_w1, flux_w2, fluxlbs_w1, fluxlbs_w2, mag_w1_vg, mag_w2_vg, ra]. The rest of the columns will not be needed and make the file unnecessarily large.
Choose a radius of 1 arcsecond, and select "Nearest neighbor", "Exclude non-matching rows" and "Download results to your computer only". The result will be a ~54MB .txt file (best converted to .csv for easy access) with now only 190049 objects; this file is `selecting_data/objs_2e5_x_WISE.csv`.
### Perform cuts in WISE data
Run through the notebook `selecting_data/wise_processing.ipynb`, which progressively performs cuts to the WISE data, removing non-detections and many contaminating dwarf stars. This will generate a .csv file (`selecting_data/objs_2910.csv`) containing 2910 objects, a reduction on the original 218241 by a factor of 75!

## Downloading Data
### Grab URLs
Visit https://datalab.noirlab.edu/ (account required) and launch a jupyter notebook.
Upload both `selecting_data/objs_2910.csv` and `downloading_data/fetch_urls.ipynb` to the working directory. Run through the latter notebook, which in about 5mins generates a .txt file containing 13735 URLs of, which contain images of 2747 objects (not 2910 as some images were on the edge of the tile; perhaps other problems too). This file is `downloading_data/img_url_list.txt`
### Download Images
Run the shell script `downloading_data/download_imgs.sh`, a wget command which downloads all the image files into the `img_files` folder. This takes several hours, so if you can take advantage of a cluster that's even better.
### Compile Images
Run through `compile_imgs.ipynb`, which generates a .npz file containing a numpy array with [2737?] images and the corresponding IDs; this file is `images.npz`

## Clustering Images using Contrastive Learning
The notebook `contrastive_learning/trainer.ipynb` trains separates out the remaining objects into clusters. Due to the intensive tensorflow calculations required, it is only tractable to run this on a GPU. I only have access to one via Google Colab, so the notebook is written for that. If you are using a local GPU, you'll have to tweak things a bit.
The result of this notebook is a UMAP embedding - a list of points in 2D space that are representative of the separations of the images in the 1024-dimensional output of the encoder of the neural network. This embedding is stored in `embedding.npz`, along with the respective IDs. The notebook also gives a list of [117?] IDs of objects on a particular island, which is suspected to contain the gravitationally-lensed high-redshift quasars.
