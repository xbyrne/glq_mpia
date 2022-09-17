# glq_mpia
## Searching for Gravitationally-Lensed High-Redshift Quasars using Unsupervised Machine Learning

This is code and files for a Summer Internship I did at MPIA with Romain Meyer, in search of gravitationally lensed quasars.

This repo is for reproducibility's sake (mostly by myself!)

The project proceeds in the following order:

## Selecting Data
### Select Quasi-Quasar Objects from the Dark Energy Survey
Run the SQL query in selecting_data/loose_cuts.txt at https://des.ncsa.illinois.edu/desaccess/db-access (login required). These selection criteria are similar to those often used to select high-redshift quasars, but are somewhat looser in order not to exclude lensed quasars.
The result is a ~33MB .csv file, containing data on 218241 objects; this file is `selecting_data/cut_des.csv`
### Cross-match to the WISE and VHS Surveys
Visit https://datalab.noirlab.edu/xmatch.php. Under the "Table Management" tab, upload the above .csv file. After a few minutes, it will be uploaded and will then appear in the first drop-down menu on the "Xmatch" tab. Select this file, then select the ra and dec columns in the next dropdowns. In the "select output columns", select All.
From the 2nd table dropdowns, select "unwise_dr1", "unwise_dr1.object" "ra" and "dec". In the "select output columns" dropdown, select 
[coadd_id, dec, dfluxlbs_w1, dfluxlbs_w2, flux_w1, flux_w2, fluxlbs_w1, fluxlbs_w2, mag_w1_vg, mag_w2_vg, ra]. The rest of the columns will not be needed and make the file unnecessarily large.
Choose a name for this cross-matched table, a radius of 1 arcsecond, and select "Nearest neighbor" and "Exclude non-matching rows". After ~1min, the resulting table will be saved to your MyDB under the name you chose.
To crossmatch to VHS, select this table in the first dropdown; select "t1_ra", "t1_dec", and all output columns. For the second table, select "vhs_dr5", "vhs_dr5.vhs_cat_v3", "ra2000", "dec2000", and the columns [dec2000, japermag4, japermag4err, ksapermag4, ksapermag4err, ra2000]. Choose a table name, a radius of 1", and select "Nearest neighbor, "Exclude non-matching rows" and "Download results to your computer only".
The resulting table will be a ~55MB .txt file (best converted to .csv for easy access) with now only 152028 objects; this file is `selecting_data/cut_desxwisexvhs.csv`.
### Perform cuts in WISE data
Run through the notebook `selecting_data/wise_processing.ipynb`, which progressively performs cuts to the WISE data, removing non-detections and many contaminating dwarf stars. This will generate a .csv file (`selecting_data/objs_2910.csv`) containing 2910 objects, a reduction on the original 218241 by a factor of 75!

## Downloading Data
### Grab URLs
Visit https://datalab.noirlab.edu/ (account required) and launch a jupyter notebook.
Upload both `selecting_data/objs_2910.csv` and `downloading_data/fetch_urls.ipynb` to the working directory. Run through the latter notebook, which in about 5mins generates a .txt file containing 13735 URLs of, which contain images of 2747 objects (not 2910 as some images were on the edge of the tile; perhaps other problems too). This file is `downloading_data/img_url_list.txt`
### Download Images
Run the shell script `downloading_data/download_imgs.sh`, a wget command which downloads all the image files into an `img_files` folder. This takes ~12 hours, so if you can take advantage of a cluster that's better. The resulting images are stored in `downloading_data/img_files.tar.gz`, so you can just extract them from there.
### Compile Images
Run through `compile_imgs.ipynb`, which generates a .npz file containing a numpy array with 2747 images and the corresponding IDs; this file is `images.npz`, and also contains the COADD IDs in the same order.

## Clustering Images using Contrastive Learning
The notebook `contrastive_learning/trainer.ipynb` trains separates out the remaining objects into clusters. Due to the intensive tensorflow calculations required, it is only tractable to run this on a GPU. I only have access to one via Google Colab, so the notebook is written for that (if you are using a local GPU, you'll have to tweak things a bit). On Colab, click on files tab on the left and drag the `images.npz` file into the current directory.
The result of this notebook is a UMAP embedding - a list of points in 2D space that are representative of the separations of the images in the 1024-dimensional output of the encoder of the neural network. This embedding is stored in `embedding.npz`, along with the respective IDs. The notebook also gives a list of [117?] IDs of objects on a particular island, which is suspected to contain the gravitationally-lensed high-redshift quasars.

## SED Fitting
We now have a set of interesting objects and their photometry. We now attempt to fit their photometry to SED models for stars, galaxies, quasars, and of course lensed quasars (i.e. a galaxy + a quasar). This is done using two different pieces of software: LePHARE (downloaded from https://www.cfht.hawaii.edu/~arnouts/LEPHARE/lephare.html) and BAGPIPES (a Python package).
