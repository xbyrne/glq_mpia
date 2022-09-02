# glq_mpia
---
## Searching for Gravitationally-Lensed High-Redshift Quasars using Unsupervised Machine Learning

This is code for a Summer Internship I did at MPIA with Romain Meyer, in search of gravitationally lensed quasars.

This repo is for reproducibility's sake (mostly by myself!)

The project proceeds in the following order:

## Selecting Data
### Select Quasi-Quasar Objects from the Dark Energy Survey
Run the SQL query in selecting_data/loose_cuts.txt at https://des.ncsa.illinois.edu/desaccess/db-access (login required). These selection criteria are similar to those often used to select high-redshift quasars, but are somewhat looser in order not to exclude lensed quasars.
The result is a ~33MB .csv file, containing data on 218241 objects; this file is selecting_data/objs_2e5.csv
### Cross-match to the WISE Survey
Visit https://datalab.noirlab.edu/xmatch.php. Under the `Table Management` tab, upload the above .csv file. After a few minutes, it will be uploaded and will then appear in the first drop-down menu on the `Xmatch` tab. Select this file, then select the ra and dec columns in the next dropdowns. In the `select output columns`, select All.
From the 2nd table dropdowns, select `unwise_dr1`, `unwise_dr1.object`, `ra` and `dec`. In the `select output columns` dropdown, select 
