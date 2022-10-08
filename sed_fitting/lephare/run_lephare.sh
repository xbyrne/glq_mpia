# Run all the LePHARE code, with my specifications
#
# Setting a directory name - change for your own if the repo is in a different dir
export GLQMPIADIR='/data/beegfs/astro-storage/groups/walter/byrne/glq_mpia'
# Assembling libraries
$LEPHAREDIR/source/sedtolib -t S -c zphot.para
$LEPHAREDIR/source/sedtolib -t Q -c zphot.para
$LEPHAREDIR/source/sedtolib -t G -c zphot.para
# Assembling filters
$LEPHAREDIR/source/filter -c zphot.para
# Computing magnitudes?
$LEPHAREDIR/source/mag_star -c zphot.para
$LEPHAREDIR/source/mag_gal -t Q -c zphot.para -EB_V 0
$LEPHAREDIR/source/mag_gal -t G -c zphot.para -MOD_EXTINC 4,8 -LIB_ASCII YES -EM_LINES YES
# Running photo-z code?
$LEPHAREDIR/source/zphota -c zphot.para
# Moving output spectra to the output_spectra folder, and then tarring
mv *.spec output_spectra
