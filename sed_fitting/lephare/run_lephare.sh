# Run all the LePHARE code, with my specifications
#
# Assembling libraries
source/sedtolib -t S -c config/zphot.para
source/sedtolib -t Q -c config/zphot.para
source/sedtolib -t G -c config/zphot.para
# Assembling filters
source/filter -c config/zphot.para
# Computing magnitudes?
source/mag_star -c config/zphot.para
source/mag_gal -t Q -c config/zphot.para -EB_V 0
source/mag_gal -t G -c config/zphot.para -MOD_EXTINC 4,8 -LIB_ASCII YES -EM_LINES YES
# Running photo-z code?
source/zphota -c config/zphot.para
