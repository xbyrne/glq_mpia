export LEPHAREDIR=$(pwd)

source/sedtolib -t S -c ./config/zphot.para
source/sedtolib -t Q -c ./config/zphot.para
source/sedtolib -t G -c ./config/zphot.para

source/filter -c ./config/zphot.para

source/mag_star -c ./config/zphot.para
source/mag_gal -t Q -c ./config/zphot.para -EB_V 0
source/mag_gal -t G -c ./config/zphot.para -MOD_EXTINC 4,8  -LIB_ASCII NO

source/zphota -c ./config/zphot.para

mv ./*.chi ./*.spec output_spectra_chi