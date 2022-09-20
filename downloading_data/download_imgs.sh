# Downloads all files in img_url_list.txt to ./img_files.
# -nv: Non-verbal, not much output
# -nc: no clobber, doesn't overwrite any files that are already downloaded
# -b: background. Switch off for more info about how it's going
# -i: draw URLs from the following text file
# -P: send image files to the following folder

wget -nv -nc -i img_url_list.txt -P ./img_files
