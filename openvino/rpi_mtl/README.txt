# unzipped the file
# let's say this zipped file is downloaded to Documents
cd Documents
cd rpi_mtl

# APPLICATION 1: Detect Video
python3 detect_video.py --model model/rpi_float16/saved_model.xml --input sample_images/demo_video.mp4 --device MYRIAD

# APPLICATION 2: Detect Image
python3 detect_image.py --model model/rpi_float16/saved_model.xml --input sample_images/672465551357255680_0.jpg --device MYRIAD


