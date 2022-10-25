####################################
# Setup
	- detect_image.py and detect_video.py is for Intel CPU/GPU/iGPU
	- rpi_detect_image.py and rpi_detect_video.py is for Intel VPU/NCS2
####################################
# let's say the files are downloaded to Documents
cd Documents
cd deployment





####################################
# For RPI + NCS2
	- the model used is inside model/rpi_float_16
	- NCS2 only support OpenVINO FP16 type
	- NCS2 does not support NMS layer, so NMS are removed from the model, and is implemented in:
		- rpi_detect_video.py 
		- rpi_detect_image.py
####################################
# APPLICATION 1: Detect Video
python3 rpi_detect_video.py --model model/rpi_float16/saved_model.xml --input sample_images/demo_video.mp4 --device MYRIAD

# APPLICATION 2: Detect Image
python3 rpi_detect_image.py --model model/rpi_float16/saved_model.xml --input sample_images/672465551357255680_0.jpg --device MYRIAD





####################################
# For Intel iGPU or GPU
	- the model used is inside model/mtl
	- GPU support both FP32 and FP16 (FP16 is prefered because faster and smaller, but I think I provided FP32)
####################################
# APPLICATION 1: Detect Video
python3 detect_video.py --model model/mtl/saved_model.xml --input sample_images/demo_video.mp4 --device GPU

# APPLICATION 2: Detect Image
python3 detect_image.py --model model/mtl/saved_model.xml --input sample_images/672465551357255680_0.jpg --device GPU





####################################
# For Intel CPU
	- the model used is inside model/int8
	- CPU support FP32, FP16, int8 (int8 is prefered because faster and smaller)
####################################
# APPLICATION 1: Detect Video
python3 detect_video.py --model model/int8/mtl_mixed_DefaultQuantization.xml --input sample_images/demo_video.mp4 --device CPU

# APPLICATION 2: Detect Image
python3 detect_image.py --model model/int8/mtl_mixed_DefaultQuantization.xml --input sample_images/672465551357255680_0.jpg --device CPU