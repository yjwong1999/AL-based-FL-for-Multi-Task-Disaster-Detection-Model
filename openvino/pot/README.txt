###########################################
 IMPORTANT PREREQUISITES
This guide is only proofcheck for OpenVINO 2021.4
If you are not using this version, some of the dependencies will not work
- you will have to debug yourself
###########################################

1) POT by API.txt
- Guide on how to install the openvino and openvino-dev API for POT using custom code
- Extra details:
	- If Im not mistaken, in the later version (OpenVINO 2022), some of the api in openvino-dev 
	  is moved to openvino api
	- So if you are using OpenVINO 2022, you may not need to install openvino-dev
	- But as mentioned, this guide is meant for OpenVINO 2021.4, so we have to install the
	  openvino-dev also


2) POT by DL Workbench
- Guide on how to use DL Workbench for OpenVINO 2021.4
- Extra details:
	- If Im not mistaken, the guide Dr Tham provided for me is not for OpenVINO 2021.4
	- So, there may have some issues using the guide
	- This guide should be no issue for OpenVINO 2021.4

3) mtl_quant.py
- The python code i used to quantize our multi-task model to int8 using custom code
- for other application, kindly change the code according to some tips provided in "POT by API.txt"

4) int8 quantization guide.txt
- how to quantize your own custom model using mtl_quant.py code