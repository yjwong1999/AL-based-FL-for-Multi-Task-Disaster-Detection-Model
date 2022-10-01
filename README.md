# AL-based-FL-for-Multi-Task-Disaster-Detection-Model

# Tools
TensorFlow 2.0 </br>
Open Federated Learning (OpenFL) </br>

## Federated Learning Framework

Federated learning (FL) is an approach to Machine Learning (ML) or Deep Learning (DL), where a shared global model is trained across many participating clients that keep their training data locally. Some of the popular existing FL framework inclucdes (i) [Google's TensorFlow Federated (TFF)](https://www.tensorflow.org/federated/tutorials/tutorials_overview) and (ii) [Intel's Open Federated Learning (OpenFL)](https://github.com/intel/openfl). 

### Drawback(s) of TFF
As of today (18/9/2022), TFF is not production-ready yet. According to [TFF FAQ](https://www.tensorflow.org/federated/faq): "The current release is intended for experimentation uses, such as expressing novel federated algorithms, or trying out federated learning with your own datasets, using the included simulation runtime."

### Drawback(s) of OpenFL
OpenFL is a production-ready Python 3-based FL framework. Users can use OpenFL for both real-time application and also simulations. Similar to TFF, it also allows users to experiment and develop their own novel FL algorithm. 
