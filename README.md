# Joint Disaster Classification and Victim Detection using Multi-Task Learning
This repo provides some of the codes used in my following research projects: </br>
1. [Joint Disaster Classification and Victim Detection using Multi-Task Learning](https://ieeexplore.ieee.org/document/9666576)
2. [An Optimized Multi-Task Learning Model for Disaster Classification and Victim Detection in Federated Learning Environments](www.google.com) \* </br>
\* Research Work No. 2 is an extension based on Research Work No. 1


## Overview of Our Project
show images of our mtl model </br>
show images of rsa </br>
show images of our decoupled FL </br>


## Novelty
1. Existing studies focus on solving single-task issue of disaster classification [13,16,27-29] and victim detection separately. In contrast, we introduce a MTL model by attaching a disaster classification head model to the backbone of a victim detection model. 
2. The framework design decouples training of two tasks.
3. Most AL methods advocate uncertainty sampling, which selects the most uncertain samples from the unlabeled data pool to label [22]. Such strategy is ill-suited for disaster dataset, where samples from different classes exhibit high similarity. To enable efficient AL-based FL, we introduce a simple heuristic by combining both uncertainty and diversity samplings. 
4. The majority of the research tries to accelerate the inference process without detailing the degree of accuracy loss. In contrast, our measurement outputs are based on open-source and production-ready frameworks to ensure reusability, interoperability, and scalability.


## Other Repo(s)
For better readability, I separated the following codes from this repo, as they are not directly related to the AL-based-FL for the Multi-Task model. </br>
1. [MobileNetV2 for Disaster Classification](https://github.com/yjwong1999/MobileNetV2-for-Disaster-Classification)
2. [Representation Similarity Analysis](https://github.com/yjwong1999/Representation-Similarity-Analysis)
3. [GradCAM for YOLOv3](https://github.com/yjwong1999/GradCAM-for-YOLOv3)


## Overview of Federated Learning

Federated learning (FL) is an approach to Machine Learning (ML) or Deep Learning (DL), where a shared global model is trained across many participating clients that keep their training data locally. Some of the popular existing FL framework inclucdes (i) [Google's TensorFlow Federated (TFF)](https://www.tensorflow.org/federated/tutorials/tutorials_overview) and (ii) [Intel's Open Federated Learning (OpenFL)](https://github.com/intel/openfl). 

### Drawback(s) of TFF
As of today (18/9/2022), TFF is not production-ready yet. According to [TFF FAQ](https://www.tensorflow.org/federated/faq): "The current release is intended for experimentation uses, such as expressing novel federated algorithms, or trying out federated learning with your own datasets, using the included simulation runtime."

### Drawback(s) of OpenFL
OpenFL is a production-ready Python 3-based FL framework. Users can use OpenFL for both real-time application and also simulations. Similar to TFF, it also allows users to experiment and develop their own novel FL algorithm. 


## Tools
TensorFlow 2.0 </br>
Open Federated Learning (OpenFL) </br>
