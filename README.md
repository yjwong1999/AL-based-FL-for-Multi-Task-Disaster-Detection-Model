# Joint Disaster Classification and Victim Detection using Multi-Task Learning
This repo provides some of the codes used in my following research projects: </br>
1. [Joint Disaster Classification and Victim Detection using Multi-Task Learning](https://ieeexplore.ieee.org/document/9666576)
2. [An Optimized Multi-Task Learning Model for Disaster Classification and Victim Detection in Federated Learning Environments](www.google.com) \* </br>
\* Research Work No. 2 is an extension based on Research Work No. 1

Feel free to check out our Research Work 2, which is OpenAccess and free to download :3

## Overview of Our Project
![image](https://user-images.githubusercontent.com/55955482/197429318-e3f33cc0-581e-4546-afd6-6c774643d999.png)
We design a Multi-Task model for joint disaster classification and victim detection using Representation Similarity Analysis (RSA). We train the model using both the conventional Centralized Learning (CL) and Federated Learning (FL) methods. We also tried Active Learning (AL) to see how it could help in reducing the labeling workload for disaster dataset. Finally, we use OpenVINO for model optimization and inference optimization.

### Representation Similarity Analysis
Representation Similarity Analysis (RSA) is to measure the similarity of the feature maps extracted by two models. The paper suggested that "RSA could beused for deciding different branching out locations for different tasks, depending on their similarity with the representations at different depth of the shared root". Our work exploits RSA to pinpoint the optimal branching location (for the multi-task model).

### Federated Learning
Federated learning (FL) is an approach to Machine Learning (ML) or Deep Learning (DL), where a shared global model is trained across many participating clients that keep their training data locally. Some of the popular existing FL framework inclucdes (i) [Google's TensorFlow Federated (TFF)](https://www.tensorflow.org/federated/tutorials/tutorials_overview) and (ii) [Intel's Open Federated Learning (OpenFL)](https://github.com/intel/openfl). 

### Active Learning
Active Learning (AL) is a special case of semi-supervised machine learning. Citing a quote from [Datacamp](https://www.datacamp.com/tutorial/active-learning), "The main hypothesis in active learning is that if a learning algorithm can choose the data it wants to learn from, it can perform better than traditional methods with substantially less data for training."


## Novelty
1. Existing studies focus on solving single-task issue of disaster classification and victim detection separately. In contrast, we introduce a MTL model by attaching a disaster classification head model to the backbone of a victim detection model. We employ an efficient mathematical analysis to pinpoint the optimal branching location and to prune the head model.
2. The framework design decouples training of two tasks.
3. Most AL methods advocate uncertainty sampling, which selects the most uncertain samples from the unlabeled data pool to label [22]. Such strategy is ill-suited for disaster dataset, where samples from different classes exhibit high similarity. To enable efficient AL-based FL, we introduce a simple heuristic by combining both uncertainty and diversity samplings. 
4. The majority of the research tries to accelerate the inference process without detailing the degree of accuracy loss. In contrast, our measurement outputs are based on open-source and production-ready frameworks to ensure reusability, interoperability, and scalability.


## Other Repo(s)
For better readability, I separated the following codes from this repo, as they are not directly related to the AL-based-FL for the Multi-Task model. </br>
1. [MobileNetV2 for Disaster Classification](https://github.com/yjwong1999/MobileNetV2-for-Disaster-Classification)
2. [Representation Similarity Analysis](https://github.com/yjwong1999/Representation-Similarity-Analysis)
3. [GradCAM for YOLOv3](https://github.com/yjwong1999/GradCAM-for-YOLOv3)


## Tools
TensorFlow 2.0 </br>
Open Federated Learning (OpenFL) </br>
