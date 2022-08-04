# **Team 1 UMBC REU Summer 2022** 

## **Title:** Big Data and Machine Learning Techniques for Atmospheric Gravity Wave Detection

**Team Members:** Hannah Nguyen, Jorge Gonzalez, Kathryn Chen, Theodore Chapman, Logan Chambers

**RA:** Seraj A.M. Mostafa

**Mentor:** Dr. Jianwu Wang

**Colaborator:** Dr. Jia Yue, Dr. Chenxi Wang, Dr. Sanjay Purushotham

**Abstract:** Atmospheric gravity waves are produced when gravity attempts to restore disturbances through stable layers in the atmosphere. They have a visible effect on many atmospheric phenomena such as global circulation and air turbulence. Despite their importance, however, little research has been conducted on how to detect gravity waves using machine learning algorithms. We faced two major challenges in our research: our labeled dataset was extremely small, and our raw data had a lot of noise. In this study, we explored various methods of preprocessing and transfer learning in order to address those challenges. We pre-trained an autoencoder on unlabeled data before training it to classify labeled data. We also created a custom CNN by combining certain pre-trained layers from the InceptionV3 Model trained on ImageNet with custom layers and a custom learning rate scheduler. Our best model outperformed state-of-the-art architectures with a test accuracy 6.36% higher than the best performing baseline architecture.
