# **Team 2 UMBC REU Summer 2023** 

## **Title:** Accelerating Real-Time Imaging for Radiotherapy: Leveraging Multi-GPU Training with PyTorch

**Team Members:** Ruth Obe, Sam Kadel, Kaelen Baird, Brandt Kaufmann, Yasmin Soltani

**RA:** Mostafa Cham

**Mentor:** Dr. Matthias K. Gobbert

**Colaborator:** Carlos A. Barajas, Zhuoran Jiang, Vijay R. Sharma, Dr. Lei Ren, Dr. Stephen W. Peterson, Dr. Jerimy C. Polf 

**Abstract:** Proton beam therapy is an advanced form of cancer radiotherapy that uses high-energy proton beams to deliver precise and targeted radiation to tumors. This helps to mitigate unnecessary radiation exposure in healthy tissues. Real-time imaging of prompt gamma rays with Compton cameras has been suggested to improve therapy efficacy. However, the camera's non-zero time resolution leads to incorrect interaction classifications and noisy images that are insufficient for accurately assessing proton delivery in patients. To address the challenges posed by the Compton camera's image quality, machine learning techniques are employed to classify and refine the generated data. These machine-learning techniques include recurrent and feedforward neural networks. A PyTorch model was designed to improve the data captured by the Compton camera. This decision was driven by PyTorch’s flexibility, powerful capabilities in handling sequential data, and enhanced GPU usage. This accelerates the model’s computations on large-scale radiotherapy data. Through hyperparameter tuning, the validation accuracy of our PyTorch model has been improved from an initial 7\% to over 60\%. Moreover, the PyTorch Distributed Data Parallelism strategy was used to train the RNN models on multiple GPUs, which significantly reduced the training time with a minor impact on model accuracy.
