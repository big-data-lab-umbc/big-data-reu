
# **Evaluating Machine Learning and Statistical Models for Greenland Bed Topography**

Team Members: Katherine Yi, Angelina Dewar, Tartela Tabassun, Jason Lu, Ray Chen


RA: Homayra Alam

Mentor: Dr. Jianwu Wang

Collaborator: Omar Faruque, Ph.D. student, Department of Information Systems, University of Maryland, Baltimore County

Collaborator: Sikan Li, Research Engineer, Texas Advanced Computing Center, University of Texas at Austin

Collaborator: Dr. Mathieu Morlinghem, Professor, Department of Earth Sciences, Dartmouth College

Abstract: 
The purpose of this research is to study how different machine learning and statistical models can be used to predict bed topography in Greenland using ice-penetrating radar and satellite imagery data. Accurate bed topography representations are crucial for understanding ice sheet stability, melt, and vulnerability to climate change. We explored nine predictive models including dense neural network, LSTM, variational auto-encoder (VAE), extreme gradient boosting (XGBoost), gaussian process regression, and kriging-based residual learning. Model performance was evaluated with mean absolute error (MAE), root mean squared error (RMSE), coefficient of determination (R$^2$), and terrain ruggedness index (TRI). In addition to testing various models, different interpolation methods, including nearest neighbor, bilinear, and kriging, were also applied in preprocessing. The XGBoost model with kriging interpolation exhibited strong predictive capabilities but demands extensive resources. Alternatively, the XGBoost model with bilinear interpolation showed robust predictive capabilities and required fewer resources. These models effectively captured the complexity of the Greenland ice sheet terrain with precision and efficiency, making them valuable tools for representing spatial patterns in diverse landscapes.
