# **Accurate and Interpretable Radar Quantitative Precipitation Estimation with Symbolic Regression**

**Team Members:** Olivia Zhang, Brianna Grissom, Julian Pulido, Kenia Munoz-Ordaz, Jonathan He

**RA:** Mostafa Cham

**Mentor:** Dr. Jianwu Wang

**Collaborators:** Haotong Jing, Weikang Qian, Dr. Yixin Wen

**Abstract:** Accurate quantitative precipitation estimation (QPE) is essential for managing water resources, monitoring flash floods, creating hydrological models, and more. Traditional methods of obtaining precipitation data from rain gauges and radars have limitations such as sparse coverage and inaccurate estimates for different precipitation types and intensities. Symbolic regression, a machine learning method that generates mathematical equations fitting the data, presents a unique approach to estimating precipitation that is both accurate and interpretable. Using WSR-88D dual-polarimetric radar data from Oklahoma and Florida over three dates, we tested symbolic regression models from genetic programming to deep learning, symbolic regression on separate clusters of the data, and the incorporation of knowledge-based loss terms into the loss function. We found that symbolic regression is both accurate in estimating rainfall and interpretable through learned equations. Accuracy and simplicity of the learned equations can be slightly improved by clustering the data based on select radar variables and by adjusting the loss function with knowledge-based loss terms. This research provides insights into improving QPE accuracy through interpretable symbolic regression methods.
