
# **Interpolation code**
**Nearest Neighbor Interpolation**:Nearest Neighbor Interpolation involved associating each radar bedrock observation with its spatially nearest surface observation.

**Bilinear Interpolation**:Bilinear Interpolation predicted the value of each surface variable by calculating the weighted average of the four nearest neighbor observations of that variable. The weights of each neighbor were determined based on their distances from the prediction location.

**Kriging Interpolation**:Universal Kriging incorporated all observations in the dataset (or a specified neighborhood due to memory constraints) to calculate a weighted average. For more inormation there is a separate readme file Universal Kriging interpolation. 



