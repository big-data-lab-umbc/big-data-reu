*Bilinear interpolation*: the goal is to estimate interpolated bedmachine value for each track pixel. It is the same process of doing bilinear interpolation for other variables.  
Step 1: train RegularGridInterpolator based on surf_x, surf_y and bedmachine
Step 2: use trained function to get interpolated bedmachine based on each track pixel's track_x and track_y 

