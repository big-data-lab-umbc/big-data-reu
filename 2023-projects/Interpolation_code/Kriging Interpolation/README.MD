# **Interpolation using UK_Interpolation.py**

## Introduction
This guide provides step-by-step instructions on utilizing the `UK_Interpolation.py` script to perform Universal Kriging-based interpolation on individual features from the `hackathon.h5` dataset.

## Usage
To perform interpolation for a single variable, follow these steps:

1. Download the `hackathon.h5` dataset and move it to the same directory as the code files.

2. Open a terminal or command prompt.

3. Navigate to the directory where the `UK_Interpolation.py` script is located.

4. Run the following command, replacing `surf_vy` with the variable you wish to interpolate:
python UK_Interpolation.py  surf_vy

5. Upon completion, a file named `surf_vy.csv` will be generated.

6. Repeat the above command for all five features (`surf_vx`, `surf_vy`, `surf_elv`, `surf_SMB`, and `surf_dhdt`).

7. Combine the resulting CSV files' columns to obtain a consolidated CSV file containing the interpolated output for all surface features.

Please note that you should execute steps 4 to 6 for each feature individually before combining the outputs.

Remember that the Universal Kriging approach enhances the accuracy of interpolation by considering both spatial trends and correlations among the features.
