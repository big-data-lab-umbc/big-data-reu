**Team 2b** of the Big Data REU program at UMBC for Summer 2026

**Title:** Improving Geant4-Simulated Compton Camera Image Reconstruction for Radiopharmaceutical Therapy using 3D Residual U-Nets

**Team Members:** Ayan Kabaria<sup>1</sup>, Muhammad Khalid<sup>2</sup>, Sophia Lopez<sup>3</sup>, Abby Nam<sup>4</sup>, Ertan Dogan<sup>5</sup>, Sidhya Pathak<sup>6</sup>, Victor Sandrin<sup>7</sup>, Xueying Sun<sup>8</sup>, Ehsan Shakeri<sup>9</sup>, Harrison Lewis<sup>9</sup>, Hussam Fateen<sup>9</sup>, Matthias K. Gobbert<sup>9</sup>, Farshad Safavi<sup>10</sup>, Ananta Chalise<sup>10</sup>, Lei Ren<sup>11</sup>, Stephen W. Peterson<sup>12</sup>, and Jerimy C. Polf<sup>13</sup>

<sup>1</sup>River Hill High School, Howard County, Maryland\
<sup>2</sup>Department of Mathematics, Baruch College, City University of New York\
<sup>3</sup>Department of Mathematics and Statistics, University of North Carolina at Greensboro\
<sup>4</sup>Department of Psychology, Lafayette College\
<sup>5</sup>A. James Clark School of Engineering, University of Maryland, College Park\
<sup>6</sup>Department of Computer Science, University of Virginia\
<sup>7</sup>Department of Neuroscience, University of Arizona\
<sup>8</sup>Department of Information Technology and Management, Illinois Institute of Technology\
<sup>9</sup>Department of Mathematics and Statistics, University of Maryland, Baltimore County\
<sup>10</sup>Department of Radiation Oncology, University of Maryland School of Medicine\
<sup>11</sup>Department of Radiation Oncology, Northwestern University\
<sup>12</sup>Department of Physics, University of Cape Town, South Africa\
<sup>13</sup>M3D, Inc.\

**Abstract:** Accurate 3D imaging of radiopharmaceutical distribution is critical for personalized dosimetry and monitoring of malignant tumor characteristics. This study presents a comprehensive pipeline for simulating and reconstructing Yttrium-90 (Y-90) emissions using a one-stage Compton camera comprised of four crystal modules. We developed a Geant4-based Monte Carlo simulation of an isotropically distributed Y-90 source within a synthesized regular tumor, separated from the detector by an air gap. To emulate a tomographic scan, the simulated camera swept in a circular path, capturing the positions and energy depositions of gamma photons from 60 distinct orientations around both the X and Z axes of the centrally placed tumor. These readings were then reconstructed into 3D images using a kernel weighted back projection (KWBP) method. Because this physics-based reconstruction is limited by noise, blurring, and artifacts due to angular sampling restrictions, we developed an advanced 3D U-Net to denoise the reconstructed volumes. The model was trained using a hybrid loss function that combines mean squared error (MSE) and the structural similarity index measure (SSIM) to ensure both pixel-level accuracy and structural fidelity. We evaluated multiple network configurations, ultimately optimizing performance by integrating a dynamic learning rate, an additional residual U-Net block, and online data augmentation during training. Evaluated on a withheld test set, our best-performing configuration effectively mitigated reconstruction noise and artifacts, achieving a hybrid loss of 0.02730, an MSE of 0.00663, and an SSIM of 0.89002. These findings demonstrate that our two-step pipeline, combining physics-based KWBP reconstruction with optimized, data-driven 3D denoising, is an effective method for refining Compton camera images for clinical dosimetry and tumor monitoring applications.

## Navigating this Repository
This repository contains our models for image reconstruction, generalized for use with new data. Each model is a python script named accordingly to the model type and methods implemented with u-net being our original model adapted from 2D image reconstruction to 3D for this project.
