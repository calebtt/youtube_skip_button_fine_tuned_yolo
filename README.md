## Environment Setup

This project uses a conda environment for reproducibility. Follow these steps to set it up:

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) if you haven't already.

2. Clone the repository

3. Create the environment from the provided `environment.yml`: conda env create -f environment.yml

4. Activate the environment: conda activate adskip

5. (Optional) Verify the installation: conda list

First commit of some scripts in the workspace for collecting training images and fine-tuning of a yolo v11 model to do the classification task of accurately detecting "skip ad" buttons on youtube videos. False positives are a particularly annoying issue if the detection is used to auto-click the "button", and FP may occur when running over long periods of time. Thus the detection images are saved and added to the training data to hopefully achieve a margin of error so low it would rarely if ever be encountered.

Actual training data not included, it consists of various public image datasets of compute usage as well as real data of me browsing the internet or playing video games, etc. with positive detection images annotated manually on makesense.ai (download labels as yolo .zip)
