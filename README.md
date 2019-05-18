# deeplearning-textures
Aalto University Deep learning course project. Experiments in texture classification from images.

Download MINC-2500 dataset from http://opensurfaces.cs.cornell.edu/publications/minc/.
Download FMD dataset from http://people.csail.mit.edu/celiu/CVPR2010/FMD/

Set corresponding `data_path` variables in `fit_minc.py` and `fit_fmd.py` and run to fit models. Set `workers` to the number of processor thread used for data preprocessing. 
Should reach an accuracy of about 58% on MINC and 30% on FMD.

The conda environment in `environment.yml` requires CUDA 10.0.
