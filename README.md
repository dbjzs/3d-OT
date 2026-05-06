# 3d-OT
[![Nature Methods](https://img.shields.io/badge/Published%20in-Nature%20Methods-0077b6?style=flat&logo=nature)](https://www.nature.com/articles/s41592-026-03034-9)
[![Stars](https://img.shields.io/github/stars/dbjzs/3d-OT?logo=GitHub&color=yellow)](https://github.com/dbjzs/3d-OT/stargazers)
![Python 3.10.13](https://img.shields.io/badge/python->=3.10-blue.svg)
[![Docs](https://readthedocs.org/projects/3d-OT/badge/?version=latest)](https://3d-ot.readthedocs.io/en/latest)
[![All Versions Unique Downloads](https://img.shields.io/badge/dynamic/json?color=blue&label=Data%20Downloads&query=$.stats.unique_downloads&url=https://zenodo.org/api/records/15089427)](https://zenodo.org/records/15089427)
[![License](https://img.shields.io/badge/License-Apache-blue.svg)](https://github.com/dbjzs/3d-OT/blob/main/LICENSE)

3d-OT can be used for single-modal and multimodal spatial domain recognition, single-modal, multi-modal, and cross-platform alignment tasks, as well as 3D reconstruction.
![workframe.png](/framework.jpg)


## Overview
3d-OT leverages the PointNet++ model and optimal transport with soft communication as its foundation. Through a modular integration approach, it seamlessly supports single-modal, multi-modal, and diverse spatial omics data from various technologies. By precisely extracting meaningful features from positional information, 3d-OT achieves high-resolution analysis of spatial omics data. Utilizing the framework of soft communication optimal transport, it effectively tackles challenges such as non-rigid alignment and inconsistent resolution during the alignment process. Furthermore, it introduces a novel evaluation metric, the Chamfer distance, to assess alignment quality.


## Installation via Github
#### 📥 Download
```
git clone https://github.com/dbjzs/3d-OT.git
cd 3d-OT
```
#### 🔧 environment
3d-OT is available for Python 3.10. We recommend to train 3d-OT models on a device with GPU support.  
* Using the conda install environment
```
conda create -n 3d-OT -c conda-forge python==3.10.13 libopenblas=0.3.25 r-base=4.3.1 r-mclust -y
conda activate 3d-OT
```
#### 🛠️package
* Then using pip install 3d-OT.
```
pip install -r requirements.txt
pip install .
```

## Jupyter Tutorial
```
pip install ipykernel
python -m ipykernel install --user --name=3d-OT --display-name="Python (3d-OT)" 
```
Please use the core name as follows:"Python [conda env:3d-OT]"


## Requirements
You'll need to install the following packages in order to run the codes.
* anndata==0.10.5.post1
* numpy==1.26.3
* pandas==2.2.3
* scanpy==1.9.8
* torch==2.2.0
* torch_geometric==2.4.0
* scikit-learn==1.4.0
* scipy==1.12.0
* scikit-misc==0.5.1
* tqdm==4.67.1
* rpy2==3.5.11
* plotly==6.0.1
* nbformat==5.10.4
* matplotlib-inline==0.1.6



## Tutorial
All the result tutorials mentioned in the text can be found [here](https://3d-ot.readthedocs.io/en/latest/)  
<img src='docs/show674.png' width='250'> <img src='docs/H3K27ac.PNG' width='350'> <img src='docs/3D.png' width='200'>  
All the h5ad files used for reproducing the results can be found in the [Zenodo repository](https://zenodo.org/records/15089427).
- Please use [issues](https://github.com/dbjzs/3d-OT/issues) to submit bug reports.


### Reference
- If you find 3d-OT useful for your research, please consider citing the 3d-OT manuscript [Nature Methods](https://www.nature.com/articles/s41592-026-03034-9).

```
@article{Dai,
  title = {3d-OT: a deep geometry-aware framework for heterogeneous slices alignment of spatial multi-omics},
  author = {Dai, et al.},
  year = {2026},
  journal = {Nature Methods},
  doi = {https://doi.org/10.1038/s41592-026-03034-9},
}
```

