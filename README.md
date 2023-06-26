# UDA-GS
Official implementation of "Generalize Model from Labeled BraTS to Local Unlabeled Datasets: A Cross-Center Multimodal Unsupervised Domain Adaptation Glioma Segmentation Framework". Submitted to Medical Image Analysis.

> Authors: Zhaoyu Hu, Yuhao Sun, Liuguan Bian, Chun Luo, Junle Zhu, Jin Zhu, Shiting Li, Zheng Zhao, Yuanyuan Wang, Huidong Shi,Zhifeng Shi,Jinhua Yu.

# Prerequisite

> - CUDA/CUDNN
> - Python3
> - PyTorch==1.8
> - Packages found in requirements.txt
1. Creat a new conda environment
```
conda create -n gs_env python=3.7
conda activate gs_env
conda install pytorch=1.8 torchvision torchaudio cudatoolkit -c pytorch
pip install -r requirements.txt
```
2. Download the code from github and change the directory

```
git clone https://github.com/ZacHu-ZYH/UDA-GS/
cd UDA-GS
```
3. Prepare dataset

Download FeTS dataset, then split source datasets (FeTS 1-14) and target datasets (FeTS 15-17). Organize the folder as follows:

```
├── ../../dataset/
│   ├── BraTs/     
|   |   ├── images/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/
│   ├── FeTS15/
|   |   ├── Flair/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/
│   ├── xinhua/ 
|   |   ├── Flair/
|   |   ├── labels/
|   |   ├── T1/
|   |   ├── T1c/
|   |   ├── T2/
...
```
BraTs folder includes FeTS 1 to FeTS 14, "images" in the folder means Flair modality.


# Training and Evaluation example

> Training and evaluation are on a single GPU.

### Train with unsupervised domain adaptation 

```
python train.py
```
### Evaluation 

```
python test.py
```


