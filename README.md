# [TM-DOVI] tomato maturity detection of segmentation
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.9-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.4.1-purple)


## A Robust Tomato Maturity Detection Method Against Occlusion and Variable Illumination
![image](all_image.png)
*****
## Requirements
Python                  3.9.19  
torch                   2.4.1  
torchvision             0.19.1  
mamba-ssm               2.2.4   
numpy                   1.23.5  
matplotlib              3.9.2  
tqdm                    4.65.2  
einops                  0.8.0  
causal-conv1d           1.5.0.post8  


*****
## Data preparation
- Laboro Tomato: Download this dataset from [Laboro](https://github.com/laboroai/LaboroTomato)
- Yu Tomato: Download this dataset from [Yu](http://dx.doi.org/10.57760/sciencedb.j00001.00946)
*****
## Get start
- Train
Running > train.py
- Test
Running predict.py
****
## Pretrained Models
Model	#param.	Top-1 Acc.	Top-5 Acc.	Huggingface Repo
Vim-tiny	7M	76.1	93.0	https://huggingface.co/hustvl/Vim-tiny-midclstok
Vim-tiny⁺	7M	78.3	94.2	https://huggingface.co/hustvl/Vim-tiny-midclstok
Vim-small	26M	80.5	95.1	https://huggingface.co/hustvl/Vim-small-midclstok
Vim-small⁺	26M	81.6	95.4	https://huggingface.co/hustvl/Vim-small-midclstok
Vim-base	98M	81.9	95.8	https://huggingface.co/hustvl/Vim-base-midclstok

## Acknowledgement
We are very grateful for these excellent works [Hrnet](https://github.com/bubbliiiing/hrnet-pytorch)、[Vision mamba](https://github.com/hustvl/Vim), which have provided the basis for our framework.
*****
## Citation
