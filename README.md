# CAS-Colon Dataset & Framework

Official implementation for ​**​"CAS-Colon: A Comprehensive Colonoscopy Anatomical Segmentation Dataset for Artificial Intelligence Development"​**​

## Code Architecture
```bash
src/
├── config/               # Runtime configuration
├── data/RJ/             # Dataset and splits
├── models/              # Model architectures
├── utils/               # Dataloader & trainer utilities
├── pre.ipynb            # Video→frame & video crop preprocessing 
├── setting.py           # Runtime configuration
└── train.py             # Main training script
```
## Data Preparation
1. get data from ​**​"CAS-Colon: A Comprehensive Colonoscopy Anatomical Segmentation Dataset for Artificial Intelligence Development"​**
   
## Training
```
python train.py --save_dir xx --gpu_id --config xx
