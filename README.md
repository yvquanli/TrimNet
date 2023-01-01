# TrimNet

TrimNet is a lightweight message passing neural network for multiple molecular property predictions.

TrimNet can accurately complete multiple molecular properties prediction tasks with significant parameter reduction, including the quantum properties, bioactivity, physiology, and compound-protein interaction (CPI) prediction tasks.

### Requirements 

```
PyTorch >= 1.4.0
torch-geometric >= 1.3.2
rdkit >= '2019.03.4'
```

### Usage example

**For quantum dataset**
```sh
git clone https://github.com/yvquanli/trimnet
cd ./TrimNet/trimnet_quantum/src
# download dataset from https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/molnet_publish/qm9.zip
# unzip the file to trimnet_quantum/dataset/raw
python run.py

usage: python run.py [task] [depth] [cuda_device] [seed]
```

**For drug dataset**
```sh
git clone https://github.com/yvquanli/trimnet
cd ./TrimNet/trimnet_drug/source
python run.py --dataset bace

usage: python run.py [-h] [--data DATA] [--dataset DATASET] [--seed SEED]
                     [--gpu GPU [GPU ...]] [--hid HID] [--heads HEADS]
                     [--depth DEPTH] [--dropout DROPOUT] [--batch_size BATCH_SIZE]
                     [--epochs EPOCHS] [--lr LR] [--weight_decay WEIGHT_DECAY]
                     [--lr_scheduler_patience LR_SCHEDULER_PATIENCE]
                     [--early_stop_patience EARLY_STOP_PATIENCE]
                     [--lr_decay LR_DECAY] [--focalloss] [--eval]
                     [--exps_dir EXPS_DIR] [--exp_name EXP_NAME]

```


## Authors 

* **Yuquan Li** - *Initial work, model design, benckmark on the qm9 dataset* - [Yuquan](https://github.com/yvquanli)
* **Pengyong Li** - *Model design, benckmark on drug datasets and CPI datasets* - [Pengyong](https://github.com/pyli0628)

## Citation

Pengyong Li, Yuquan Li, Chang-Yu Hsieh, et al. TrimNet: learning molecular representation from 
let messages for biomedicine[J]. Briefings in bioinformatics, 2020.

@article{10.1093/bib/bbaa266,  
    author = {Li, Pengyong and Li, Yuquan and Hsieh, Chang-Yu and Zhang, Shengyu and Liu, Xianggen and Liu, Huanxiang and Song, Sen and Yao, Xiaojun},  
    title = "{TrimNet: learning molecular representation from triplet messages for biomedicine}",  
    journal = {Briefings in Bioinformatics},  
    year = {2020},  
    month = {11},  
    issn = {1477-4054},  
    doi = {10.1093/bib/bbaa266},  
    url = {https://doi.org/10.1093/bib/bbaa266},  
    note = {bbaa266},  
}  





