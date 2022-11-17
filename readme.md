# DataSampling4DLVD
This is the official replication repository for our paper
> *Does data sampling improve deep learning-based
vulnerability detection? Yeas! and Nays!*

## 0.datasets
### 0.1 proceeded datasets dump availble in Zenodo:
Please click to see our zenodo site for our proceed datasets at:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7057996.svg)](https://doi.org/10.5281/zenodo.7057996)


### 0.2 raw datasets:
This repo consist three model that are developed based on their official releases github repo.
1. [IVDetect](https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch)
2. [Reveal](https://github.com/VulDetProject/ReVeal)
3. [LineVul/codebert](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Defect-detection)

We thank for researchers for their hard work.
## 1 models
The repository consist of 3 model replication folder. Also required package are listed.
###  1.1 IVDetect_model
Package:
1. Pytorch
2. Pytorch-geometric
3. imblearn
4. sklearn
5. gensim
6. nni
### 1.2 Reveal_model
>**Devign_model** also inside Reveal_model folder. This is because the author of Devign didn't open source the code, our implementation of Devign are based on the replicate written by Reveal's author.

Package:
1. pytorch
2. dgl (which includes the GNNExplainer implementation for XAI)
3. imblearn
4. sklearn

### 1.3 LineVul_model
package:
1. Pytorch
2. Transformer (by Huggingface)
3. Lime (if you want to use the XAI tool Lime)
4. tensorflow
5. imblearn
6. sklearn

## 2 Datasets
We use three datasets in the experiment, we provide only the link to the **raw** dataset here, and we will provide the **processed** datasets(as model input) in zenodo 
1. [Devign](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view) dataset
    1. for more detailed info about the devign datasets, check [Devign's official webpage](https://sites.google.com/view/devign)
2. [Reveal](https://drive.google.com/drive/folders/1KuIYgFcvWUXheDhT--cBALsfy1I4utOy) dataset
    1. for more detailed info about the reveal datasets, check [Reveal github](https://github.com/VulDetProject/ReVeal)
3. [BigVul](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view) dataset
    1. we use a cleaned version of BigVul, the origin BigVul contain much more information to digest, we suggest researchers to check the origin BigVul dataset at [BigVul](https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset) official repo
    
## 3 Interpretable Tool
We provide the code/jupyter-notebook that we use in our RQ and discussion part for future study
1. **Lime** is in LineVul_model folder
2. **GNNExplainer** is in Reveal_model folder


## 4 The whole pipeline
We tried our best to describe how to conduct the experiment from **raw data -> proceed data -> model traning -> evaluation** in each readme file in different folder of model