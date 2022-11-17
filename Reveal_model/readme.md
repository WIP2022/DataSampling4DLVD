# Reveal model & Devign model readme

This repo is developed based on the official github repo of [Reveal](https://github.com/VulDetProject/ReVeal)

The Reveal & Devign model share same input/output format and shape, only network is different. 

## 0 pre-process with Reveal repo (optional)
For this step, please follow extactly how Reveal repo do. We provide a instince pipeline of how to process from raw code to processed data. Please note that the embedding and graph extraction are also finish at this step.
Please also note that for the BigVul(aka MSR)datasets, many code cannot be parsed by the reveal tool, so data point will be less than the raw datasets.

**starting from here, you need the pre-processed dataset to continue.**

## 1 train-test split
1.  use `data_sampler.py` to split the train-test set with ROS_R or RUS_R

## 2 train model with sampling_R
1.  see the bash file`exp.bash` for a example of quick start
2.  see the `main.py` arguments for further modification of parameters

## 3 train model with sampling_L
1. This required the model trained by previous step that train with noSampling.
2. see the `exp_latent.sh` for quick start
3. see the `backbone.py` arguments for futher modification parameters

## 4 GNNExplainer
1. this steps required the model trained by previous steps. Either a Sampling_R or Sampling_L model
2. this steps required your datasets comes with extra information of the `vulnerable line` as the groundtruth. Which is only provided in the BigVul(aka msr)datasets,
when proceeding the BigVul datasets in the pre-procesing step 0, please preserve the information. Or you can do a cross-join of the bigvul datasets with vulnerable line with the processed datasets by their tokens to add the line to the processed datasets.
3. see&run  `gnn_explainer.ipynb` to start the XAI with GNNExplainer,change model location and test data location at your demand.
4. run `GNN_explainer_result_analysis.ipynb` to calculate the hit rate.