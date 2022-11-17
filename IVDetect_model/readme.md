# IVDetect readme
This repo is based on official github repo of [IVDetect](https://github.com/vulnerabilitydetection/VulnerabilityDetectionResearch)

This implementation also require a specific version of joern and a specific script for joern to extract the graph. please see the above link to the official github of IVDetect.
However, we will provide a pre-proceed dataset(as state in main readme file) in zenodo so that you can skip the joern step.

## 0 pre-process with joern(optional)
This step extract graph from code. We provide pre-processed dataset. But if you want to use your datasets, you need this step. 
1. unzip `joern.zip`
2. prepare raw datasets
3. check code in `preprocess.py`, change dataset directory,joern directory, and destination folder
4. run `preprocess.py
   `
**starting from here, you need the pre-processed dataset to continue.**
## 1 generate glove

embed word->vec with glove
1. download glove to this dir
2. uncomment line 75 in gen_graph.py
3. run `gen_graph.py`
4. run `glove/ast.sh` and `glove/phg.sh` to generate the glove embedding

## 2 generate graph file
1. run `gen_graphs.py`
   1. line#171 for output dir

## 3 train/test split
1. run`shard_splitter.py`

shard splitter will split dataset into 5 shard, use 4 for train, 1 for test. You decide.

## 4. model training & **raw code level sampling**
1. `--proceed_dir` where the shards at
2. `--sampling` to do **Sampling_R (ROS_R or RUS_R)**

## 5. parameter
1. line#175 for parameter
2. uncimment all #nni line and  comment line#195 for nni autoML
   1. `nnictl create --config config.yml` etc
   2. see microsoft NNI for more autoML fine-tuning detail

## 6. **latent level sampling** + retrain final classifier
1. use the `train_X, train_y ,test_X, test_y`vector(in numpy format) obtain from the previous step. Which should be train on origin imbalanced data.
2. see example notebook in `latent_find_tune.ipynb` 
3. change the imbalance learn class for other latent level samlpling approach to do **Sampling_L**

## 7. example result
```
train loss: 0.6700113380164868 acc: 0.579321892005023
evaluate >
54.24
auc: 59.92 acc: 54.24 precision: 49.77 recall: 72.11 f1: 58.89 
report:
              precision    recall  f1-score   support

     non-vul       0.63      0.39      0.48      1433
         vul       0.50      0.72      0.59      1194

    accuracy                           0.54      2627
   macro avg       0.56      0.56      0.54      2627
weighted avg       0.57      0.54      0.53      2627

matrix:
[[564 869]
 [333 861]]
 ```