This readme file also the repo is based on 'codeXglue'[https://microsoft.github.io/CodeXGLUE/] which is the same as the backbone network of the model proposed in 'LineVul'[https://github.com/awsm-research/LineVul]
# CodeXGLUE -- Defect Detection

## Task Definition

Given a source code, the task is to identify whether it is an insecure code that may attack software systems, such as resource leaks, use-after-free vulnerabilities and DoS attack.  We treat the task as binary classification (0/1), where 1 stands for insecure code and 0 for secure code.


### Data Format

After preprocessing dataset, you can obtain three .jsonl files, i.e. train.jsonl, valid.jsonl, test.jsonl

For each file, each line in the uncompressed file represents one function.  One row is illustrated below.

   - **func:** the source code
   - **target:** 0 or 1 (vulnerability or not)
   - **idx:** the index of example

### Input predictions

A predications file that has predictions in TXT format, such as evaluator/predictions.txt. For example:

```shell
0	0
1	1
2	1
3	0
4	0
```

I append a prediction with probability file as like the above one.

## Pipeline-CodeBERT

We also provide a pipeline that fine-tunes [CodeBERT](https://arxiv.org/pdf/2002.08155.pdf) on this task.

## data split & data sampling
run file `data_splitter.py` to do data splitting and data sampling
  1. change parameter `--sampling_type` to do **raw code level sampling**

### run with **Sampling_R**
See example in `exp.bash` or see under:
```shell
cd code
CUDA_VISIBLE_DEVICES=3, python my_run.py \
    --output_dir=./devign_output/origin/saved_models_0 \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base \
    --do_train \
    --do_test\
    --train_data_file=../devign_dataset/origin/data_split_0/train.jsonl \
    --test_data_file=../devign_dataset/origin/data_split_0/test.jsonl \
    --epoch 4 \
    --block_size 400 \
    --train_batch_size 32 \
    --eval_batch_size 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0
```

### run with **Sampling_L**
require the NoSampling trained model from previous step
see example run script in `exp_latent.bash`

### explaination:
1. CUDA_VISIBLE_DEVICES -> which gpu to use. Count from 0. I usually might using the 0, you can use the other.
2. do_train -> actually the fine tuning process
3. do_test -> use with do_test, it will run the test set and calculate the performance 
4. epochs -> usually 4 is enough, you can try other
5. train_batch_size -> dont modify. 3090 gpu can only hold 32
6. learning_rate 5e-5 -> this one you can try from 2e-5 to 5e-5
7. data_split_x -> used in my project for see performance variance, you can just use the 0 for now i guess
8. in order to validate if you methods works(i,e, augmentation/transformation on code), only change the data in train file and keep the test file unmodified for fair comparision with the baseline
9. model might take hours or days to run
### Result
1. result will be printed out at console
2. fine-tuned model will be save at out_dir/model.bin
3. also prediction and prediction_probability file will be at output dir, you can run `python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p outdir/predictions.txt -b outdir/prediction_prob.txt` to evaluate again

### Lime
1. For sampling_R, run the jupyter notebook `lime_explainer.ipynb`
2. For samplong_L, rin the jupyter notebook `lime_explainer-latent.ipynb`
3. run `lime_result_analyser.ipynb` to calculate the hit rate


## Reference
<pre><code>
@inproceedings{fu2022linevul,
  title={LineVul: A Transformer-based Line-Level Vulnerability Prediction},
  author={Fu, Michael and Tantithamthavorn, Chakkrit},
  booktitle={2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR)},
  year={2022},
  organization={IEEE}
}
</code></pre>
