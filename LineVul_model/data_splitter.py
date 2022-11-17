import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import OneSidedSelection
import argparse
import pandas as pd
import itertools
from imblearn.under_sampling import OneSidedSelection
from tqdm import tqdm
tqdm.pandas()
def get_emb(model,tokenizer,code):
    code_token = tokenizer(code, padding=True, truncation=True, return_tensors='pt')
#     code_token.to(device)
    input_ids = code_token.input_ids.to('cuda')
#     print(input_ids)
#     context_embeddings=model(torch.tensor(code_token.input_ids)[None,:])[0]
    context_embeddings=model(input_ids)[0]
    mean_context_embeddings = torch.mean(context_embeddings,1).squeeze().cpu().detach().numpy()
    return mean_context_embeddings

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_type', type=str, help='Type of the sampling',
                        choices=['origin','ros', 'oss','rus'], default='origin')
    parser.add_argument('--json_dir', type=str,required=True)
    parser.add_argument('--out_dir', type=str,required=True)
    parser.add_argument('--data_split_number', type=int, default=20)
    args = parser.parse_args()
    
    df = pd.read_json(args.json_dir)
    print(df.info())
    print(df.head())
    # merged = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(df['targets'].tolist()))))
    # print(merged)
    if args.sampling_type == 'origin':
        for i in range(args.data_split_number):
            train, test = train_test_split(df,random_state=i,test_size=0.2)
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train.to_json(f'{args.out_dir}data_split_{i}/train.jsonl',orient='records', lines=True)
            test.to_json(f'{args.out_dir}data_split_{i}/test.jsonl',orient='records', lines=True)
    if args.sampling_type == 'ros':
        for i in range(args.data_split_number):
            train, test = train_test_split(df,random_state=i,test_size=0.2)
            # print(train['target'].tolist())
            # train_targets = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(train['target'].tolist())))) 
            train_targets = train['target'].tolist()
            ros = RandomOverSampler()
            train_resampled, _ = ros.fit_resample(train, train_targets)
            print(train_resampled.info())
            print(test.info())
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train_resampled.to_json(f'{args.out_dir}data_split_{i}/train.jsonl',orient='records', lines=True)
            test.to_json(f'{args.out_dir}data_split_{i}/test.jsonl',orient='records', lines=True)
        
    if args.sampling_type == 'rus':
        for i in range(args.data_split_number):
            train, test = train_test_split(df,random_state=i,test_size=0.2)
            train_targets = train['target'].tolist()
            rus = RandomUnderSampler()
            train_resampled, _ = rus.fit_resample(train, train_targets)
            print(train_resampled.info())
            print(test.info())
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train_resampled.to_json(f'{args.out_dir}data_split_{i}/train.jsonl',orient='records', lines=True)
            test.to_json(f'{args.out_dir}data_split_{i}/test.jsonl',orient='records', lines=True)
    if args.sampling_type == 'oss':
        import torch
        from transformers import AutoTokenizer, AutoModel
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base",return_tensors="pt")
        # tokenizer.to(device)
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        model.to(device)
        for i in range(args.data_split_number):
            train, test = train_test_split(df,random_state=i,test_size=0.2)
            train_targets = train.target.tolist()
            oss = OneSidedSelection()
            fea_mean = []
            train.progress_apply(lambda row: fea_mean.append(get_emb(model,tokenizer,row['func'])),axis=1)
            _, _ = oss.fit_resample(np.array(fea_mean), train_targets)
            train_resampled = train.iloc[oss.sample_indices_]
            print(train_resampled.info())
            print(test.info())
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train_resampled.to_json(f'{args.out_dir}data_split_{i}/train.jsonl',orient='records', lines=True)
            test.to_json(f'{args.out_dir}data_split_{i}/test.jsonl',orient='records', lines=True)
        
          
            