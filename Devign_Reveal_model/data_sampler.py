import os
import random
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import argparse
import pandas as pd
import itertools

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling_type', type=str, help='Type of the sampling',
                        choices=['ros', 'oss', 'rus'], default='rus')
    parser.add_argument('--json_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--data_split_number', type=int, default=1)
    args = parser.parse_args()

    df = pd.read_json(args.json_dir)
    print(df.info())
    # merged = list(itertools.chain.from_iterable(list(itertools.chain.from_iterable(df['targets'].tolist()))))
    # print(merged)
    if args.sampling_type == 'ros':
        for i in range(args.data_split_number):
            train, test = train_test_split(df, random_state=i, test_size=0.2)
            train_targets = list(
                itertools.chain.from_iterable(list(itertools.chain.from_iterable(train['targets'].tolist()))))
            ros = RandomOverSampler()
            train_resampled, _ = ros.fit_resample(train, train_targets)
            print(train_resampled.info())
            print(test.info())
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train_resampled.to_json(f'{args.out_dir}data_split_{i}/train_GGNNinput.json', orient='records')
            test.to_json(f'{args.out_dir}data_split_{i}/test_GGNNinput.json', orient='records')

    if args.sampling_type == 'rus':
        for i in range(args.data_split_number):
            train, test = train_test_split(df, random_state=i, test_size=0.2)
            train_targets = list(
                itertools.chain.from_iterable(list(itertools.chain.from_iterable(train['targets'].tolist()))))
            rus = RandomUnderSampler()
            train_resampled, _ = rus.fit_resample(train, train_targets)
            print(train_resampled.info())
            print(test.info())
            if not os.path.exists(f'{args.out_dir}data_split_{i}'):
                os.makedirs(f'{args.out_dir}data_split_{i}')
            train_resampled.to_json(f'{args.out_dir}data_split_{i}/train_GGNNinput.json', orient='records')
            test.to_json(f'{args.out_dir}data_split_{i}/test_GGNNinput.json', orient='records')