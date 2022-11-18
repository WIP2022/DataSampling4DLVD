import argparse
from cmath import log
import os
import pickle
import sys

import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import Adam

from data_loader.dataset import DataSet
from modules.model import DevignModel, GGNNSum
# from trainer import train
from my_trainer import my_train
from utils import tally_param, debug
import logging

if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--sampling', type=str, required=True, help='sampling method')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--data_split', type=str, default='1')
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=128)

    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    model_dir = os.path.join(f'{args.dataset}_result', f'{args.model_type}_model', args.sampling, args.data_split)
    print('out dir ', model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=f'{model_dir}.log', level=logging.INFO, format=LOG_FORMAT)
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        logging.info('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
        logging.info(f'{len(dataset.train_examples)}, {len(dataset.valid_examples)}, {len(dataset.test_examples)}')
    else:
        debug('generate new dataset')
        logging.info('generate new dataset')
        dataset = DataSet(train_src=os.path.join(input_dir, 'train_GGNNinput.json'),
                          valid_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          test_src=os.path.join(input_dir, 'test_GGNNinput.json'),
                          batch_size=args.batch_size, n_ident=args.node_tag, g_ident=args.graph_tag,
                          l_ident=args.label_tag)
        file = open(processed_data_path, 'wb')
        pickle.dump(dataset, file)
        file.close()
        debug(f'processed file dump to {processed_data_path}')
        logging.info(f'processed file dump to {processed_data_path}')
    assert args.feature_size == dataset.feature_size, \
        'Dataset contains different feature vector than argument feature size. ' \
        'Either change the feature vector size in argument, or provide different dataset.'
    if args.model_type == 'ggnn':
        debug('model: GGNN')
        logging.info('model: GGNN')
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
    else:
        debug('model: Devign')
        logging.info('model: Devign')
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)

    debug('Total Parameters : %d' % tally_param(model))
    logging.info('Total Parameters : %d' % tally_param(model))
    # model.cuda()
    model.to(args.device)
    # loss_function = BCELoss(reduction='sum')
    loss_function = BCELoss(reduction='mean')
    optim = Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    # train(model=model, dataset=dataset, max_steps=1000000, dev_every=128,
    #       loss_function=loss_function, optimizer=optim,
    #       save_path=model_dir + '/GGNNSumModel', max_patience=100, log_every=None, device=args.device)
    my_train(model=model, epochs=50, dataset=dataset, loss_function=loss_function, optimizer=optim,
             save_path=model_dir + '/Model', device=args.device)
