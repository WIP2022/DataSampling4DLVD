import argparse
from cmath import log
import os
import pickle
import sys
from data_loader.batch_graph import GGNNBatchGraph, BatchGraph
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f
import copy
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
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import OneSidedSelection
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc


class GGNNSum_MLP(nn.Module):
    def __init__(self, backbone):
        super(GGNNSum_MLP, self).__init__()
        self.inp_dim = backbone.inp_dim
        self.out_dim = backbone.out_dim
        self.max_edge_types = backbone.max_edge_types
        self.num_timesteps = backbone.num_timesteps
        self.ggnn = backbone.ggnn
        self.classifier = backbone.classifier

    #         self.sigmoid = nn.Sigmoid()

    def forward(self, batch, device):
        graph, features, edge_types = batch.get_network_inputs(cuda=True, device=device)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = batch.de_batchify_graphs(outputs)
        return h_i.sum(dim=1)


#         ggnn_sum = self.classifier(h_i.sum(dim=1))
#         result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
#         return ggnn_sum

class DevignModel_MLP(nn.Module):
    def __init__(self, backbone):
        super(DevignModel_MLP, self).__init__()
        self.inp_dim = backbone.inp_dim
        self.out_dim = backbone.out_dim
        self.max_edge_types = backbone.max_edge_types
        self.num_timesteps = backbone.num_timesteps
        self.ggnn = backbone.ggnn
        self.conv_l1 = backbone.conv_l1
        self.maxpool1 = backbone.maxpool1
        self.conv_l2 = backbone.conv_l2
        self.maxpool2 = backbone.maxpool2

        self.concat_dim = backbone.concat_dim
        self.conv_l1_for_concat = backbone.conv_l1_for_concat
        self.maxpool1_for_concat = backbone.maxpool1_for_concat
        self.conv_l2_for_concat = backbone.conv_l2_for_concat
        self.maxpool2_for_concat = backbone.maxpool2_for_concat

        self.mlp_z = backbone.mlp_z
        self.mlp_y = backbone.mlp_y

    def forward(self, batch, device):
        graph, features, edge_types = batch.get_network_inputs(cuda=True, device=device)
        #         print("batch contain:",batch.num_of_subgraphs)
        graph = graph.to(device)
        features = features.to(device)
        edge_types = edge_types.to(device)
        outputs = self.ggnn(graph, features, edge_types)
        #         print("features shape",features.shape)
        #         print("outputs shape",outputs.shape)
        x_i, _ = batch.de_batchify_graphs(features)
        #         print("x_i shape",x_i.shape)
        h_i, _ = batch.de_batchify_graphs(outputs)
        #         print("h_i shape",h_i.shape)
        c_i = torch.cat((h_i, x_i), dim=-1)
        #         print("c_i shape",c_i.shape)
        #         print(h_i.transpose(1, 2).shape)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(
            f.relu(
                self.conv_l1(h_i.transpose(1, 2))
            )
        )
        Y_2 = self.maxpool2(
            f.relu(
                self.conv_l2(Y_1)
            )
        ).transpose(1, 2)
        Z_1 = self.maxpool1_for_concat(
            f.relu(
                self.conv_l1_for_concat(c_i.transpose(1, 2))
            )
        )
        Z_2 = self.maxpool2_for_concat(
            f.relu(
                self.conv_l2_for_concat(Z_1)
            )
        ).transpose(1, 2)
        #         print("Y_2 shape",Y_2.shape)
        #         print("Z_2 shape",Z_2.shape)
        Z_3 = self.mlp_z(Z_2)
        Y_3 = self.mlp_y(Y_2)
        ap_3 = torch.cat((Y_2, Z_2), 2)
        #         print("ap_3 shape",ap_3.shape)
        #         print("Y_3 shape",Y_3.shape)
        #         print("Z_3 shape",Z_3.shape)
        #         before_avg = torch.mul(Y_3, Z_3)
        #         print("beforeavg shape",before_avg.shape)
        avg = ap_3.mean(dim=1)
        #         print("avg shape",avg.shape)
        #         result = self.sigmoid(avg).squeeze(dim=-1)
        return avg


if __name__ == '__main__':
    torch.manual_seed(1000)
    np.random.seed(1000)
    parser = argparse.ArgumentParser()
    #     parser.add_argument('--backbone_sampling_type', type=str, help='Type of the backbone sampling',
    #                         choices=['rus', 'ros','oss','smote'], default='smote')
    parser.add_argument('--model_type', type=str, help='Type of the model (devign/ggnn)',
                        choices=['devign', 'ggnn'], default='devign')
    parser.add_argument('--model_state_dir', type=str, required=True, help='Dir of the model bin file')
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset for experiment.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input Directory of the parser')
    parser.add_argument('--node_tag', type=str, help='Name of the node feature.', default='node_features')
    parser.add_argument('--graph_tag', type=str, help='Name of the graph feature.', default='graph')
    parser.add_argument('--label_tag', type=str, help='Name of the label feature.', default='target')
    parser.add_argument('--data_split', type=str, default='1')
    parser.add_argument('--feature_size', type=int, help='Size of feature vector for each node', default=100)
    parser.add_argument('--graph_embed_size', type=int, help='Size of the Graph Embedding', default=200)
    parser.add_argument('--num_steps', type=int, help='Number of steps in GGNN', default=6)
    parser.add_argument('--batch_size', type=int, help='Batch Size for training', default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    if torch.cuda.device_count() > 1:
        if int(args.data_split) % 2 == 0:
            args.device = 'cuda:0'
        else:
            args.device = 'cuda:1'
    print(f'running split {args.data_split} on gpu {args.device}\n')
    LOG_FORMAT = "%(asctime)s - %(message)s"
    logging.basicConfig(filename=f'{args.dataset}_result/backbone_{args.model_type}_{args.data_split}.log',
                        level=logging.INFO, format=LOG_FORMAT)
    if args.feature_size > args.graph_embed_size:
        print('Warning!!! Graph Embed dimension should be at least equal to the feature dimension.\n'
              'Setting graph embedding size to feature size', file=sys.stderr)
        args.graph_embed_size = args.feature_size

    #     model_dir = os.path.join('models', f'{args.model_type}_model', args.dataset, args.data_split)
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    input_dir = args.input_dir
    processed_data_path = os.path.join(input_dir, 'processed.bin')
    print(processed_data_path)
    if os.path.exists(processed_data_path):
        debug('Reading already processed data from %s!' % processed_data_path)
        #         logging.info('Reading already processed data from %s!' % processed_data_path)
        dataset = pickle.load(open(processed_data_path, 'rb'))
        debug(len(dataset.train_examples), len(dataset.valid_examples), len(dataset.test_examples))
    #         logging.info(f'{len(dataset.train_examples)}, {len(dataset.valid_examples)}, {len(dataset.test_examples)}')
    else:
        debug('ERROR require processed bin file!')
        exit()
    #     dataset.batch_size = args.batch_size
    print('dataset batch size:', dataset.batch_size)
    # create model instance
    if args.model_type == 'ggnn':
        debug('model: GGNN')
        #         logging.info('model: GGNN')
        model = GGNNSum(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                        num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        model.load_state_dict(torch.load(args.model_state_dir, map_location='cuda:0'))
        my_model = GGNNSum_MLP(model)
    else:
        debug('model: Devign')
        #         logging.info('model: Devign')
        model = DevignModel(input_dim=dataset.feature_size, output_dim=args.graph_embed_size,
                            num_steps=args.num_steps, max_edge_types=dataset.max_edge_type)
        model.load_state_dict(torch.load(args.model_state_dir, map_location='cuda:0'))
        my_model = DevignModel_MLP(model)

    debug('Total Parameters : %d' % tally_param(my_model))
    #     logging.info('Total Parameters : %d' % tally_param(model))
    # model.cuda()
    my_model.to(args.device)
    device = args.device

    #     if args.model_type == 'devign':
    if True:
        with torch.no_grad():
            my_model.eval()
            trainX_np_array = list()
            trainy_list = []
            train_batch_len = dataset.initialize_train_batch()
            # train
            for i in tqdm(range(train_batch_len), desc=f'trainset'):
                graph, targets = dataset.get_next_train_batch()
                #                 print(targets)
                predictions = my_model(graph, device=device)
                trainX_np_array.append(predictions.cpu().detach().numpy())
                #                 print(len(trainX_np_array))
                trainy_list.extend(targets.cpu().detach().tolist())
            trainy_np_list = np.array(trainy_list)
            trainX_np_array = np.vstack(trainX_np_array)
            print(trainX_np_array.shape)
            print(trainy_np_list.shape)
            train_X_dump_dir = f'{args.dataset}_result/backbone_{args.model_type}'
            if not os.path.exists(train_X_dump_dir):
                os.makedirs(train_X_dump_dir)
            torch.save(trainX_np_array, f'{train_X_dump_dir}/ros_{args.data_split}_trainX.pt')
            torch.save(trainy_np_list, f'{train_X_dump_dir}/ros_{args.data_split}_trainy.pt')
            # test

            testX_np_array = []
            testy_list = []
            test_batch_len = dataset.initialize_test_batch()
            for i in tqdm(range(test_batch_len), desc=f'testset'):
                graph, targets = dataset.get_next_test_batch()
                predictions = my_model(graph, device=device)
                testX_np_array.append(predictions.cpu().detach().numpy())
                testy_list.extend(targets.cpu().detach().tolist())
            testy_np_list = np.array(testy_list)
            testX_np_array = np.vstack(testX_np_array)
            print(testX_np_array.shape)
            print(testy_np_list.shape)

    print('latent training and testing phase')
    for type in ['rus', 'ros', 'oss', 'smote']:
        if type == 'rus':
            sampler = RandomUnderSampler()
        elif type == 'ros':
            sampler = RandomOverSampler()
        elif type == 'oss':
            sampler = OneSidedSelection()
        else:
            sampler = SMOTE()

        X_res, y_res = sampler.fit_resample(trainX_np_array, trainy_np_list)
        clf = MLPClassifier(max_iter=1000).fit(X_res, y_res)
        #         print(clf.score(testX_np_array, testy_np_list))
        all_predictions = clf.predict(testX_np_array)
        all_probabilities = clf.predict_proba(testX_np_array)[:, 1]
        fpr, tpr, _ = roc_curve(testy_np_list, all_probabilities)
        logging.info(f'{type}: acc: {accuracy_score(testy_np_list, all_predictions)} \
        precision: {precision_score(testy_np_list, all_predictions)} \
        recall: {recall_score(testy_np_list, all_predictions)}\
        f1: {f1_score(testy_np_list, all_predictions)} \
        auc: {auc(fpr, tpr)}')
        print(f'{type}: acc: {accuracy_score(testy_np_list, all_predictions)} \
        precision: {precision_score(testy_np_list, all_predictions)} \
        recall: {recall_score(testy_np_list, all_predictions)}\
        f1: {f1_score(testy_np_list, all_predictions)} \
        auc: {auc(fpr, tpr)}')
        zipped_result = zip(testy_np_list.tolist(), all_predictions.tolist(), all_probabilities.tolist())
        sorted_zip = sorted(zipped_result, key=lambda x: x[2], reverse=True)
        dump_dir = f'{args.dataset}_result/backbone_{args.model_type}/{type}'
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        pickle.dump(sorted_zip, open(f'{dump_dir}/zip_ans_pred_prob_{args.data_split}.pkl', "wb"))
        pickle.dump(clf, open(f'{dump_dir}/sk_model.pkl', 'wb'))






