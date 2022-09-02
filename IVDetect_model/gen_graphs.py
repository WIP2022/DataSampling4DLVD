import os
import numpy as np
from torch.nn.utils.rnn import pack_sequence
from tqdm import tqdm

import utils.process as process
import pandas as pd
import torch
from torch_geometric.data import Data
import vul_model


def generate_glove_file(data):
    print("Generating glove file")
    sample_big_data_pdg = process.collect_code_data(data)
    sample_big_data_ast = process.collect_tree_info(data)
    with open('glove/msr_pdg_word.txt', 'w') as file:
        for sentence in sample_big_data_pdg:
            for token in sentence:
                file.write(f'{token} ')
    with open('glove/msr_ast_word.txt', 'w') as file:
        for sentence in sample_big_data_ast:
            for token in sentence:
                file.write(f'{token} ')


# this function check all nodes in pdg graphs, if src and des
# all in ast nodes,then keep it, else discard
# return a new graph + mapping of old_node_index -> new_node_index
def clean_graph(_pdg_graph, _ast_nodes):
    node_list = []
    all_nodes = torch.flatten(_pdg_graph)
    for node in all_nodes:
        if node in _ast_nodes:
            if node not in node_list:
                node_list.append(node.item())
    # sort all ast nodes
    node_list.sort()
    index_list = list(range(0, len(node_list)))
    new_dict = dict(zip(node_list, index_list))
    new_src_nodes = []
    new_des_nodes = []
    for i in range(len(_pdg_graph[0])):
        src_node = _pdg_graph[0][i]
        des_node = _pdg_graph[1][i]
        # source and destination all in ast nodes
        if src_node in node_list and des_node in node_list:
            new_src_nodes.append(new_dict[src_node.item()])
            new_des_nodes.append(new_dict[des_node.item()])
    edge_index = torch.tensor([new_src_nodes, new_des_nodes], dtype=torch.long)
    return edge_index, new_dict


if __name__ == '__main__':
    data_file = "MSR_data/msr_with_vul_type.csv"
    # data = process.read_data(data_file, 100)
    data = pd.read_csv(data_file)
    print(data.info())
    # generate_glove_file(data)
    # exit()
    from gensim.test.utils import get_tmpfile
    from gensim.models import KeyedVectors
    from gensim.scripts.glove2word2vec import glove2word2vec

    embedding_size = 100

    ast_glove_file_dir = f'{os.getcwd()}/glove/msr_ast_vectors.txt'
    ast_tmp_file_dir = f'{os.getcwd()}/glove/msr_ast_gensim.txt'
    pdg_glove_file_dir = f'{os.getcwd()}/glove/msr_pdg_vectors.txt'
    pdg_tmp_file_dir = f'{os.getcwd()}/glove/msr_pdg_gensim.txt'
    ast_temp_file = get_tmpfile(ast_tmp_file_dir)
    pdg_temp_file = get_tmpfile(pdg_tmp_file_dir)
    if not os.path.isfile(ast_tmp_file_dir):
        glove2word2vec(ast_glove_file_dir, ast_temp_file)
    if not os.path.isfile(pdg_tmp_file_dir):
        glove2word2vec(pdg_glove_file_dir, pdg_temp_file)

    ast_glove_vector = KeyedVectors.load_word2vec_format(ast_temp_file)
    pdg_glove_vector = KeyedVectors.load_word2vec_format(pdg_temp_file)

    # model = vul_model.Vulnerability(h_size=100, num_node_feature=5, num_classes=2,
    #                                 feature_representation_size=100,
    #                                 drop_out_rate=0, num_conv_layers=2)
    # model.eval()
    # torch.no_grad()

    # generate graphs for torch-geometric
    pdg_graphs = process.collect_pdg(data)

    # fea 1: sub token of each line of code -> embedding -> average
    feas_1 = process.generate_feature_1(data, pdg_glove_vector, embedding_size)
    # fea 2: tree-lstm of each line of code in AST
    feas_2 = process.generate_feature_2(data, ast_glove_vector, embedding_size)
    # fea 3: variable name and variable type of each line in AST
    # feas_3 = process.generate_feature_3(data, pdg_glove_vector, 100)
    feas_3 = process.generate_feature_3(data, ast_glove_vector, embedding_size)
    # fea 4: control dependency CDG context in AST
    feas_4 = process.generate_feature_4(data, pdg_glove_vector, embedding_size)
    # fea 5: data dependency DDG context in AST
    feas_5 = process.generate_feature_5(data, pdg_glove_vector, embedding_size)

    starting_index = 0  # some graph can't proceed, use a extra index to generate and store graph
    wrong_cumulate = 1
    for i in tqdm(range(0, len(data))):
        try:
            # get each feature
            fea_1 = feas_1[i]
            fea_2 = feas_2[i]
            fea_3 = feas_3[i]
            fea_4 = feas_4[i]
            fea_5 = feas_5[i]

            valid_fea = True
            for fea_index, fea in enumerate([fea_1, fea_2, fea_3, fea_4, fea_5]):
                if fea is None:
                    valid_fea = False
                    print(f'feature {fea_index} in data {i} is None, total wrong {wrong_cumulate}')
            if not valid_fea:
                wrong_cumulate += 1
                continue

            code = data.at[i, "code"]
            loc = len(code.splitlines())
            # skip large file
            if loc > 500 or loc < 5:
                continue
            # get all nodes from ast graph
            ast_nodes = list(fea_2.keys())
            raw_pdg_graph = pdg_graphs[i]
            # clean the pdg graph, all node in pdg graph must also shown in the AST nodes
            # point exist in ast+pdg will be preserved
            new_pdg_graph, mapping = clean_graph(raw_pdg_graph, ast_nodes)
            # Create a PyG Data object
            graph_i = Data(edge_index=new_pdg_graph)
            # clean generated features
            new_fea_1 = []
            new_fea_2 = []
            new_fea_3 = []
            new_fea_4 = []
            new_fea_5 = []
            mapping_key_list = list(mapping.keys())
            if len(mapping_key_list) == 0:
                continue
            # calculate the new_pdg/ast ratio, discard if too small
            ratio = len(mapping) / len(fea_2)
            if ratio < 0.3:
                continue
            # for each line in feature, if line also in the cleaned pdg graph, add its feature to the list
            for j in range(len(mapping)):
                # get key from the mapping
                key = mapping_key_list[j]
                # get feature representation of line j
                fea1_j = fea_1[key - 1]
                if len(fea1_j) == 0:
                    new_fea_1.append(torch.from_numpy(np.stack(np.zeros(embedding_size))))
                else:
                    new_fea_1.append(torch.from_numpy(np.stack(fea1_j)))
                new_fea_2.append(fea_2[key])
                # feature 3,4,5 have different key
                key_in_str = f'{key}" '
                if key_in_str in fea_3.keys():
                    new_fea_3.append(torch.from_numpy(np.stack(fea_3[key_in_str])))
                else:
                    new_fea_3.append(torch.from_numpy(np.stack([np.zeros(embedding_size)])))
                if key_in_str in fea_4.keys():
                    fea4_j = fea_4[key_in_str]
                    if len(fea4_j) == 0:
                        new_fea_4.append(torch.from_numpy(np.stack([np.zeros(embedding_size)])))
                    else:
                        new_fea_4.append(torch.from_numpy(np.stack(fea4_j)))
                else:
                    new_fea_4.append(torch.from_numpy(np.stack([np.zeros(embedding_size)])))
                if key_in_str in fea_5.keys():
                    fea5_j = fea_5[key_in_str]
                    if len(fea5_j) == 0:
                        new_fea_5.append(torch.from_numpy(np.stack([np.zeros(embedding_size)])))
                    else:
                        new_fea_5.append(torch.from_numpy(np.stack(fea5_j)))
                else:
                    new_fea_5.append(torch.from_numpy(np.stack([np.zeros(embedding_size)])))
            # padding
            new_fea_1 = pack_sequence(new_fea_1, enforce_sorted=False)
            new_fea_3 = pack_sequence(new_fea_3, enforce_sorted=False)
            new_fea_4 = pack_sequence(new_fea_4, enforce_sorted=False)
            new_fea_5 = pack_sequence(new_fea_5, enforce_sorted=False)
            #  now all features have the same number of lines, format: [lines,sequences,embedding]
            graph_i.my_data = [new_fea_1, new_fea_2, new_fea_3, new_fea_4, new_fea_5]
            vul = data.at[i, "bug"]
            graph_i.y = torch.tensor([vul], dtype=int)
            graph_i.code = code
            graph_i.cve_type = data.at[i, "cve"]
            graph_i.cwe_type = data.at[i, "cwe"]

            # valid data by passing to sample model
            # model(graph_i.my_data,graph_i.edge_index)

            # no train-test set here, save each data point one by one
            torch.save(graph_i, f'{os.getcwd()}/MSR_data/pyg_graph/data_{starting_index}.pt')
            starting_index += 1
            # run train_test_valid.py to split the graphs
        except:
            print(f'error in {i}')