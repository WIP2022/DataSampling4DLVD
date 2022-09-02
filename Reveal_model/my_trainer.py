import copy
from sys import stderr

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,roc_curve, auc, confusion_matrix
from tqdm import tqdm
import logging
from utils import debug
from torch import nn
from torch.utils.tensorboard import SummaryWriter
def my_evaluate_metrics(model, loss_function, num_batches, dataset, device='cuda:0'):
    if type(loss_function) is nn.BCELoss:
        logging.info('sigmoid')
        model.eval()
        with torch.no_grad():
            _loss = []
            all_predictions, all_targets = [], []
            all_probabilities = []
            for _ in tqdm(range(num_batches),desc='valid >'):
                graph, targets = dataset.get_next_valid_batch()
                targets = targets.cuda()
                predictions = model(graph, device=device)
                all_probabilities.extend(predictions.detach().cpu().tolist())
                batch_loss = loss_function(predictions, targets)
                _loss.append(batch_loss.detach().cpu().item())
                predictions = predictions.detach().cpu()
                if predictions.ndim == 2:
                    all_predictions.extend(np.argmax(predictions.numpy(), axis=-1).tolist())
                else:
                    all_predictions.extend(
                        predictions.ge(torch.ones(size=predictions.size()).fill_(0.5)).to(
                            dtype=torch.int32).numpy().tolist()
                    )
                all_targets.extend(targets.detach().cpu().numpy().tolist())
            model.train()
            print(confusion_matrix(all_targets,all_predictions))
            fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
            return accuracy_score(all_targets, all_predictions) , \
                precision_score(all_targets, all_predictions) , \
                recall_score(all_targets, all_predictions) , \
                f1_score(all_targets, all_predictions) , \
                auc(fpr, tpr)
    else:
        print('softmax')
        with torch.no_grad():
            all_predictions, all_targets = [], []
            all_probabilities = []
            for _ in tqdm(range(num_batches),desc='valid >'):
                graph, targets = dataset.get_next_valid_batch()
                # targets = targets.cuda()
                out = model(graph, device=device)
                predictions = out.argmax(dim=1).cpu().detach().numpy()
                all_predictions.extend(predictions)
                all_targets.extend(targets.numpy())
                # print(out[0])
                prob_1 = out.cpu().detach().numpy()[:,-1]
                all_probabilities.extend(prob_1)
            model.train()
            fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
            return accuracy_score(all_targets, all_predictions) , \
                precision_score(all_targets, all_predictions) , \
                recall_score(all_targets, all_predictions) , \
                f1_score(all_targets, all_predictions) , \
                auc(fpr, tpr)




def my_train(model,epochs, dataset, loss_function, optimizer, save_path,device='cuda:0'):
#     writer = SummaryWriter()
#     debug('Start Training')
#     logging.info('Start Training')
    best_model = None
    patience_counter = 0
    best_f1 = 0
    for e in range(epochs):
        train_losses = []
        model.train()
        train_batch_len = dataset.initialize_train_batch()
        for i in tqdm(range(train_batch_len),desc=f'train {e}'):
            optimizer.zero_grad()
            graph, targets = dataset.get_next_train_batch()
#             if type(loss_function) is not nn.BCELoss:
#                 targets = targets.type(torch.LongTensor)
            targets = targets.to(device)
            predictions = model(graph, device=device)
            batch_loss = loss_function(predictions, targets)
            train_losses.append(batch_loss.detach().cpu().item())
            batch_loss.backward()
            optimizer.step()
        train_loss_avg = sum(train_losses) / len(train_losses)
        debug(f'training loss in epochs {e} -> {train_loss_avg}')
        # logout every epochs, also save model
        _save_file = open(save_path + f'_ep_{e}.bin', 'wb')
        torch.save(model.state_dict(), _save_file)
        # skip eval
        # number_valid_batch = dataset.initialize_valid_batch()
        # acc, pr, rc, f1, auc_score = my_evaluate_metrics(model, loss_function, number_valid_batch, dataset, device)
#         writer.add_scalar('Loss/train', train_loss_avg, e)
#         writer.add_scalar('auc/test', auc_score, e)
#         writer.add_scalar('f1/test', f1, e)
        # debug('%s\t Epochs %d\tTest Accuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f\t AUC: %f' % (save_path, e, acc, pr, rc, f1,auc_score))
        # logging.info('%s\t Epochs %d\tTest Accuracy: %f\tPrecision: %f\tRecall: %f\tF1: %f\t AUC: %f' % (save_path, e, acc, pr, rc, f1,auc_score))
        # debug('=' * 100)
