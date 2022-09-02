# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import sys
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, auc, confusion_matrix


def read_answers(filename):
    answers = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            answers[js['idx']] = js['target']
    return answers


def read_predictions(filename):
    predictions = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions[int(idx)] = int(label)
    return predictions


def read_predictions_prob(filename):
    predictions_prob = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            idx, label = line.split()
            predictions_prob[int(idx)] = float(label)
    return predictions_prob


def calculate_scores(answers, predictions, predictions_prob):
    Acc = []
    Ans = []
    Pred = []
    Pred_prob = []
    for key in answers:
        Ans.append(answers[key])
        if key not in predictions:
            logging.error("Missing prediction for index {}.".format(key))
            sys.exit()
        Acc.append(answers[key] == predictions[key])
    for key in predictions:
        Pred.append(predictions[key])
    for key in predictions_prob:
        Pred_prob.append(predictions_prob[key])
    scores = {}
    scores['Acc'] = np.mean(Acc)
    fpr, tpr, _ = roc_curve(Ans, Pred_prob)
    print('auc\t', auc(fpr, tpr))
    print('acc\t', accuracy_score(Ans, Pred))
    print('f1\t', f1_score(Ans, Pred))
    print('recall\t', recall_score(Ans, Pred))
    print('precision\t', precision_score(Ans, Pred))
    return scores


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate leaderboard predictions for Defect Detection dataset.')
    parser.add_argument('--answers', '-a', help="filename of the labels, in txt format.")
    parser.add_argument('--predictions', '-p', help="filename of the leaderboard predictions, in txt format.")
    parser.add_argument('--predictions_prob', '-b', help="filename of the leaderboard predictions prob, in txt format.")

    args = parser.parse_args()
    answers = read_answers(args.answers)
    predictions = read_predictions(args.predictions)
    predictions_prob = read_predictions_prob(args.predictions_prob)
    scores = calculate_scores(answers, predictions, predictions_prob)
    print(scores)


if __name__ == '__main__':
    main()
