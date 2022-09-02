from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    classification_report, confusion_matrix
import numpy as np
from sklearn.neural_network import MLPClassifier
# from autosklearn.classification import AutoSklearnClassifier as ASC
# from autosklearn.metrics import balanced_accuracy, precision, recall, f1
# import autoPyTorch
# from autoPyTorch.api.tabular_classification import TabularClassificationTask
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    classification_report, confusion_matrix
# from autosklearn.classification import AutoSklearnClassifier as ASC
# from autosklearn.metrics import balanced_accuracy, precision, recall, f1
# import autoPyTorch
# from autoPyTorch.api.tabular_classification import TabularClassificationTask
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.neural_network import MLPClassifier
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def calculate_backbone_metric(trainX, trainy, testX, testy, sampling_method):
    if sampling_method == 'smote':
        smote = SMOTE(n_jobs=-1, random_state=42)
        trainX_res, trainy_res = smote.fit_resample(trainX, trainy)
    elif sampling_method == 'rus':
        rus = RandomUnderSampler(random_state=42)
        trainX_res, trainy_res = rus.fit_resample(trainX, trainy)
    elif sampling_method == 'ros':
        ros = RandomOverSampler(random_state=42)
        trainX_res, trainy_res = ros.fit_resample(trainX, trainy)
    elif sampling_method == 'oss':
        oss = OneSidedSelection(n_jobs=-1, random_state=42)
        trainX_res, trainy_res = oss.fit_resample(trainX, trainy)

    clf = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(768, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    clf.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    clf.fit(trainX_res, trainy_res, batch_size=128, epochs=100, verbose=0)
    print(clf.evaluate(testX, testy))
    pred_prob = clf.predict(testX)
    # print(pred_prob)
    asign = lambda t: 0 if t < 0.5 else 1
    pred = list(map(asign, pred_prob))
    # print(pred)
    fpr, tpr, _ = roc_curve(testy, pred_prob)
    auc_score = auc(fpr, tpr) * 100
    f1 = f1_score(testy, pred, zero_division=0) * 100
    recall = recall_score(testy, pred, zero_division=0) * 100
    precision = precision_score(testy, pred, zero_division=0) * 100
    acc = accuracy_score(testy, pred) * 100
    print(classification_report(testy, pred))
    print(confusion_matrix(testy, pred))
    print(f'auc: {auc_score} acc: {acc} precision: {precision} recall: {recall} f1: {f1}')
    zipped_result = zip(testy, pred, pred_prob)
    sorted_zip = sorted(zipped_result, key=lambda x: x[2], reverse=True)
    return sorted_zip


for i in range(20):
    train_X = np.load(f'devign_output/latent/data_split_{i}/train_X.npy')
    train_y = np.load(f'devign_output/latent/data_split_{i}/train_y.npy')
    test_X = np.load(f'devign_output/latent/data_split_{i}/test_X.npy')
    test_y = np.load(f'devign_output/latent/data_split_{i}/test_y.npy')
    for sampling_method in ['smote', 'rus', 'ros', 'oss']:
        sorted_zip = calculate_backbone_metric(train_X, train_y, test_X, test_y, sampling_method=sampling_method)
        with open(f'devign_output/latent/data_split_{i}/{sampling_method}_sorted_zip.pkl', 'wb') as f:
            pickle.dump(sorted_zip, f)
