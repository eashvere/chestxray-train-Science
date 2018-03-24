from ensemble import *
from zoo import *
from tqdm import tqdm
import cv2
import keras
import keras.backend as K
import pandas as pd
import random
import numpy as np
from paths import *
from collections import defaultdict
import os
import sys
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools

random.seed(2018)
np.random.seed(2018)

models = individual_models()
graph = tf.get_default_graph()


def preprocess_image(cnn, image):
    input_size = get_input_shape(cnn)
    image = cv2.resize(image, input_size, cv2.INTER_LANCZOS4)
    image = np.reshape(image, (-1, input_size[0], input_size[1], 3))
    image = preprocess_input_overall(cnn, image)

    return image


thres = 0.163
print('Threshold Value: {}'.format(thres))


def predict(image):
    preds = []
    for i in range(len(paths.models)):
        model = models[i]
        cnn = paths.models[i]
        img = image
        img = preprocess_image(cnn, image)

        with graph.as_default():
            prediction = model.predict(img)
        preds.append(prediction[0])
    # print(preds)
    full_pred = np.mean(preds, axis=0)
    full_pred[full_pred > thres] = 1
    full_pred[full_pred <= thres] = 0
    return full_pred


def get_labels(df):
    labels = df['classes'].apply(lambda x: x.split('|'))

    counts = defaultdict(int)
    for l in labels:
        for l2 in l:
            counts[l2] += 1

    return labels, counts


def get_indexes(df):
    labels, counts = get_labels(df)
    indexes = sorted(list(counts.keys()))
    for i in range(len(indexes)):
        df['label_{}'.format(i)] = 0

    return indexes


def training_mean_eval():
    df = pd.read_csv(paths.input_path + 'input/data.csv')
    #   df = df.drop(range(int(df.shape[0] * 0.55), df.shape[0]))
    deletes = []
    counter = 0
    for row in tqdm(range(len(df['classes']))):
        if df['classes'][row] != 'Pneumonia':
            deletes.append(row)
            counter += 1
        if counter == 33000:
            pass
            # break

    df = df.drop(deletes)
    df = df.reset_index()
    # print(df.head())
    labels, counts = get_labels(df)

    # print(labels)

    print(counts)

    indexes = get_indexes(df)

    print(indexes)

    # print(df.head())

    files = []
    for id in df['img'].values:
        files.append(paths.input_path + "input/images/" + id)

    files = np.array(files)

    print(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    test_num = np.random.choice(range(0, len(df.columns)), 500)
    #test_num = test_num.tolist()
    # test_num.append(49)
    # test_num.append(127)
    # test_num.append(254)
    # test_num.append(277)

    # print(test_num)

    #test_data = [cv2.imread(files[num]) for num in tqdm(test_num)]
    #test_label = [lbl[num] for num in test_num]

    #test_pred = [predict(img)[0] for img in tqdm(test_data)]

    test_pred = []
    test_label = []

    for ids in tqdm(test_num):
        img = cv2.imread(files[ids])
        test_pred.append(predict(img)[0])
        label_id = lbl[ids]
        test_label.append(label_id)

    test_pred = np.array(test_pred)
    test_label = np.array(test_label)

    print(test_pred[0])
    print(test_label[0])

    print(test_pred.shape)
    print(test_label.shape)

    from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score, confusion_matrix

    if False:
        multi_class_roc(y_true=test_label, y_pred=test_pred)
    else:
        # cm = confusion_matrix(y_true=test_label.flatten(),
        #                      y_pred=test_pred.flatten())
        #plot_confusion_matrix(cm, ["Positive", "Negative"], cmap=plt.cm.YlOrRd)

        print(f1_score(y_true=test_label, y_pred=test_pred, average='weighted'))

        # print(classification_report(y_true=test_label,
        #                            y_pred=test_pred, target_names=indexes))
        #print(top_k_categorical_accuracy(test_label, test_pred))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def multi_class_roc(y_true, y_pred):
    from sklearn.metrics import roc_curve, auc
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(15):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    import matplotlib.pyplot as plt

    for i in range(15):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.show()

    return 0


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


if __name__ == '__main__':
    print(training_mean_eval())
