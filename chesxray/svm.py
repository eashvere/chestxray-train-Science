import os
import sys
import datetime
import numpy as np
import glob
import pandas as pd
import io
import json
import keras
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import cv2
from tqdm import tqdm
import random
from zoo import *
import paths
from collections import defaultdict
import tensorflow as tf

random.seed(2018)
np.random.seed(2018)

EPOCHS = paths.epochs
CLASSES = paths.num_classes
PATIENCE = paths.patience
INPUT_PATH = paths.input_path
PERCENTAGE = paths.percent
MODELS_PATH = paths.model_path
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = paths.history
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)
RETRAIN_MODELS = False
RESTORE_FROM_LAST_CHECKPOINT = False
UPDATE_BEST_MODEL = False

IMAGE_ARRAY = dict()


def read_all_images(files):
    total_files = 0
    for f in tqdm(files):
        IMAGE_ARRAY[os.path.basename(f)] = cv2.resize(
            cv2.imread(f), (300, 300), cv2.INTER_LANCZOS4)
        total_files += 1

    print('Read {} images from disk'.format(total_files))


def batch_generator(files, labels, augmentation=False):
    import keras.backend as K
    global IMAGE_ARRAY

    dim_ordering = K.image_dim_ordering()

    while True:
        index = random.sample(range(len(files)), 1)
        img = files[index][0]
        # print(img)
        batch_files = []
        for i in range(len(paths.models)):
            batch_files.append(img)
        img = None
        # print(batch_files)
        batch_labels = labels[index]

        image_list = []
        for i in range(len(batch_files)):
            cnn = paths.models[i]

            input_size = get_input_shape(cnn)

            image = IMAGE_ARRAY[os.path.basename(batch_files[i])]

            if cnn == 'INCEPTION_V3' or cnn == 'INCEPTION_V4' or cnn == 'XCEPTION':
                random_border = 20
                start0 = random.randint(0, random_border)
                start1 = random.randint(0, random_border)
                end0 = random.randint(0, random_border)
                end1 = random.randint(0, random_border)
                image = image[start0:image.shape[0] -
                              end0, start1:image.shape[1] - end1]
                image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
                image = np.reshape(image, (-1, 299, 299, 3))
            else:
                box_size = random.randint(200, 256)
                start0 = random.randint(0, image.shape[0] - box_size)
                start1 = random.randint(0, image.shape[1] - box_size)
                image = image[start0:start0 +
                              box_size, start1:start1 + box_size]
                image = cv2.resize(image, input_size, cv2.INTER_LANCZOS4)
                image = np.reshape(
                    image, (-1, input_size[0], input_size[1], 3))

            image_list.append(image.astype(np.float32))
        #image_list = np.array(image_list)
        #image_list = image_list.transpose((0, 3, 1, 2))
        #image_list = preprocess_input_overall(cnn, image_list)
        # if dim_ordering == 'tf':
        #    image_list = image_list.transpose((0, 2, 3, 1))
        mask_list = np.array(batch_labels)
        yield image_list, mask_list


def train_model(x_hi, y_hi):
    from ensemble import individual_models

    models = individual_models()

    train_generator = batch_generator(x_hi, y_hi)

    x = []
    y = []
    graph = tf.get_default_graph()

    # print
    import itertools
    for i in tqdm(range(50)):
        images, label = next(train_generator)
        hi = []
        for k in range(len(images)):
            image = images[k]
            model = models[k]
            # print(image)
            with graph.as_default():
                pred = model.predict(image)
            hi.append(pred[0])
            # hi.append(pred)

        #hi = list(itertools.chain.from_iterable(hi))
        #hi = list(itertools.chain.from_iterable(hi))
        hi = np.mean(hi, axis=0)
        hi = np.array(hi)
        x.append(hi)
        hi = None
        y.append(label[0])

    print(x[0])
    print(y[0])

    x = np.array(x)
    y = np.array(y)

    print(x.shape)
    print(y.shape)

    import xgboost as xgb
    from sklearn.multioutput import MultiOutputClassifier

    # max_depth=100, n_estimators=6700, n_jobs=6, random_state=1970

    #model = xgb.XGBClassifier()
    from sklearn.svm import SVC
    model = MultiOutputClassifier(SVC())
    kfold = KFold(n_splits=5, random_state=2007)

    results = cross_val_score(model, x, y, cv=kfold)
    return results


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


def run_validation():
    df = pd.read_csv(INPUT_PATH + 'input/data.csv')
    df = df.drop(range(int(df.shape[0] * 0.55), df.shape[0]))
    deletes = []
    counter = 0
    for row in tqdm(range(len(df['classes']))):
        if df['classes'][row] == 'No Finding':
            deletes.append(row)
            counter += 1
        if counter == 33000:
            break

    df = df.drop(deletes)
    df = df.reset_index()

    labels, counts = get_labels(df)

    print(counts)

    indexes = get_indexes(df)

    print(indexes)

    files = []
    for id in df['img'].values:
        files.append(INPUT_PATH + "input/images/" + id)

    read_all_images(files)

    files = np.array(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    print('Labels shape:', lbl.shape)
    print('Files shape:', files.shape)

    score1 = train_model(files, lbl)
    print("Accuracy: %.2f%% (%.2f%%)" %
          (score1.mean() * 100, score1.std() * 100))


if __name__ == '__main__':
    run_validation()
