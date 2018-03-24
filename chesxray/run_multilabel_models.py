# Train 11 different models on all 13 classes

# Train a model for every classes

# Densenet repository https://github.com/flyyufelix/DenseNet-Keras

# Resnet repository https://github.com/raghakot/keras-resnet

import re
import os
import sys
import datetime
import numpy as np
import glob
import pandas as pd
pd.options.mode.chained_assignment = None
import io
import json
import keras
import itertools
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
tqdm.monitor_interval = 0
import random
from zoo import *
import paths
from collections import defaultdict

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


def batch_generator(cnn, files, labels, augmentation=False):
    import keras.backend as K
    global IMAGE_ARRAY

    dim_ordering = K.image_dim_ordering()
    input_size = get_input_shape(cnn)
    batch_size = get_batch_size(cnn)

    while True:
        index = random.sample(range(len(files)), batch_size)
        batch_files = files[index]
        batch_labels = labels[index]

        image_list = []
        mask_list = []
        for i in range(len(batch_files)):
            # print(batch_files)
            image = cv2.imread(batch_files[i])
            #image = IMAGE_ARRAY[os.path.basename(batch_files[i])]

            '''if cnn == 'INCEPTION_V3' or cnn == 'INCEPTION_V4' or cnn == 'XCEPTION':
                # random_border = 20
                # start0 = random.randint(0, random_border)
                # start1 = random.randint(0, random_border)
                # end0 = random.randint(0, random_border)
                # end1 = random.randint(0, random_border)
                # image = image[start0:image.shape[0] -
                #              end0, start1:image.shape[1] - end1]
                #image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
                print('hi')
            else:
                # box_size = random.randint(200, 256)
                # start0 = random.randint(0, image.shape[0] - box_size)
                # start1 = random.randint(0, image.shape[1] - box_size)
                # image = image[start0:start0 +
                #              box_size, start1:start1 + box_size]
                #image = cv2.resize(image, input_size, cv2.INTER_LANCZOS4)
                print('else hi')'''
            image = cv2.resize(image, (256, 256), cv2.INTER_LANCZOS4)

            image = image.astype(np.float32)

            if augmentation:
                # mirroring and poisson noise and slight angle change
                # (in total there are only 4 possible configurations not including angle change)
                mirror = random.randint(0, 1)
                if mirror == 1:
                    image = cv2.flip(image, 0)
                noise = random.randint(0, 1)
                if noise == 1:
                    image = noisy('poisson', image)
                # angle = random.randint(0, 3)
                # if angle != 0:
                #    image = np.rot90(image, k=angle)
                rotate = random.randint(0, 1)
                if rotate == 1:
                    angle = random.randint(-10, 10)
                    image_center = tuple(np.array(image.shape[1::-1]) / 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
                    image = cv2.warpAffine(
                        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

                # image = random_intensity_change(image, 10)

            # print(batch_labels)
            mask = batch_labels[i]
            image_list.append(image)
            mask_list.append(mask)
        image_list = np.array(image_list)
        image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_input_overall(cnn, image_list)
        if dim_ordering == 'tf':
            image_list = image_list.transpose((0, 2, 3, 1))
        mask_list = np.array(mask_list)
        yield image_list, mask_list


def train_model(cnn, train_x, test_x, train_y, test_y, class_weights=None):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    print('Creating {}'.format(cnn))

    print('Length of Training images:', len(train_x))
    print('Length of Training labels:', len(train_y))
    print('Length of Testing images:', len(test_x))
    print('Length of Testing labels:', len(test_y))

    model = get_pretrained_model(
        cnn, CLASSES, learning_rate=0.001, class_weights=class_weights)

    final_model_path = MODELS_PATH + '{}_best.h5'.format(cnn)
    temp_model_path = MODELS_PATH + '{}_temp.h5'.format(cnn)

    if (os.path.isfile(final_model_path) and RETRAIN_MODELS == False):
        print('Model already created... Skipping')
        return 0
    if os.path.isfile(temp_model_path) and RESTORE_FROM_LAST_CHECKPOINT:
        print('Load model from last point: ', temp_model_path)
        model.load_weights(temp_model_path)
    elif os.path.isfile(final_model_path) and UPDATE_BEST_MODEL:
        print('Load model from best point: ', final_model_path)
        model.load_weights(final_model_path)
    else:
        print('Start training with loading weights')

    print("Learning Rate: {}".format(get_learning_rate(cnn)))

    print("Batch Size: {}".format(get_batch_size(cnn)))

    #batch_size = get_batch_size(cnn)
    batch_size = 20

    training_steps = len(train_x) // batch_size
    print(training_steps)
    validation_steps = len(train_y) // batch_size

    print('Steps per Epoch: {}, Validation samples per Epoch: {}'.format(
        training_steps, validation_steps))

    callbacks = [
        ModelCheckpoint(temp_model_path, monitor='val_loss',
                        save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=PATIENCE // 2),
        TensorBoard(log_dir=INPUT_PATH + 'logs/' + cnn,
                    write_graph=True, write_images=True)
    ]

    history = model.fit_generator(generator=batch_generator(cnn, train_x, train_y, True),
                                  validation_data=batch_generator(
                                      cnn, test_x, test_y, True),
                                  epochs=EPOCHS,
                                  verbose=1, callbacks=callbacks,
                                  steps_per_epoch=training_steps,
                                  validation_steps=validation_steps,
                                  max_queue_size=300)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given model: ', min_loss)
    model.load_weights(temp_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{:.4f}_lr_{}_{}.csv'.format(
        cnn, min_loss, get_learning_rate(cnn), now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    return min_loss


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


def calculating_class_weights(label):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(label)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0., 1.], label[:, i])
    return weights


def undersample_cool(df, classs, limit=None):
    deletes = []
    counter = 0
    for row in tqdm(range(len(df['classes']))):
        if classs in df['classes'][row]:
            deletes.append(row)
            if limit is not None:
                counter += 1
        if limit is not None:
            if counter >= limit:
                break

    df = df.drop(deletes)
    df = df.reset_index()
    del df['index']
    return df


def undersample(df, classs, replaces='', limit=None):
    deletes = []
    counter = 0
    for row in tqdm(range(len(df['classes']))):
        if classs in df['classes'][row]:
            df['classes'][row] = df['classes'][row].replace(classs, replaces)
            df['classes'][row] = ''.join(
                ch for ch, _ in itertools.groupby(df['classes'][row]))
            df['classes'][row] = df['classes'][row].replace(
                'Efusion', 'Effusion')
            df['classes'][row] = df['classes'][row].replace('Mas', 'Mass')
            if limit is not None:
                counter += 1
        if df['classes'][row] == '|' or not df['classes'][row] or df['classes'][row] == ' ':
            deletes.append(row)
        if limit is not None:
            if counter >= limit:
                break

    df = df.drop(deletes)
    df = df.reset_index()
    del df['index']
    return df


def run_validation():
    df = pd.read_csv(INPUT_PATH + 'input/data.csv')
    df = undersample_cool(df, 'No Finding', limit=59000)
    #df = undersample_cool(df, 'Hernia')

    df = undersample(df, 'Infiltration', limit=17500)
    df = undersample(df, 'Cardiomegaly', limit=500)
    df = undersample(df, 'Effusion', limit=12000)
    df = undersample(df, 'Atelectasis', limit=9500)
    df = undersample(df, 'Mass', limit=3500)
    df = undersample(df, 'Nodule', limit=4000)
    df = undersample(df, 'Consolidation', limit=2500)
    df = undersample(df, 'Pneumothorax', limit=3000)
    df = undersample(df, 'Pleural_Thickening', limit=1000)

    #df = undersample_cool(df, '')
    for row in tqdm(range(len(df['classes']))):
        if df['classes'][row][-1] == '|':
            df['classes'][row] = df['classes'][row][:-1]
        if df['classes'][row][0] == '|':
            df['classes'][row] = df['classes'][row][1:]

    deletes = []
    for row in tqdm(range(len(df['classes']))):
        if df['classes'][row] == '|' or not df['classes'][row] or df['classes'][row] == ' ':
            deletes.append(row)

    df = df.drop(deletes)
    df = df.reset_index()
    del df['index']

    df.to_csv('out.csv')

    print(df.head())

    labels, counts = get_labels(df)

    # print(labels)

    print(counts)

    indexes = get_indexes(df)

    print(indexes)

    # print(df.head())

    files = []
    for id in df['img'].values:
        files.append(INPUT_PATH + "input/images/" + id)

    # read_all_images(files)

    files = np.array(files)

    # print(files)

    lbl = np.zeros((len(labels), len(indexes)))
    for j in range(len(labels)):
        l = labels[j]
        for i in range(len(indexes)):
            if indexes[i] in l:
                lbl[j][i] = 1

    class_weights = calculating_class_weights(lbl)
    #class_weights = None
    print(class_weights)

    # print(len(df))
    # print(len(labels))
    print('Labels shape:', lbl.shape)
    print('Files shape:', files.shape)

    list1 = []

    list2 = ['INCEPTION_V3_DENSE_LAYERS', 'INCEPTION_V4', 'DENSENET_121', 'DENSENET_169', 'DENSENET_161', 'RESNET50_DENSE_LAYERS', 'RESNET101',
             'VGG16_KERAS', 'VGG19_KERAS', 'XCEPTION', 'INCEPTION_RESNET_V2']

    for cnn in ['DENSENET_121']:
        train_x, test_x, train_y, test_y = train_test_split(
            files, lbl, test_size=0.2, random_state=get_random_state(cnn))
        # print(len(train_x))
        # print(len(train_y))
        score1 = train_model(cnn, train_x=train_x,
                             train_y=train_y, test_x=test_x, test_y=test_y, class_weights=class_weights)


if __name__ == '__main__':
    run_validation()
