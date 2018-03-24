# Train an Ensemble Model Based on https://github.com/ZFTurbo/Kaggle-Planet-Understanding-the-Amazon-from-Space

# Train 11 different models on all 13 classes

# Train a model for every classes

# Densenet repository https://github.com/flyyufelix/DenseNet-Keras

# Resnet repository https://github.com/raghakot/keras-resnet

import os
import sys
import datetime
import numpy as np
import glob
import pandas as pd
import io
import json
import keras
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
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
MODELS_PATH = paths.model_single_path
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH[:-1])
HISTORY_FOLDER_PATH = paths.history_single
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH[:-1])
RETRAIN_MODELS = True
RESTORE_FROM_LAST_CHECKPOINT = False
UPDATE_BEST_MODEL = False

IMAGE_ARRAY = dict()


def read_all_images(files):
    total_files = 0
    for f in tqdm(files):
        IMAGE_ARRAY[os.path.basename(f)] = cv2.resize(
            cv2.imread(f), (300, 300), cv2.INTER_AREA)
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
            # image = cv2.imread(batch_files[i])
            image = IMAGE_ARRAY[os.path.basename(batch_files[i])]

            '''if cnn == 'INCEPTION_V3' or cnn == 'INCEPTION_V4' or cnn == 'XCEPTION':
                # random_border = 20
                # start0 = random.randint(0, random_border)
                # start1 = random.randint(0, random_border)
                # end0 = random.randint(0, random_border)
                # end1 = random.randint(0, random_border)
                # image = image[start0:image.shape[0] -
                #              end0, start1:image.shape[1] - end1]
                # image = cv2.resize(image, (299, 299), cv2.INTER_LANCZOS4)
                print('hi')
            else:
                # box_size = random.randint(200, 256)
                # start0 = random.randint(0, image.shape[0] - box_size)
                # start1 = random.randint(0, image.shape[1] - box_size)
                # image = image[start0:start0 +
                #              box_size, start1:start1 + box_size]
                # image = cv2.resize(image, input_size, cv2.INTER_LANCZOS4)
                print('else hi')'''
            image = cv2.resize(image, (256, 256), cv2.INTER_AREA)

            image = image.astype(np.float32)

            if augmentation:
                # mirroring and poisson noise and slight angle change
                # (in total there are only 4 possible configurations not including angle change)
                mirror = random.randint(0, 1)
                if mirror == 1:
                    image = cv2.flip(image, 0)
                '''noise = random.randint(0, 1)
                if noise == 1:
                    image = noisy('poisson', image)'''
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


def train_model(cnn, label, train_x, test_x, train_y, test_y):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    print('Creating {}'.format(cnn))

    print('Length of Training images:', len(train_x))
    print('Length of Training labels:', len(train_y))
    print('Length of Testing images:', len(test_x))
    print('Length of Testing labels:', len(test_y))

    model = get_pretrained_model(
        cnn, classes_number=2, learning_rate=0.001, final_layer_activation='softmax')

    final_model_path = MODELS_PATH + '{}_{}_best.h5'.format(cnn, label)
    temp_model_path = MODELS_PATH + '{}_{}_temp.h5'.format(cnn, label)

    if os.path.isfile(final_model_path) and RETRAIN_MODELS == False:
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

    batch_size = get_batch_size(cnn)
    # batch_size = 4

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
        TensorBoard(log_dir=INPUT_PATH + 'logs/{}'.format(label),
                    write_graph=True, write_images=True)
        #,EarlyStopping(monitor='val_loss', min_delta=0.01, patience=10)
    ]

    history = model.fit_generator(generator=batch_generator(cnn, train_x, train_y, True),
                                  validation_data=batch_generator(
                                      cnn, test_x, test_y, True),
                                  epochs=75,
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


def run_validation():
    df = pd.read_csv(INPUT_PATH + 'input/data.csv')
    # df = df.drop(range(int(df.shape[0] * 0.55), df.shape[0]))
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
    # print(df.head())
    labels, counts = get_labels(df)

    # print(labels)

    print(counts)

    indexes = get_indexes(df)

    print(indexes)

    # print(df.head())

    # print(files)

    # print(len(df))
    # print(len(labels))

    cnn = 'RESNET'

    for label in ['Pneumonia']:
        # if label == 'Hernia' or label == 'No Finding':
        #    continue

        print('Training Single Model Classifier on {}'.format(label))

        files_ther = []
        files_no = []
        lbl = []
        counter = 0
        for ids, ther in zip(df['img'].values, df['classes'].values):
            if counter == 700:
                break
            if label in ther:
                files_ther.append(INPUT_PATH + "input/images/" + ids)
                lbl.append(1)
            #    counter += 1
            '''else:
                files_no.append(INPUT_PATH + "input/images/" + ids)'''

        counter = 0
        bye = len(files_ther)
        for ids, ther in zip(df['img'].values, df['classes'].values):
            if counter == bye:
                break
            if 'No Finding' == ther:
                files_ther.append(INPUT_PATH + 'input/images/' + ids)
                lbl.append(2)
                counter += 1
            '''else:
                files_no.append(INPUT_PATH + "input/images/" + ids)'''

        print(files_ther[0], lbl[0])
        print(files_ther[2], lbl[2])
        print(files_ther[117], lbl[117])
        print(files_ther[190], lbl[190])

        '''no = random.sample(range(0, len(files_no) - 1), len(files_ther))

        for i in no:
            files_ther.append(files_no[i])
            lbl.append(2)'''

        '''c = list(zip(files_ther, lbl))
        print(lbl[0])

        random.shuffle(c)

        files_ther, lbl = zip(*c)'''

        files = np.array(files_ther)
        lbl = np.array(lbl)

        lblb = sklearn.preprocessing.OneHotEncoder(sparse=False)
        lbl = lbl.reshape(len(lbl), 1)
        lbl = lblb.fit_transform(lbl)

        print("hi", lbl[0])

        read_all_images(files)

        print('Labels shape:', lbl.shape)
        print('Files shape:', files.shape)

        train_x, test_x, train_y, test_y = train_test_split(
            files, lbl, test_size=0.1, random_state=get_random_state(cnn))
        # print(len(train_x))
        # print(len(train_y))
        score1 = train_model(cnn, label, train_x=train_x,
                             train_y=train_y, test_x=test_x, test_y=test_y)


if __name__ == '__main__':
    run_validation()
