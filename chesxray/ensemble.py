import os
import io
import keras
import sklearn
import cv2
from statistics import mean
from tkinter import filedialog
from tkinter import *

from zoo import *
from paths import *

keras.backend.set_learning_phase(0)


def get_model(cnn):
    from keras.layers.core import Reshape
    model = get_pretrained_model(cnn, num_classes)
    if os.path.isfile(os.getcwd() + '/models/' + cnn + '_best.h5'):
        model.load_weights(os.getcwd() + '/models/' + cnn + '_best.h5')
    elif os.path.isfile(os.getcwd() + '/models/' + cnn + '_temp.h5'):
        model.load_weights(os.getcwd() + '/models/' + cnn + '_temp.h5')
    else:
        return model

    return model


def get_model_single(cnn, label):
    from keras.layers.core import Reshape
    model = get_pretrained_model(cnn, 2)
    if os.path.isfile(os.getcwd() + '/models-single/' + cnn + '_' + label + '_best.h5'):
        model.load_weights(os.getcwd() + '/models-single/' +
                           cnn + '_' + label + '_best.h5')
    elif os.path.isfile(os.getcwd() + '/models-single/' + cnn + '_' + label + '_temp.h5'):
        model.load_weights(os.getcwd() + '/models-single/' +
                           cnn + '_' + label + '_temp.h5')
    else:
        return model

    return model


def get_all_input_shape(models):
    return [get_input_shape(model) for model in models]


def voting_classifier(models):
    from sklearn.ensemble import VotingClassifier
    estimators = [(model, get_model(model)) for model in models]

    vc = VotingClassifier(estimators=estimators, voting='soft')

    return vc


def averaging_model(models):
    from keras.layers import average, concatenate, Input
    from keras.models import Model
    input_test = Input(shape=(None, None, 3))
    inputs = []
    outputs = []
    for model in models:
        hi = get_input_shape(model)
        inp = Input(shape=(hi[0], hi[1], 3))
        cnn = get_model(model)
        out = cnn(inp)
        inputs.append(inp)
        outputs.append(out)

    x = average(outputs)

    model = Model(inputs, outputs)

    return model


def individual_models():
    models = []
    for inp in paths.models:
        print("Model being loaded..", inp)
        model = get_model(inp)
        print("Model {} has been loaded.".format(inp))
        models.append(model)
    return models


def predict_some_stuff():
    thres = 0.163
    inputs = []
    #root = Tk()
    # root.img_path = filedialog.askopenfilename(initialdir=os.getcwd(
    #), title="Select file", filetypes=(("png files", "*.png"), ("all files", "*.*")))
    pth = os.getcwd() + '/input/images/00000061_015.png'
    print(pth)
    for inp in paths.models:
        model = get_model(inp)
        hi = get_input_shape(inp)
        #img = cv2.resize(cv2.imread(root.img_path), hi, cv2.INTER_LANCZOS4)
        img = cv2.resize(cv2.imread(pth), hi, cv2.INTER_AREA)
        img = np.reshape(img, (-1, hi[0], hi[1], 3))
        img = preprocess_input_overall(inp, img)
        pred = model.predict(img)
        #print("Chicken ", pred)
        inputs.append(pred)

    resp = []
    indexes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
               'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
    print(inputs)
    pred = np.mean(inputs, axis=0)
    pred = pred[0]
    # print(pred)
    pred[pred > thres] = 1
    pred[pred <= thres] = 0
    for val in range(0, len(pred)):
        if pred[val] == 1:
            resp.append(indexes[val])

    return resp


if __name__ == '__main__':
    print(paths.models)
    print(os.getcwd())
    #model = averaging_model(paths.models)

    predict_some_stuff()
