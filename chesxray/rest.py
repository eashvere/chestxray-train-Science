import cv2
import flask
import io
import numpy as np
import tensorflow as tf

from zoo import *
from ensemble import *

app = flask.Flask(__name__)
models = None

def load_model():
    global models
    global graph = tf.get_default_graph()
    models = individual_models()


def prepare_image(image, model):
    input_size = get_input_shape(model)
    image = cv2.resize(image, input_size, cv2.INTER_LANCZOS4)
    img = np.reshape(img,(-1, input_size[0], input_size[1], 3))
    image = np.expand_dims(image, axis=0)

    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    indexes = {'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2,
     'Edema': 3, 'Effusion': 4, 'Emphysema': 5,
    'Fibrosis': 6, 'Hernia': 7, 'Infiltration': 8, 'Mass': 9, 'No Finding': 10,
     'Nodule': 11, 'Pleural_Thickening': 12,
    'Pneumonia': 13, 'Pneumothorax': 14}

    inv_label_map = {i: l for l, i in indexes.items()}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            stream = io.BytesIO(image)

            buff = numpy.fromstring(stream.getvalue(), dtype=numpy.uint8)

            #Now creates an OpenCV image
            image = cv2.imdecode(buff, 1)

            # preprocess the image and prepare it for classification
            image = prepare_image(image)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = [model.predict(image) for model in models]
            preds = np.mean(preds, axis=0)
            preds = preds>0.08

            data["predictions"] = []

            # loop over the results and add them to the list of
            # returned predictions
            for index in range(15):
                label = None
                if(preds[index]):
                    label = inv_label_map[index]
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run(debug=True)
