from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from collections import Counter
import os
from werkzeug.utils import secure_filename
import pickle
import glob
import shutil

# The folder where the user uploaded file will be temporary stored and
# below it the allowed extensions
UPLOAD_FOLDER = 'static/uploaded'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# check for the allowed file formats
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    # There is a bug in keras 2.2+ versions. When  a model is loaded the structure has to be
    # re-created manually in order to load the weights without an error.
    # So, we do the same as in CNN.py and then load the weights which contain the saved features
    vgg_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # model will be accessed in convert_image() so we need it to be global
    global model
    model = Sequential()
    for layer in vgg_model.layers:
        layer.trainable = False
        model.add(layer)

    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(1000, activation="softmax"))

    # re-load the model's trained weights
    model.load_weights("model_weights.h5")
    # model._make_predict_function() # - fix for keras 2.2.5 and tf 1.1.5

    # model.summary()


# load the model
get_model()


def normalise_img(img_data):
    # load an image file to test, resizing it to 64x64 pixels (as required by this model)
    img = image.load_img(img_data, target_size=(128, 128))

    # convert the image to a numpy array
    image_array = image.img_to_array(img)

    # add a forth dimension to the image (since Keras expects a bunch of images, not a single image)
    images = np.expand_dims(image_array, axis=0)

    # normalize the data
    images = vgg16.preprocess_input(images)

    results = model.predict(images)
    return results


def output_predicted_values(received_data):
    # Create new dictionary using the .txt file
    data_class_indices = {}
    with open("class_labels.txt") as f:
        for line in f:
            key, val = line.strip().split(':', 1)
            data_class_indices[key] = val.strip()

    class_name_list = []
    prediction_list = []

    classification_results = normalise_img(received_data)[0, :]

    # get the top 2 predicted keys
    best_classes = (-classification_results).argsort()[:3]

    # sort the array in descending order
    classification_results[::-1].sort()

    # convert the list of integers to a list of strings
    best_classes_names = [str(i) for i in best_classes]

    counter = 0
    # add the top predicted values to the prediction_list
    while counter < len(best_classes_names):
        prediction_list.append(classification_results.flat[counter] * 100)
        counter += 1

    # add the names of the top predicted values to the class_name_list
    for code in best_classes_names:
        class_name_list.append(data_class_indices[code])

    # format the lists
    # replace capitalize() with title() to capitalize every word
    formatted_prediction_list = ['%.2f' % elem for elem in prediction_list]
    capitalised_class_name_list = [elem.title() for elem in class_name_list]

    # print(capitalised_class_name_list)
    # print(formatted_prediction_list)

    # create strings for each prediction, which will be send to the browser
    result1 = capitalised_class_name_list[0] + ": " + formatted_prediction_list[0] + "%"
    result2 = capitalised_class_name_list[1] + ": " + formatted_prediction_list[1] + "%"
    result3 = capitalised_class_name_list[2] + ": " + formatted_prediction_list[2] + "%"

    # print(result1)
    # print(result2)
    return '{} {} {} {} {}'.format(result1, "", result2,  "",  result3)


# check if the directory is empty and delete the files in it if it is not
def delete_folder_contents():
    if len(os.listdir(UPLOAD_FOLDER)) == 0:
        print("")
    else:
        for img_name in os.listdir(UPLOAD_FOLDER):
            img_path = os.path.join(UPLOAD_FOLDER, img_name)
            if os.path.isdir(img_path):
                shutil.rmtree(img_path)


# the home page for the app
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/make_predictions", methods=["GET", "POST"])
def make_predictions():
    data_received = ""

    # requests the file and saves it to the upload_folder
    if request.method == "POST":
        # check if the directory is empty and delete the files in it if it is not
        delete_folder_contents()

        # save the file
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # navigate to the upload_folder, grab the extension of the file,
        # and rename the file to TempFile
        ext = ""
        for file_names in os.listdir(UPLOAD_FOLDER):
            ext = os.path.splitext(file_names)[-1].lower()
            new_file_name = "TempFile" + ext
            src = UPLOAD_FOLDER + "/" + file_names
            new_file_name = UPLOAD_FOLDER + "/" + new_file_name
            os.rename(src, new_file_name)

            # return redirect(url_for('make_predictions',
            #                         filename=filename))


    img_path = UPLOAD_FOLDER + "/TempFile" + ext
    # get the first (highest) prediction value
    result = output_predicted_values(img_path)
    # delete the TempFile
    os.remove(img_path)
    return result

