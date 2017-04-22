import os
import numpy as np
import pandas as pd
from PIL import Image
from models.settings import *
from keras.models import model_from_json
from models.utils import quadratic_weighted_kappa

# test data dir
test_data_dir = 'data/validation/'

# read names of images and correct labels for them
correct_data = pd.read_csv('trainLabels.csv')

# create pool of images to be evaluated
image_file_pool = []
image_name_pool = []
class_names = [name for name in os.listdir(test_data_dir)]
for class_name in class_names:
    for file in os.listdir(test_data_dir+class_name+'/'):
        if file.endswith(".jpeg"):
            image_file_pool.append(test_data_dir+class_name+'/'+file)
            image_name_pool.append(file[:-5])

# list of trained models to check
list_of_checkpoints = [ "weights-improvement-00-0.20.hdf5",
        "weights-improvement-04-0.32.hdf5",
        "weights-improvement-08-0.36.hdf5",
        "weights-improvement-15-0.51.hdf5",
        "weights-improvement-50-0.69.hdf5",
        "weights-improvement-00-0.24.hdf5",
        "weights-improvement-06-0.23.hdf5",
        "weights-improvement-09-0.37.hdf5",
        "weights-improvement-16-0.48.hdf5",
        "weights-improvement-80-0.70.hdf5",
        "weights-improvement-02-0.22.hdf5",
        "weights-improvement-06-0.41.hdf5",
        "weights-improvement-10-0.39.hdf5",
        "weights-improvement-21-0.54.hdf5",
        "weights-improvement-03-0.25.hdf5",
        "weights-improvement-07-0.24.hdf5",
        "weights-improvement-15-0.45.hdf5",
        "weights-improvement-25-0.66.hdf5"]

for checkpoint in list_of_checkpoints:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    # load model's weights
    model.load_weights(WEIGHTS_DIR + checkpoint)
    
    predicted_label = []
    correct_label = []

    # predict and store labels
    for image_idx in range(len(image_file_pool)):
        img_array = np.array(Image.open(image_file_pool[image_idx]))
        img_array = img_array[None, :]
        predicted_label.append(np.argmax(model.predict(img_array).ravel()))
        correct_label.append(correct_data['level'].loc[correct_data['image'] == image_name_pool[image_idx]].tolist()[0])

    print 'checkpoint {0}: {1:5.4f}'.format(checkpoint, quadratic_weighted_kappa(predicted_label, correct_label))
