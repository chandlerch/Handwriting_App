#---------Imports
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os
import cv2
import pandas as pd
import numpy as np
#---------End of imports

### FUNCTIONS ###
def get_data_paths():
    ''' returns dictionary of pathnames corresponding to images of the characters '''
    
    digits = [chr(i) for i in range(48, 58)]
    uppercase = [chr(i) for i in range(65, 91)]
    lowercase = [chr(i) for i in range(97, 123)]
    characters = digits + uppercase + lowercase

    paths = {characters[i] : [] for i in range(len(characters))}

    for i in range(1, 63):
        filepath = "data/English/Hnd/Img"
        files = [os.path.join(filepath, f"Sample{str(i).zfill(3)}/img{str(i).zfill(3)}-{str(j).zfill(3)}.png") for j in range(1, 56)]
        paths[characters[i - 1]] = files

    return paths

def get_data(paths):
    ''' load the MNIST dataset '''
    width, length = (900, 1200)
    df = pd.DataFrame(columns=[f'pixel{i}' for i in range(1, width*length + 1)])

    for key, value in paths.items():
        for i in range(len(value)):
            image = cv2.imread(value[i])
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to grayscale
            image_2d = np.reshape(gray, (gray.shape[0], gray.shape[1])) # resize to 2D array
            df.loc[len(df)] = list(image_2d.flatten())
            print(value)
    
    print(df)


    # test = os.path.join("data/English/Hnd/Img", f"Sample{'1'.zfill(3)}/img{'1'.zfill(3)}-{'2'.zfill(3)}.png")
    # image = cv2.imread(test)
    # print(image.flatten())
    # print(image.shape)

    # mnist = fetch_openml('mnist_784', version=1) # load data
    # data, labels = mnist['data'], mnist['target'] # get data and labels

    # # print('Sucessfully loaded data')
    # print(data[:60000])
    # print(type(data))
    # print(labels[:60000])
    # print(type(labels))


    # return (data[:60000], data[60000:], labels[:60000], labels[60000:])
    return None

def save_model(model, name=''):
    ''' saves the model into a .pckl file '''

    name = 'Chars74K_random_forest_classifier.pckl' if not name else name

    with open(name, 'wb') as f:
        pickle.dump(model, f)

### MAIN FLOW ###
if __name__ == '__main__':
    paths = get_data_paths()
    data_train, data_test, labels_train, labels_test = get_data(paths) # get mnist data

    # clf = RandomForestClassifier(random_state=42) # initialize classifier
    # clf.fit(data_train, labels_train) # train classifier
    # print('Training Random Forest Classifier ...')

    # # evaluate model
    # y_pred = clf.predict(data_test)
    # score = accuracy_score(labels_test, y_pred)
    # print(f'Test accuracy: {score}')

    # save_model(clf)
