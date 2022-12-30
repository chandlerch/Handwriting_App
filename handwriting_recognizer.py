#---------Imports
from keras.models import load_model
import tkinter as tk
from PIL import ImageDraw, Image, ImageOps
import numpy as np
import os
import sys
import pickle
import cv2
import math
from scipy import ndimage
#---------End of imports

### CLASSES ###
class App(tk.Tk):
    def __init__(self, model):
        tk.Tk.__init__(self)
        self.model = model
        self.x = self.y = 0
        self.details = tk.StringVar()
        self.txt = ''
        self.res = []

        # configure the root window
        self.title('Recognize Handwritten Digits!')
        self.geometry('950x350')

        # Creating elements
        self.image = Image.new("RGB", (300, 300), 'black') # internal image of canvas
        self.image_draw = ImageDraw.Draw(self.image)
        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", cursor="cross") # canvas to draw on
        self.label = tk.Label(self, text="Thinking ...", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command=self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)
        self.checkbox = tk.Checkbutton(self, text='Show Probabilities', command=self.agreement_changed,
                                        variable=self.details, onvalue='agree', offvalue='')

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=tk.W)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.checkbox.grid(row=1, column=2, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        ''' clear canvas and image '''

        self.canvas.delete("all")
        self.image_draw.rectangle((0, 0, 300, 300), fill=(0, 0, 0, 0))
        self.label.configure(text="Thinking ...", font=("Helvetica", 48))
        self.txt = ''
        self.res = []

    def classify_handwriting(self):
        ''' classify the digit drawn '''
        # HWND = self.canvas.winfo_id() # get the handle of the canvas
        # self.res = predict_digit(self.image, self.model)

        # if self.res is None:
        #     self.label.configure(text="No digit detected", font=("Helvetica", 48))
        # else:

        #     self.txt = f'PREDICTED DIGIT: {np.argmax(self.res)}'

        #     if str(self.details.get()):

        #         self.txt += '\n\n\n\n\nProbabilities of Handwritten Digit Being Specified Digit:\n'

        #         for i in range(len(self.res)):
        #             self.txt += f'{i}, {int(self.res[i] * 100)}%       '

        #             if i == 4:
        #                 self.txt += '\n'

        #     self.label.configure(text=self.txt, font=("Helvetica", 20))

        # Convert the image to grayscale
        img = ImageOps.grayscale(self.image) # convert image to grayscale
        gray = np.array(img) # convert to numpy array

        # Apply thresholding to create a binary image
        thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]

        # Use morphological operations to separate the digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Use contour detection to find the digits
        cnts = cv2.findContours(opening.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        objs = [cv2.boundingRect(c) for c in cnts]
        objs.sort(key=lambda x: x[0])

        characters = ""

        # loop through all the characters
        for o in objs:
            x,y,w,h = o
            roi = thresh[y:y+h, x:x+w]

            # Use object detection to classify the digit
            prob = predict_digit(roi, self.model)

            characters += str(np.argmax(prob))
        
        self.txt = f'PREDICTED DIGIT: {characters}'
        self.label.configure(text=self.txt, font=("Helvetica", 20))

    def draw_lines(self, event):
        ''' draw figure '''
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='white', outline='') # show on canvas
        self.image_draw.ellipse((self.x - r, self.y - r, self.x + r, self.y + r), fill='white', outline=None) # internal image

    def agreement_changed(self):
        ''' add prediction scores to the displayed text '''

        if str(self.details.get()) and self.txt and self.txt[-1] != " ":
            self.txt += '\n\n\n\n\nProbabilities of Handwritten Digit Being Specified Digit:\n'

            for i in range(len(self.res)):
                self.txt += f'{i}, {int(self.res[i] * 100)}%       '

                if i == 4:
                    self.txt += '\n'

            self.label.configure(text=self.txt, font=("Helvetica", 20))



### FUNCTIONS ###
def predict_digit(img, model):
    ''' makes prediction of digit in the image using the model '''

    # img = ImageOps.grayscale(img) # convert image to grayscale
    # img = np.array(img) # convert to numpy array

    if (img.sum() == 0): # nothing drawn on canvas
        return None

    img = preprocess_image(img) # preprocess the image for better result
    res = model.predict_proba(img)[0] # predict class

    return res

def preprocess_image(img):
    ''' preprocess handwritten digit to make it suitable for classifier '''

    # remove unnecessary surrounding black rows and columns in image
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:, 0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:, -1]) == 0:
        img = np.delete(img, -1, 1)

    rows, cols = img.shape # get dimension of image

    # resize to be contained within 20 x 20 box
    if rows > cols:
        factor = 20.0/rows
        rows = 20
        cols = int(round(cols * factor))
        img = cv2.resize(img, (cols, rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows * factor))
        img = cv2.resize(img, (cols, rows))

    # pad to make 28 x 28 image
    colsPadding = (int(math.ceil((28 - cols) / 2.0)), int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)), int(math.floor((28 - rows) / 2.0)))
    img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')

    # center number using center of mass
    shiftx, shifty = getBestShift(img)
    shifted = shift(img, shiftx, shifty)

    # flatten and transpose data
    data = shifted.flatten().transpose()

    return data.reshape(1, -1)

def getBestShift(img):
    ''' determine optimal shift of number in image '''

    cy, cx = ndimage.measurements.center_of_mass(img)

    rows, cols = img.shape
    shiftx = np.round(cols / 2.0 - cx).astype(int)
    shifty = np.round(rows / 2.0 - cy).astype(int)

    return shiftx, shifty

def shift(img,sx,sy):
    ''' shifts the image contained within the boundaries by the specified amount '''

    rows, cols = img.shape

    M = np.float32([[1, 0, sx],[0, 1, sy]])
    shifted = cv2.warpAffine(img, M, (cols, rows))

    return shifted

def load_classifier(model_path):
    ''' loads the specified model from file '''

    extention = os.path.splitext(model_path)[1]

    if extention == '.h5': # Sequential Deep Learning
        return load_model(model_path)
    elif extention == '.pckl':
        with open(model_path, 'rb') as f: # Random Forest Classifier
            model = pickle.load(f)

        return model
    else:
        print('Invalid model file')
        sys.exit()

### MAIN FLOW ###
if __name__ == '__main__':

    model_path = 'models/mnist_random_forest_classifier.pckl' # path to file containing model

    if not os.path.exists(model_path):
        print('Please run build_model for the appropriate first to build the classifier model')
        sys.exit()

    model = load_classifier(model_path) # load classifier from file
    app = App(model)
    tk.mainloop() # run app
