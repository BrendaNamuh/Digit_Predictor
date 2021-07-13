import tkinter as tk
from tkinter import *
import numpy as np
import tensorflow as tf
import cv2
import PIL
from PIL import ImageGrab

class Main:
    def __init__(self, master):
        # Attributes
        self.master = master
        self.color_fg = "white"  # Foreground colour
        self.color_bg = "black"  # Background colour
        self.old_x = None
        self.old_y = None
        self.penwidth = 20

        # Loading the CNN model
        self.model = tf.keras.models.load_model('digit.model')

        # Setting up the Canvas
        self.canv = Canvas(self.master, width=500, height=500, bg=self.color_bg)
        self.setButtons()
        label = Label(self.master, text=" -- \nAccuracy:       %")
        label.config(font=("Courrier,14"))
        label.pack()
        self.result_label =label

        self.canv.bind('<B1-Motion>', self.paint)



    def paint(self, event):
        if self.old_x and self.old_y :
            self.canv.create_line(self.old_x, self.old_y, event.x, event.y, width=self.penwidth, fill=self.color_fg,
                                 capstyle=ROUND, smooth=True)
        self.old_x = event.x
        self.old_y = event.y

    def clear(self):
        self.canv.delete(ALL)  # Clears canvas
        self.old_x = None
        self.old_y = None
        self.result_label.config(text=" -- \nAccuracy:       %")  # Resets text



    def predict(self):

        # Save canvas as png ------------------------------------
        x = root.winfo_rootx() + self.canv.winfo_x()
        y = root.winfo_rooty() + self.canv.winfo_y()
        xx = x + self.canv.winfo_width()
        yy = y + self.canv.winfo_height()
        ImageGrab.grab(bbox=(x, y, xx, yy)).save("drawn_digit.png")
        # -------------------------------------------

        #Process Image
        image = cv2.imread("drawn_digit.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Set to from colour to grayscale
        image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA) # Resize
        image = tf.keras.utils.normalize(image, axis=1) # 0 to 1 scaling
        image = np.array(image).reshape(-1, 28, 28, 1) #Reshape

        # Make prediciton
        prediction = self.model.predict(image)
        # Display prediction on GUI
        index = np.argmax(prediction[0])
        txt = "{}\nAccuracy: {}%".format(index, round(prediction[0][index]*100, 3))
        self.result_label.config(text = txt)



    def setButtons(self):
        # Button Setup
        btnClear = Button(text="Clear", command=self.clear)
        btnExit = Button(text="Exit", command=self.master.destroy)
        btnPredict = Button(text="Predict", command=self.predict)

        btnExit.pack()
        btnClear.pack()
        btnPredict.pack()
        self.canv.pack(fill=BOTH, expand=True)


if __name__ == '__main__':
    root = Tk()
    main = Main(root)
    root.title("Digit Predictor")
    root.mainloop()
