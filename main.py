import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from window import Ui_Myapp
import cv2
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_Myapp()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.loadImage)
        self.ui.pushButton_2.clicked.connect(self.colorSeperation)
        self.ui.pushButton_3.clicked.connect(self.colorTransformation)

    def loadImage(self):
        name = 'Sun.jpg'
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/'+name)
        cv2.imshow(name, pic)

    def colorSeperation(self):
        name = 'Sun.jpg'
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/' + name)
        (B, G, R) = cv2.split(pic)
        zeros = np.zeros(pic.shape[:2], dtype='uint8')
        cv2.imshow('R', cv2.merge([zeros, zeros, R]))
        cv2.imshow('G', cv2.merge([zeros, G, zeros]))
        cv2.imshow('B', cv2.merge([B, zeros, zeros]))

    def colorTransformation(self):
        name = 'Sun.jpg'
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/' + name)
        l1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        cv2.imshow('l1', l1)
        (B, G, R) = cv2.split(pic)
        l2 = (R.astype(int) + G.astype(int) + B.astype(int))/3
        l2 = l2.astype('uint8')
        cv2.imshow('l2', l2.astype('uint8'))

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())