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
        self.ui.pushButton_4.clicked.connect(self.blending)
        self.ui.pushButton_5.clicked.connect(self.gaussianBlur)
        self.ui.pushButton_6.clicked.connect(self.bilateralFilter)
        self.ui.pushButton_7.clicked.connect(self.medianFilter)


    def loadImage(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        cv2.imshow('Sun.jpg', pic)

    def colorSeperation(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        (B, G, R) = cv2.split(pic)
        zeros = np.zeros(pic.shape[:2], dtype='uint8')
        cv2.imshow('R', cv2.merge([zeros, zeros, R]))
        cv2.imshow('G', cv2.merge([zeros, G, zeros]))
        cv2.imshow('B', cv2.merge([B, zeros, zeros]))

    def colorTransformation(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        l1 = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        cv2.imshow('l1', l1)
        (B, G, R) = cv2.split(pic)
        l2 = (R.astype(int) + G.astype(int) + B.astype(int))/3
        l2 = l2.astype('uint8')
        cv2.imshow('l2', l2.astype('uint8'))

    def blending(self):
        def func(x):
            pass
        big = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Strong.jpg')
        small = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Dog_Weak.jpg')
        img = np.zeros(big.shape, np.uint8)
        cv2.namedWindow('Blend')
        cv2.createTrackbar('Blend', 'Blend', 0, 255, func)
        while(True):
            cv2.imshow('Blend', img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            r = cv2.getTrackbarPos('Blend', 'Blend')
            r = float(r) / 255.0
            img = cv2.addWeighted(big, 1.0 - r, small, r, 0)
        cv2.destroyWindow('Blend')

    def gaussianBlur(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')
        blur = cv2.GaussianBlur(pic, (5, 5), 0)
        cv2.imshow('Gaussian Blur', blur)

    def bilateralFilter(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_whiteNoise.jpg')
        bilateral = cv2.bilateralFilter(pic, 9, 90, 90)
        cv2.imshow('Bilateral Filter', bilateral)

    def medianFilter(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q2_Image/Lenna_pepperSalt.jpg')
        median3 = cv2.medianBlur(pic, 3)
        median5 = cv2.medianBlur(pic, 5)
        cv2.imshow('Median Filter 3x3', median3)
        cv2.imshow('Median Filter 5x5', median5)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())