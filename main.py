import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from window import Ui_Myapp
import cv2
import numpy as np
from math import pi, exp

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
        self.ui.pushButton_13.clicked.connect(self.gaussianBlur2)
        self.ui.pushButton_14.clicked.connect(self.sobelX)
        self.ui.pushButton_15.clicked.connect(self.sobelY)
        self.ui.pushButton_16.clicked.connect(self.magnitude)
        self.ui.pushButton_9.clicked.connect(self.resizee)
        self.ui.pushButton_10.clicked.connect(self.translation)
        self.ui.pushButton_11.clicked.connect(self.rotation)


    def loadImage(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q1_Image/Sun.jpg')
        cv2.imshow('Sun.jpg', pic)
        print(pic.shape[:2])

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

    def gaussianBlurImplement(self, pic):
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
        result = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                result[i][j] = exp(-((i - 1) * (i - 1) + (j - 1) * (j - 1)) / (2 * 0.5)) / (2 * pi * 0.5)
        all = result.sum()
        result = result / all
        pic = pic.astype(float)
        new = np.zeros(pic.shape)
        for i in range(1, pic.shape[0] - 1):
            for j in range(1, pic.shape[1] - 1):
                t = pic[i - 1:i + 2, j - 1:j + 2]
                a = np.multiply(t, result)
                new[i][j] = a.sum()
        return new.astype('uint8')

    def gaussianBlur2(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        cv2.imshow('Gaussian Blur 2', self.gaussianBlurImplement(pic))

    def sobelImplement(self, pic, filter):
        pic = self.gaussianBlurImplement(pic).astype(float)
        new = np.zeros(pic.shape)
        for i in range(1, pic.shape[0] - 1):
            for j in range(1, pic.shape[1] - 1):
                t = pic[i - 1:i + 2, j - 1:j + 2]
                a = np.multiply(filter, t)
                new[i, j] = a.sum()
        return np.absolute(new).astype('uint8')

    def sobelX(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        filter = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        cv2.imshow('Sobel X', self.sobelImplement(pic, filter))

    def sobelY(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        filter = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        cv2.imshow('Sobel Y', self.sobelImplement(pic, filter))

    def magnitude(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q3_Image/House.jpg')
        filter = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        x = self.sobelImplement(pic, filter).astype(float)
        filter = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        y = self.sobelImplement(pic, filter).astype(float)
        m = np.sqrt(np.square(x)+np.square(y))
        new = (m*255/m.max()).astype('uint8')
        cv2.imshow('Magnitude', new)

    def resizee(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        pic = cv2.resize(pic, (256, 256))
        cv2.namedWindow('Resize', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Resize', 400, 300)
        cv2.imshow('Resize', pic)

    def translation(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        pic = cv2.resize(pic, (256, 256))
        t = np.float32([[1, 0, 0],
                        [0, 1, 60]])
        shifted = cv2.warpAffine(pic, t, (pic.shape[0], pic.shape[1]))
        cv2.namedWindow('Translation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Translation', 400, 300)
        cv2.imshow('Translation', shifted)

    def rotation(self):
        pic = cv2.imread('Dataset_OpenCvDl_Hw1/Q4_Image/SQUARE-01.png')
        pic = cv2.resize(pic, (256, 256))
        t = np.float32([[1, 0, 0],
                        [0, 1, 60]])
        shifted = cv2.warpAffine(pic, t, (pic.shape[0], pic.shape[1]))
        t = cv2.getRotationMatrix2D((128, 188), 10, 1.0)
        print(t)
        shifted = cv2.warpAffine(shifted, t, (pic.shape[0], pic.shape[1]))
        cv2.namedWindow('Rotation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Rotation', 400, 300)
        cv2.imshow('Rotation', shifted)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())