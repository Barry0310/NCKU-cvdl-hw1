import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui_cifar10 import Ui_MainWindow
from torchsummary import summary
import torch
import cv2
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.showTrainImages)
        self.ui.pushButton_2.clicked.connect(self.showHyperParameter)
        self.ui.pushButton_3.clicked.connect(self.showModelShortcut)
        self.ui.pushButton_4.clicked.connect(self.showAccuracy)
        self.ui.pushButton_5.clicked.connect(self.test)

    def showTrainImages(self):
        train_data = datasets.CIFAR10('./data/cifar10', train=True, download=True)
        classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        fig, axes = plt.subplots(3, 3)
        count = 0
        for i in range(3):
            for j in range(3):
                axes[i, j].imshow(train_data[count][0])
                axes[i, j].set_title(classes[train_data[count][1]])
                axes[i, j].axis('off')
                count += 1
        fig.tight_layout()
        fig.show()


    def showHyperParameter(self):
        print('hyperparameters:')
        print('batch size: {}'.format(128))
        print('learning rate: {}'.format(0.01))
        print('epoch: {}'.format(40))
        print('optimizer: {}'.format('SGD'))
        print('loss function: {}'.format('Cross Entropy Loss'))

    def showModelShortcut(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
        model.classifier._modules['6'] = torch.nn.Linear(in_features=4096, out_features=10)
        if torch.cuda.is_available():
            model = model.cuda()
        summary(model, (3, 32, 32))

    def showAccuracy(self):
        pic = cv2.imread('cifar10_model/curve.png')
        cv2.imshow('Loss and Accuracy', pic)

    def test(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=False)
        model.classifier._modules['6'] = torch.nn.Linear(in_features=4096, out_features=10)
        if torch.cuda.is_available():
            model = model.cuda()
        model.load_state_dict(torch.load('cifar10_model/vgg16_cifar10.pth'))
        train_data = datasets.CIFAR10('./data/cifar10', train=False, transform=transforms.ToTensor(), download=True)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        index = int(self.ui.lineEdit.text())
        pic = train_data[index][0].transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
        data = train_data[index][0].reshape(1, 3, 32, 32)
        if torch.cuda.is_available():
            data = data.cuda()
        pred = model(data)
        pred = torch.nn.functional.softmax(pred[0], dim=0).cpu().detach().numpy()
        plt.figure()
        plt.imshow(pic)
        plt.axis('off')
        plt.show()
        plt.figure()
        plt.bar(classes, pred)
        plt.show()
        print(classes[train_data[index][1]])



if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
