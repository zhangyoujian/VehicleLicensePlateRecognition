import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.transform import resize
from DigitRecognize.CNN import CNNChinese,CNNLetter
from DigitRecognize.mnist_loader import load_data
import skimage.io
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题





class LoadData():
    def __init__(self, ChineseShape, CharShape):
        self.ChineseShape = ChineseShape
        self.CharShape = CharShape



    def LoadChineseImages(self, Path,ClassID=None):
        DatasSetPath = Path+'\\chinese_letters\\'
        ChineseClassID = []
        Chinesedataset = []
        ChineseLabel = []

        for root, dirs, files in os.walk(DatasSetPath):
           for sub_dir in dirs:
               ChineseClassID.append(sub_dir)
               sub_Path = os.path.join(root, sub_dir)
               for sub_root, sub_dirs, sub_files in os.walk(sub_Path):
                   for name in sub_files:
                       image = skimage.io.imread(os.path.join(sub_root, name), True)
                       image =resize(image, self.ChineseShape)
                       Chinesedataset.append(image)
                       if not ClassID is None:
                            ChineseLabel.append(ClassID.index(sub_dir))
                       else:
                            ChineseLabel.append(ChineseClassID.index(sub_dir))

        return Chinesedataset, ChineseLabel, ChineseClassID


    def LoadCharImage(self,Path,ClassID=None):
        DatasSetPath_letters = Path + '\\letters\\'
        DatasSetPath_digits = Path + '\\digits\\'
        CharDataset =[]
        CharClassID =[]
        CharLabel = []

        # 添加字母数据
        for root, dirs, files in os.walk(DatasSetPath_letters):
           for sub_dir in dirs:
               CharClassID.append(sub_dir)
               sub_Path = os.path.join(root, sub_dir)
               for sub_root, sub_dirs, sub_files in os.walk(sub_Path):
                   for name in sub_files:
                       image = skimage.io.imread(os.path.join(sub_root, name), True)
                       image =resize(image, self.CharShape)
                       CharDataset.append(image)
                       if not ClassID is None:
                            CharLabel.append(ClassID.index(sub_dir))
                       else:
                            CharLabel.append(CharClassID.index(sub_dir))


        #添加数字数据
        for root, dirs, files in os.walk(DatasSetPath_digits):
            for sub_dir in dirs:
                CharClassID.append(sub_dir)
                sub_Path = os.path.join(root, sub_dir)
                for sub_root, sub_dirs, sub_files in os.walk(sub_Path):
                    for name in sub_files:
                        image = skimage.io.imread(os.path.join(sub_root, name), True)
                        image = resize(image, self.CharShape)
                        CharDataset.append(image)
                        if not ClassID is None:
                            CharLabel.append(ClassID.index(sub_dir))
                        else:
                            CharLabel.append(CharClassID.index(sub_dir))

        return CharDataset, CharLabel, CharClassID


    def LoadData(self,Path, type, ClassID=None):
        if type=='chinese':
            datasets,Label,classid = self.LoadChineseImages(Path,ClassID)
        else:
            datasets, Label, classid = self.LoadCharImage(Path,ClassID)
        return datasets, Label, classid


    def PrepareData(self, datasets, Label):
        trainX = np.array(datasets)
        trainY = np.array(Label).reshape(-1, 1)
        trainX = np.expand_dims(trainX, 3)
        return trainX, trainY


def ExpandLetterData():
    training_data, validation_data, test_data = load_data()
    train_img, train_label = training_data[0], training_data[1]
    valid_img, valid_label = validation_data[0], validation_data[1]
    test_img, test_label = test_data[0], test_data[1]

    train_img = np.asarray(train_img, dtype='float64')
    train_label = np.asarray(train_label)

    valid_img = np.asarray(valid_img, dtype='float64')
    valid_label = np.asarray(valid_label)

    test_img = np.asarray(test_img, dtype='float64')
    test_label = np.asarray(test_label)

    N = 50000
    X_train = train_img[range(0, N)].reshape(N, 28, 28, 1)
    Y_train = train_label


    M = 10000
    X_valid = valid_img[range(0, M)].reshape(M, 28, 28, 1)
    Y_valid = valid_label


    # M = 10000
    # X_test = test_img[range(0, M)].reshape(M, 28, 28, 1)
    # Y_test = test_label
    #

    return X_train,Y_train,X_valid,Y_valid


def trainLetterModel():
    obj = LoadData((32, 32),(28, 28))
    datasets_letters_train, Label_letters_train, ClassID_letters = obj.LoadData(r"D:\pycharmWorkspace\CarDetectedSystem\DigitRecognize\dataset\trainSet",'letter')
    print(ClassID_letters)
    datasets_letters_valid, Label_letters_valid, ClassID_letters_valid = obj.LoadData(r"D:\pycharmWorkspace\CarDetectedSystem\DigitRecognize\dataset\ValidSet", 'letter',ClassID_letters)
    trainX,trainY = obj.PrepareData(datasets_letters_train, Label_letters_train)
    ValidX,ValidY = obj.PrepareData(datasets_letters_valid, Label_letters_valid)


    offset = ClassID_letters.index('0')
    trainX_mnist, trainY_mnist, ValidX_mnist, ValidY_mnist = ExpandLetterData()

    trainY_mnist = trainY_mnist+offset
    ValidY_mnist = ValidY_mnist+offset
    trainY_mnist = trainY_mnist.reshape(-1,1)
    ValidY_mnist = ValidY_mnist.reshape(-1,1)

    merge_trainX = np.concatenate((trainX, trainX_mnist),axis=0)
    merge_trainY = np.concatenate((trainY, trainY_mnist),axis=0)

    merge_ValidX = np.concatenate((ValidX, ValidX_mnist), axis=0)
    merge_ValidY = np.concatenate((ValidY, ValidY_mnist), axis=0)


    model = CNNLetter((28, 28, 1), len(ClassID_letters))
    model.train(merge_trainX, merge_trainY, ValidX,ValidY)


def trainChineseModel():
    obj = LoadData((32, 32),(28, 28))
    datasets_chineese_train, Label_chineese_train, ClassID_chineese = obj.LoadData(r"D:\pycharmWorkspace\CarDetectedSystem\DigitRecognize\dataset\trainSet",'chinese')
    print(ClassID_chineese)
    datasets_chineese_valid, Label_chineese_valid, ClassID_chineese_valid = obj.LoadData(r"D:\pycharmWorkspace\CarDetectedSystem\DigitRecognize\dataset\ValidSet", 'chinese',ClassID_chineese)
    trainX,trainY = obj.PrepareData(datasets_chineese_train, Label_chineese_train)
    ValidX,ValidY = obj.PrepareData(datasets_chineese_valid, Label_chineese_valid)
    model = CNNChinese((32, 32, 1), len(ClassID_chineese))
    model.train(trainX, trainY, ValidX, ValidY)


def testChineseModel(Path):
    model = CNNChinese((32, 32, 1))
    model.LoadModel("OCR_model_weight.h5")

    image = skimage.io.imread(Path, True)
    if np.max(image)>=2:
        image = image/255.0
    image = resize(image, (32, 32))
    image = np.expand_dims(image,axis=2)
    image = np.array([image])
    ret = model.predict(image)
    print(ret)

def testLetterModel(Path):
    model = CNNLetter((28, 28, 1))
    model.LoadModel("OCR_model_letters_weight.h5")
    image = skimage.io.imread(Path, True)
    if np.max(image) >= 2:
        image = image / 255.0
    image = resize(image, (28, 28))
    image = np.array([image])
    ret = model.predict(image)
    print(ret)





def main():
    # trainChineseModel()
    imagePath = r"D:\pycharmWorkspace\CarDetectedSystem\DigitRecognize\dataset\test_images\6.bmp"
    # testChineseModel(imagePath)
    testLetterModel(imagePath)
    # trainLetterModel()





if __name__=='__main__':
    main()







