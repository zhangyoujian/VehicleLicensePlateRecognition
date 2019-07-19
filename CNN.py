import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D

class CNNChinese():
    def __init__(self, InputShape, ClassID=None):
        self.classID = ['云', '京', '冀', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']
        if ClassID==None:
            ClassID = len(self.classID)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(InputShape),
                         activation='relu'))
        # 添加MaxPooling2D，在2X2的格子中取最大值
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 设立Dropout层，常用的值为0.2,0.3,0.5
        model.add(Dropout(0.5))
        # 重复构造，搭建深度网络
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.5))
        # 把当前节点铺平
        model.add(Flatten())
        # 构建全连接层
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(ClassID, activation='softmax'))
        self.model = model

    def train(self,trainX,trainY, ValidX=None,ValidY=None):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self.model.fit(trainX, trainY, validation_data=(ValidX, ValidY), epochs=100, batch_size=64)
        self.model.save('OCR_model_weight.h5')

    def LoadModel(self,Path):
        self.model.load_weights(Path)

    def predict(self, image):
        image = np.expand_dims(image, axis=3)
        ret = self.model.predict(image)
        label = np.argmax(ret, axis=1)
        pred = []
        label = list(label)
        for l in label:
            pred.append(self.classID[l])
        return pred





class CNNLetter():
    def __init__(self, InputShape, ClassID=None):
        self.classID = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if ClassID == None:
            ClassID = len(self.classID)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=(InputShape),
                         activation='relu'))
        # 添加MaxPooling2D，在2X2的格子中取最大值
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # 设立Dropout层，常用的值为0.2,0.3,0.5
        model.add(Dropout(0.5))
        # 重复构造，搭建深度网络
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        # 把当前节点铺平
        model.add(Flatten())
        # 构建全连接层
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(ClassID, activation='softmax'))
        self.model = model

    def train(self,trainX,trainY, ValidX=None,ValidY=None):
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='SGD', metrics=['accuracy'])
        self.model.fit(trainX, trainY, validation_data=(ValidX, ValidY), epochs=100, batch_size=128)
        self.model.save('OCR_model_letters_weight.h5')

    def LoadModel(self, Path):
        self.model.load_weights(Path)

    def predict(self,Image):
        Image = np.expand_dims(Image, axis=3)
        ret = self.model.predict(Image)
        label = np.argmax(ret, axis=1)
        pred = []
        label = list(label)
        for l in label:
            pred.append(self.classID[l])
        return pred


