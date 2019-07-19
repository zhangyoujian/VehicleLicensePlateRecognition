import numpy as np
import cv2
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
import copy
from skimage.transform import radon, rescale, rotate, resize
import skimage.io
import math


class CharSplit():
    def __init__(self):
        pass

    @staticmethod
    def BestThrehold(image):
        ma = np.max(image)
        mi = np.min(image)
        threhold = round((ma-(ma-mi)/3))
        return threhold

    @staticmethod
    def randonTransform(image_gray):

        I = image_gray
        I = I - np.mean(I)
        sinogram = radon(I)
        r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
        rotation = np.argmax(r)
        return 90-rotation


    @staticmethod
    def imrotate(image,angle):
        h,w = image.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        dst = cv2.warpAffine(image, M, (w, h))
        return dst


    @staticmethod
    def baweraopen(image, size):
        output = image.copy()
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(image)
        for i in range(1, nlabels - 1):
            regions_size = stats[i, 4]
            if regions_size < size:
                x0 = stats[i, 0]
                y0 = stats[i, 1]
                x1 = stats[i, 0] + stats[i, 2]
                y1 = stats[i, 1] + stats[i, 3]
                for row in range(y0, y1):
                    for col in range(x0, x1):
                        if labels[row, col] == i:
                            output[row, col] = 0
        return output

    @staticmethod
    def BinaryImage(image_gray):
        # Threhold = CharSplit.BestThrehold(image_gray)
        # ret,image_binary = cv2.threshold(image_gray,Threhold,255,cv2.THRESH_BINARY)
        # image_binary = cv2.adaptiveThreshold(image_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
        ret3, image_binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_open = CharSplit.baweraopen(image_binary, 16)

        plt.subplot(1, 3, 1), plt.imshow(image_gray, 'gray')
        plt.title('image_gray')
        plt.subplot(1, 3, 2), plt.imshow(image_binary, 'gray')
        plt.title('image_binary')
        plt.subplot(1, 3, 3), plt.imshow(image_open, 'gray')
        plt.title('image_open')
        plt.show()

        return image_open


    @staticmethod
    def LevelRowAnalyse(image_open):
        height = image_open.shape[0]
        image_open_binary = copy.deepcopy(image_open)
        image_open_binary[image_open_binary == 255] = 1
        # 计算水平投影直方图
        histrow = np.sum(image_open_binary, axis=1)

        levelRow = (np.mean(histrow) + np.min(histrow)) / 2
        plt.subplot(1, 2, 1)
        plt.imshow(image_open, 'gray')
        plt.title('原始二值化图')
        plt.subplot(1, 2, 2)
        plt.bar(np.arange(1, histrow.shape[0]+1), histrow)
        plt.title('水平投影')
        plt.show()

        markrow = []
        markrow1 = []
        count1 = 0
        # 对水平投影进行峰谷分析
        for i in range(height):
            if histrow[i] <= levelRow:
                count1 += 1
            else:
                if count1 >= 1:
                    markrow.append(i)
                    markrow1.append(count1)
                count1 = 0

        markrow2 = np.diff(markrow)
        markrow2 = list(markrow2)
        n1 = len(markrow2) + 1
        markrow.append(height-1)
        markrow1.append(count1)
        markrow2.append(markrow[-1]-markrow[-2])

        # markrow3 = []
        # markrow4 = []
        # markrow5 = []
        # for i in range(n1):
        #     markrow3.append(markrow[i+1] - markrow1[i+1])
        #     markrow4.append(markrow3[i] - markrow[i])
        #     markrow5.append(markrow3[i] - (markrow4[i]//2))
        #
        #
        # print('markrow3',markrow3)
        # print('markrow4', markrow4)
        # print('markrow5', markrow5)

        return markrow, markrow1,markrow2

    # 计算车牌旋转角度
    # @staticmethod
    # def calAngle(bg4, sbw1, markrow, markrow4, markrow3):
    #     m2,n2 = bg4.shape
    #     n1 = len(markrow4)
    #     maxw = max(markrow4)
    #     xdada = []
    #     ydata= []
    #     if markrow4[0] != maxw:
    #         ysite = 0
    #         for l in range(n2):
    #             for k in range(markrow3[ysite]):
    #                 if sbw1[k,l]==1:
    #                     xdada.append(l)
    #                     ydata.append(k)
    #                     break
    #
    #     else: #检测下边
    #         ysite = n1-1
    #         if markrow4[-1]==0:
    #             if markrow4[-2]==maxw:
    #                 ysite=0
    #             else:
    #                 ysite = n1-1
    #
    #         if ysite!=0:
    #

#     计算车牌水平投影，去掉车牌水平边框，获取字符高度
    @staticmethod
    def rowAnalyse(sbw, markrow, markrow1, markrow2):
        sbw[sbw==255]=1
        histcol1 = np.sum(sbw, axis=0)
        histrow = np.sum(sbw, axis=1)

        plt.subplot(1, 3, 1), plt.imshow(sbw, 'gray')
        plt.title('车牌二值化图像')
        plt.subplot(1, 3, 2), plt.bar(np.arange(1,histcol1.shape[0]+1),histcol1)
        plt.title('垂直投影')
        plt.subplot(1, 3, 3), plt.bar(np.arange(1,histrow.shape[0]+1),histrow)
        plt.title('水平投影')
        plt.show()

        maxhight = max(markrow2)  #% 获取最大峰距，即一个字符 + 一个谷底宽
        findc = np.argmax(markrow2)
        rowtop = markrow[findc]
        rowbot = markrow[findc+1] - markrow1[findc+1]
        sbw2 = sbw[rowtop:rowbot, :]    #子图为(rowbot-rowtop+1)行  分割出最大高度所在字符
        maxhight = rowbot - rowtop + 1
        return sbw2, maxhight, rowtop, rowbot

    @staticmethod
    def colAnalyse(gray_image, subcol ,sbw2, maxhight, rowtop, rowbot):
        height,width = sbw2.shape
        if rowbot>subcol.shape[0]:
            rowbot = subcol.shape[0]
        sbw2[sbw2>1]=1
        histcol = np.sum(sbw2,axis=0)
        plt.subplot(2, 1, 1), plt.bar(np.arange(1, histcol.shape[0]+1),histcol)
        plt.title('垂直投影')
        plt.subplot(2, 1, 2), plt.imshow(sbw2*255, 'gray')
        plt.title('车牌字符高度:%d'%(maxhight))
        plt.show()


        levelcol = (np.mean(histcol) + np.min(histcol))/4
        markcol = []
        markcol1 = []
        count1 = 0
        for k in range(width):
            if histcol[k]<=levelcol:
                count1+=1
            else:
                if count1>=1:
                    markcol.append(k)
                    markcol1.append(count1)  #谷宽度
                count1 = 0

        markcol2 = np.diff(markcol)
        markcol2 = list(markcol2)
        n1 = len(markcol2) + 1
        markcol.append(width-1)
        markcol1.append(count1)
        markcol2.append(markcol[-1] - markcol[-2])

        markcol3 = np.zeros(n1-1)
        markcol4 = np.zeros(n1-1)
        markcol5 = np.zeros(n1 - 1)
        for k in range(n1-1):
            markcol3[k] = markcol[k+1] - markcol1[k+1]
            markcol4[k] = markcol3[k] - markcol[k]
            markcol5[k] = markcol3[k] - (markcol4[k]// 2)

        markcol6 = np.diff(markcol5)
        if len(markcol6)<2:
            print('无法识别该车牌')
            return
        findmax = np.argmax(markcol6)
        markcol6[findmax] = 0
        maxwidth = np.max(markcol6)

        m2,n2 = subcol.shape
        l = 1
        if findmax==0:
            findmax=1

        retGrayImage = []
        retBinaryImage =[]
        for k in range(findmax-1,findmax+6):
            if k >= len(markcol5):
                break
            cleft = markcol5[k] - maxwidth/2
            cright = markcol5[k] + maxwidth / 2 - 2
            if cleft<0:
                cleft=0
                cright=maxwidth

            if cright>n2-1:
                cright = n2-1
                cright = n2 - maxwidth

            SegGray = gray_image[rowtop:rowbot, math.floor(cleft):math.floor(cright)]
            SegBw1 = subcol[rowtop:rowbot, math.floor(cleft):math.floor(cright)]
            SegGray1 = resize(SegGray, (40, 32))
            SegBw2 = resize(SegBw1, (40, 32))
            plt.subplot(2, 7, l), plt.imshow(SegGray1, 'gray')
            retGrayImage.append(SegGray)
            plt.title('第[%d]个字符' % (l))
            plt.subplot(2, 7, 7+l), plt.imshow(SegBw2, 'gray')
            cv2.imwrite('%d.bmp' % (l),SegBw1*255.0)
            retBinaryImage.append(SegBw1*255.0)
            plt.title('第[%d]个字符'%(l))

            l+=1
        plt.show()

        return retGrayImage,retBinaryImage








