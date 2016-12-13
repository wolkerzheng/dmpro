#!/usr/bin/env python
#encoding=utf8
__author__ = 'ZGD'


import  numpy as np
import matplotlib.pyplot  as plt
import processdata
class KMean:
    def __init__(self):
        pass

    def calEuDistance(self,vecA,vecB):
        """

        :param vecA:
        :param vecB:
        :return:
        """
        return np.sqrt(np.sum(np.power(vecA-vecB,2)))

    def initCentroids(self,dataSet,k):
        """

        :param dataSet:
        :param k:
        :return:
        """
        #获取数据集的数据个数以及维数
        numSamples,dim = dataSet.shape
        centroids = np.zeros((k,dim))
        #随机初始化k个起始中心点
        for i in range(k):
            index = int(np.random.uniform(0,numSamples))
            centroids[i,:] = dataSet[index,:]
        return centroids

    def calSSE(self,clusterDict,centroidList):
        """

        :param clusterDict:
        :param centroidList:
        :return:
        """
        sum = 0.0
        for key in clusterDict.keys():
            vec1 = np.array(centroidList[key])
            distance = 0.0
            for item in clusterDict[key]:
                vec2 = np.array(item)
                distance += self.calEuDistance(vec1,vec2)
            sum += distance
        return sum


    def kmeans(self,dataSet,centroids):
        """

        :param dataSet:
        :param k:
        :return:
        """
        numSamples = dataSet.shape[0]
        clusterDict = dict()
        #选择k个点作为初始质心
        # centroids = self.initCentroids(dataSet,k)

        for item in dataset:
            vec1 = np.array(item)
            flag = 0
            minDist = float("inf")
            for i in range(len(centroids)):
                vec2 = np.array(centroids[i])
                distance = self.calEuDistance(vec1,vec2)
                if distance < minDist:
                    minDist = distance
                    flag = i
            if flag not in clusterDict.keys():
                clusterDict[flag] = list()
            clusterDict[flag].append(item)
        return clusterDict

    def getCentroids(self,clusterDict):
        # 重新得到k个质心
        centroidList = list()
        for key in clusterDict.keys():
            centroid = np.mean(np.array(clusterDict[key]), axis=0)  # 计算每列的均值，即找到质心
            # print key, centroid
            centroidList.append(centroid)

        return np.array(centroidList).tolist()


    def biKmean(self,dataset,k):
        dim = len(dataset[0])
        centroidDict = dict()
        centoid0  = np.mean(dataset,axis=0).tolist()[0]
        centroidList = [centoid0]
        while(len(centroidList)<k):
            lowestSSE = np.inf


        pass
def Main():
    kme = KMean()
    global dataset
    dataset,labels = processdata.readDataFromIris()
    stroreSSE = np.zeros(16)
    # print stroreSSE
    for kSize in range(1,16):
        centroidList = kme.initCentroids(dataset, kSize)
        clusterDict = kme.kmeans(dataset, centroidList)
        newVar = kme.calSSE(clusterDict, centroidList)  # 获得均方误差值
        oldVar = -0.0001  #
        k = 2
        while abs(newVar - oldVar) >= 0.0001:  # 通过新旧均方误差来获得迭代终止条件当连续两次聚类结果小于0.0001时，迭代结束
            centroidList = kme.getCentroids(clusterDict)  # 获得新的质心
            clusterDict = kme.kmeans(dataset, centroidList)  # 新的聚类结果
            oldVar = newVar
            newVar = kme.calSSE(clusterDict, centroidList)
            k += 1
        stroreSSE[kSize] = newVar
    x=[i for i in range(1,16)]
    y = [j for j in stroreSSE[1:]]
    plt.plot(x,y,'b')
    plt.show()

def test2():
    kme = KMean()
    global dataset
    dataset, labels = processdata.readDataFromIris()
    stroreK = np.zeros(10)
    # print stroreSSE
    kSize=3
    for i in range(10):
        centroidList = kme.initCentroids(dataset, kSize)
        clusterDict = kme.kmeans(dataset, centroidList)
        newVar = kme.calSSE(clusterDict, centroidList)  # 获得均方误差值
        oldVar = -0.0001  #
        k = 2
        while abs(newVar - oldVar) >= 0.0001:  # 通过新旧均方误差来获得迭代终止条件当连续两次聚类结果小于0.0001时，迭代结束
            centroidList = kme.getCentroids(clusterDict)  # 获得新的质心
            clusterDict = kme.kmeans(dataset, centroidList)  # 新的聚类结果
            oldVar = newVar
            newVar = kme.calSSE(clusterDict, centroidList)
            k += 1
        stroreK[i] = k
    x = [i for i in range(1,11)]
    y = [j for j in stroreK]
    plt.plot(x, y, 'b')
    plt.show()
if __name__ == '__main__':
    test2()