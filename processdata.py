#encoding=utf8
import numpy as np
import xlrd
def main():
    readDataFromGlass()
    # readDataFromDianosis()
def readDataFromGlass():
    label = []
    dataSet = []
    with open("./data/glass.data") as f:
        for line in f.readlines():
            tmp = line.strip().split(",")
            label.append(tmp[-1].strip())
            dataSet.append([float(tk) for tk in tmp[0:-1]])
    x = np.array(dataSet)
    y = np.array(label)
    # print x
    # y = np.zeros(label.shape)
    # print '数据量：%d'%(len(x))
    return x, y

def readDataFromWine():
    """

    :return:
    """
    label = []
    dataSet= []
    with open("./data/wine.data") as f:
     for line in f.readlines():
         tmp = line.split(",")
         label.append(tmp[0])
         dataSet.append([float(tk) for tk in tmp[1:-1]])
    x = np.array(dataSet)
    y = np.array(label)
    # y = np.zeros(label.shape)
    # print '数据量：%d'%(len(x))
    return  x,y

# def readDataFromCar():
#     """
#
#     :return:
#     """
#     labels = []
#     dataSet = []
#     with open('./data/car.data') as f:
#         for line in f.readlines():
#             tmp = line.strip().split(',')
#             dataSet.append(tmp[0:-2])
#             labels.append(tmp[-1])
#     # print   len(labels),len(dataSet)
#     X = np.array(dataSet)
#     Y = np.array(labels)
#     return X,Y



def readDataFromSeed():
    """

    :return:
    """
    dataSet = []
    labels = []
    with open('./data/seeds_dataset.txt') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            labels.append(tmp[-1])
            tmp = [float(k) for k in tmp[:-1] if k!=""]
            dataSet.append(tmp)
    X = np.array(dataSet)
    Y = np.array(labels)

    return X,Y
def readDataFromLR():
    """

    :return:
    """
    dataSet = []
    labels = []
    with open('./data/letter-recognition.data') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            labels.append(tmp[0])
            tmp = [int(k) for k in tmp[1:]]
            dataSet.append(tmp)
    X = np.array(dataSet)
    Y = np.array(labels)
    # print X
    return X, Y
def readDataFromIris():
    """

    :return:
    """
    dataSet = []
    labels = []
    with open('./data/iris.data') as f:
        for line in f.readlines():
            tmp = line.strip().split(',')
            labels.append(tmp[-1])
            tmp = [float(k) for k in tmp[:-1]]
            dataSet.append(tmp)
    X = np.array(dataSet)
    Y = np.array(labels)
    # print X
    return X, Y

if __name__ == '__main__':
    main()