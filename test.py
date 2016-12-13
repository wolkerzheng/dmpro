#encoding=utf8
__author__='ZGD'

# from processdata import readDataFromWine
import processdata
from sklearn.model_selection import train_test_split
import MyDecisionTree
import ShowTree
import numpy as np
import operator
def holdoutMethod(X,Y):
    """

    :param X:
    :param Y:
    :return:
    """
    print '---------------------------------------------------'
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # mmode = raw_input('mode:=')
    clf = MyDecisionTree.DesicionTree(mode='GainInfo')
    Dtree = clf.BuildTree(x_train, y_train)
    # ShowTree.drawtree(Dtree,'TreeCarID3.jpeg')
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum,len(predicted)

    print '准确率:%f'%float(rightNum * 1.0 / len(predicted))
    print "holdout success"
    print '---------------------------------------------------'

def KFoldCrossValidation(X,Y,kSize=10):
    """

    :param X:
    :param Y:
    :param kSize:
    :return:
    """
    print '---------------------------------------------------'
    dataNum = len(X)
    n1 = dataNum % 10
    n2 = dataNum //10
    current = 0
    stop =  0
    sum = 0.0
    for i in xrange(kSize):
        if i < n1:
            stop = current + n2 + 1
        else:
            stop = current + n2
        x_test,y_test = X[current:stop],Y[current:stop]
        x_train =  np.concatenate((X[0:current], X[stop:]))
        y_train = np.concatenate((Y[0:current], Y[stop:]))
        current = stop
        clf = MyDecisionTree.DesicionTree(mode="Error")
        Dtree = clf.BuildTree(x_train, y_train)
        # ShowTree.drawtree(Dtree, 'Tree10FCV.jpeg')
        predicted = clf.predict(Dtree, x_test)
        rightNum = 0
        for j in range(len(predicted)):
            if predicted[j] == y_test[j]:
                rightNum += 1
        # print '第%d轮预测正确数：%d,总预测数目：%d'%(i+1,rightNum, len(predicted))
        sum = sum + float(rightNum * 1.0 / len(predicted))
        # print '第%d轮准确率:%f' % (i+1,float(rightNum * 1.0 / len(predicted)))
    print "准确率为：%f"%(sum/kSize)
    print '10折交叉验证成功'
    print '---------------------------------------------------'

def myBootstrap(X,Y):
    """

    :param X:
    :param Y:
    :return:
    """
    print '---------------------------------------------------'
    n = len(X)
    size = 10
    alpha = np.zeros(size)
    acc = np.zeros(size)
    for k in range(size):
        train_index_list = np.random.choice(n,n)
        x_train,y_train,x_test,y_test = [],[],[],[]
        for i in range(n):
            if i in train_index_list:
                x_train.append(X[i])
                y_train.append(Y[i])
            else:
                x_test.append(X[i])
                y_test.append(Y[i])
        clf = MyDecisionTree.DesicionTree(mode='Gini')
        Dtree = clf.BuildTree(x_train, y_train)
        predicted = clf.predict(Dtree, x_test)
        rightNum = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                rightNum += 1
        alpha[k] = float(rightNum * 1.0 / len(predicted))
        predicted = clf.predict(Dtree, X)
        rightNum = 0
        for i in range(len(predicted)):
            if predicted[i] == Y[i]:
                rightNum += 1
        acc[k] = float(rightNum * 1.0 / len(predicted))
        # print k
    accb = 0.0
    for i in range(size):
        accb += alpha[i]*0.632 +   acc[i]*0.368
    accb = float(accb/size)
    print np.sum(acc) / size,np.sum(alpha)/size
    print '准确率:%f' % accb
    print "Bootstrap success"
    print '---------------------------------------------------'
def AdaBoost(X,Y):
    """

    :param X:
    :param Y:
    :return:
    """
    N = len(X)
    w = np.full(N,float(1.0/N))
    C = []
    k = 8  #表示提升的轮数
    xigma = np.full(k,0.0)
    Alpha = np.full(k,0.0)
    for i in range(k):
        C.append(MyDecisionTree.DesicionTreeNode())
    for i in range(k):
        train_index_list = np.random.choice(N, N,p=w)
        x_train, y_train = [], []
        for j in set(train_index_list):
            x_train.append(X[j])
            y_train.append(Y[j])
        clf = MyDecisionTree.DesicionTree(mode='GainInfo')
        Dtree = clf.BuildTree(x_train, y_train)
        C[i] = (clf,Dtree)
        predicted = clf.predict(Dtree, X)
        xigma[i] = 0.0
        for j in range(N):
            if predicted[j] == Y[j]: I = 1
            else: I = 0
            xigma[i] += w[j]*I
        xigma[i] = xigma[i] / N
        if xigma[i]>0.5: w = np.full(N, float(1.0 / N))
        else:
            Alpha[i] = 1.0*np.log((1-xigma[i])/xigma[i])/2
            Zsum =0.0
            for j in range(N):
                if predicted[j] == Y[j]: tmp = w[j]*np.exp(-Alpha[i])
                else: tmp = w[j] * np.exp(Alpha[i])
                Zsum += tmp
            for j in range(N):
                if predicted[j] == Y[j]:w[j] = w[j]*np.exp(-Alpha[i]) / Zsum
                else:w[j] = w[j] * np.exp(Alpha[i]) / Zsum
    AdaBoostVote(X,Y,C,Alpha,k)

def AdaBoostVote(X,Y,C,alpha,k):
    """

    :param X:
    :param Y:
    :param clf:
    :param alpha:
    :param k:
    :return:
    """
    predicted = []
    for x in X:
        tmp = {}
        for j in range(k):
            clf = C[j][0]
            predict = clf.predict(C[j][1], [x])
            if predict not in tmp.keys():
                tmp[predict] = alpha[j]
            else:
                tmp[predict] = tmp[predict] + alpha[j]
        sorted_pred = sorted(tmp.iteritems(), key=operator.itemgetter(1), reverse=True)
        predicted.append(sorted_pred[0][0])
    rightNum = 0
    for i in range(len(predicted)):
        if predicted[i] == Y[i]:
            rightNum += 1
    print rightNum, len(predicted)

    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "AdaBoost success"

def mypostprune(X, Y):
    """

    :param X:
    :param Y:
    :return:
    """
    print '---------------------------------------------------'
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # mmode = raw_input('mode:=')
    clf = MyDecisionTree.DesicionTree(mode="GainInfo")
    print "before postprune:"
    # result = clf.predict(test_x)
    Dtree = clf.BuildTree(x_train, y_train)
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)
    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "after postprune:"
    clf.postprune(Dtree, x_test,  y_test)
    # ShowTree.drawtree(Dtree,'TreeCarID3.jpeg')
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)

    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "post success"
    print '---------------------------------------------------'
def mypreprune(X,Y):
    print '---------------------------------------------------'
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    # mmode = raw_input('mode:=')
    clf = MyDecisionTree.DesicionTree(mode="GainInfo")
    print "before preprune:"
    # result = clf.predict(test_x)
    Dtree = clf.BuildTree(x_train, y_train)
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)
    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "after preprune:"
    Dtree = clf.preprune(X,Y)
    # ShowTree.drawtree(Dtree,'TreeCarID3.jpeg')
    predicted = clf.predict(Dtree, x_test)
    rightNum = 0
    for i in xrange(len(predicted)):
        if predicted[i] == y_test[i]:
            rightNum += 1
    print rightNum, len(predicted)

    print '准确率:%f' % float(rightNum * 1.0 / len(predicted))
    print "preprune success"
    print '---------------------------------------------------'
if __name__ == '__main__':
    X,Y = processdata.readDataFromWine()

    # holdoutMethod(X, Y)
    # KFoldCrossValidation(X, Y)
    myBootstrap(X, Y)
    # mypreprune(X, Y)
    # mypostprune(X, Y)
    # AdaBoost(X, Y)

    X,Y = processdata.readDataFromIris()

    # holdoutMethod(X, Y)
    # KFoldCrossValidation(X, Y)
    myBootstrap(X, Y)
    # mypreprune(X, Y)
    # mypostprune(X, Y)
    # AdaBoost(X, Y)

    X, Y = processdata.readDataFromLR()

    # holdoutMethod(X, Y)
    # KFoldCrossValidation(X, Y)
    # myBootstrap(X, Y)


    myBootstrap(X, Y)
    # mypreprune(X, Y)
    # mypostprune(X,Y)
    # AdaBoost(X, Y)

    # holdoutMethod(X, Y)
    # KFoldCrossValidation(X, Y)
    # myBootstrap(X,Y)
