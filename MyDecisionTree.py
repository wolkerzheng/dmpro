#encoding=utf8
import numpy as np
import copy
class   DesicionTreeNode:
    def __init__(self, col=-1, val=None,labelDict = None,results=None, tb=None, fb=None):
        self.col = col
        self.val = val
        self.results = results
        self.labelDict = labelDict
        self.tb = tb
        self.fb = fb
       
class DesicionTree:
    def __init__(self,mode='GainInfo'):
        self.mode = mode
        # self.FeatColList = []

    def calEntropy(self, y):
        '''
        功能：calEntropy用于计算香农熵 e=-sum(pi*log pi)
        参数：其中y为数组array
        输出：信息熵entropy
        '''
        n = len(y)
        labelCounts = {}
        for label in y:
            if label not in labelCounts.keys():
                labelCounts[label] = 1
            else:
                labelCounts[label] += 1
        entropy = 0.0
        for key in labelCounts:
            prob = float(labelCounts[key]) / n
            entropy -= prob * np.log2(prob)
        return entropy
    def calMisClassificationError(self,Y):
        """
        计算误分类错误率
        :param Y:
        :return:
        """
        n = len(Y)
        labelCount = {}
        for label in Y:
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1
        error = 1.0
        maxProb = 0
        for key in labelCount:
            prob = float(labelCount[key]) / n
            if maxProb <prob:
                maxProb = prob
        return error-maxProb**2

    def calGini(self,Y):
        """
        计算gini指数：gini = 1 - sum(prob**2)
        :param Y:
        :return:
        """
        n = len(Y) #label的长度
        labelCount = {}
        for label in Y :
            if label not in labelCount.keys():
                labelCount[label] = 0
            labelCount[label] += 1
        gini = 1.0
        for key in labelCount:
            prob = float(labelCount[key]) / n
            gini -= prob**2
        return gini

    def majorityCnt(self, labellist):
        """
        参数:labellist是类标签，序列类型为list
        输出：返回labellist中出现次数最多的label
        """
        labelCount = {}
        for vote in labellist:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1
        sortedClassCount = sorted(labelCount.iteritems(), key=lambda x: x[1], \
                                  reverse=True)
        return sortedClassCount[0][0]

    def testErro(self,tree,x_test,y_test):
        error = 0.0
        predicted = self.predict(tree,x_test)
        for i in range(len(x_test)):
            if predicted[i] != y_test[i]:
                error +=1
        return float(error)
    def testMajor(self,tree,x_test,y_test,labelList):
        """

        :param tree:
        :param x_test:
        :param y_test:
        :return:
        """
        ctree = copy.deepcopy(tree)
        ctree.tb ,ctree.fb = None,None
        ctree.results = self.majorityCnt(labelList)
        error = 0.0
        predicted = self.predict(ctree, x_test)
        for i in range(len(x_test)):
            if predicted[i] != y_test[i]:
                error += 1
        return float(error)

    def postprune(self,tree ,x_test,y_test):
        """

        :param tree:
        :param x_test:
        :param y_test:
        :return:
        """
        if tree.tb.results == None:

            self.postprune(tree.tb,x_test,y_test)
        if tree.fb.results == None:
            self.postprune(tree.fb,x_test,y_test)
        if tree.tb.results != None and tree.fb.results != None:
            # Build a combined dataset
            labelList = []
            for v, c in tree.tb.labelDict.items():
                for i in range(c):
                    labelList.append(v)
            for v, c in tree.fb.labelDict.items():
                for i in range(c):
                    labelList.append(v)
            if self.testErro(tree,x_test,y_test) >= self.testMajor(tree,x_test,y_test,labelList):
                tree.tb, tree.fb = None, None
                tree.results = self.majorityCnt(labelList)
                tree.labelDict = self.finalLabelDict(labelList)

    def preprune(self,X, Y):
        """

        :param X:
        :param Y:
        :return:
        """
        labelList = list(Y)
        if labelList.count(labelList[0]) == len(labelList):
            leaf = DesicionTreeNode()
            leaf.results = labelList[0]
            leaf.labelDict = {labelList[0]:len(labelList)}
            return leaf
        bestFeat = self.decideMode(X,Y)
        if bestFeat[0] == None:
            leaf = DesicionTreeNode()
            leaf.results = self.majorityCnt(labelList)
            leaf.labelDict = self.finalLabelDict(labelList)
            return leaf
        root = DesicionTreeNode()
        bestFeatIndex,bestFeatValue = bestFeat[0]
        TrueXSet,FalseXSet,TrueYSet,FalseYSet = bestFeat[1]
        Tset = copy.deepcopy(TrueYSet)
        Tset.extend(FalseYSet)
        if self.calEntropy(FalseYSet) >= self.calEntropy(TrueYSet)+self.calEntropy(FalseYSet):
            root.col = bestFeatIndex
            root.val = bestFeatValue
            root.tb = self.BuildTree(TrueXSet,TrueYSet)
            root.fb = self.BuildTree(FalseXSet,FalseYSet)
            return root
        else:
            leaf = DesicionTreeNode()
            leaf.results = self.majorityCnt(Tset)
            leaf.labelDict = self.finalLabelDict(Tset)
            return leaf

    def decideMode(self,X,Y):
        """
        根据模式选择相应特征分裂方法
        :param X:
        :param Y:
        :return:
        """
        if self.mode == 'Gini':
            bestFeat = self.chooseBestFeatureToSplit_Gini(X, Y)
        elif self.mode == 'GainInfo':
            bestFeat = self.chooseBestFeatureToSplit_GainInfo(X, Y)
        elif self.mode == 'Error':
            bestFeat = self.chooseBestFeatureToSplit_MIsError(X,Y)
        return  bestFeat

    def stopCondition(self,x):
        """

        :param x:
        :return:
        """
        xAraay = np.array(x)
        flag = True
        for i in range(len(x[0])):
            ay = xAraay[:,i]
            if len(set(ay)) <= 1:
                flag = False
        return flag
    def finalLabelDict(self,labellist):
        labelCount = {}
        for vote in labellist:
            if vote not in labelCount.keys():
                labelCount[vote] = 0
            labelCount[vote] += 1

        return labelCount
    def BuildTree(self,X, Y):
        """
        递归生长决策树
        :param X:
        :param Y:
        :return:
        """
        labelList = list(Y)
        if labelList.count(labelList[0]) == len(labelList):
            leaf = DesicionTreeNode()
            leaf.results = labelList[0]
            leaf.labelDict = {labelList[0]:len(labelList)}
            return leaf
        root = DesicionTreeNode()
        bestFeat = self.decideMode(X,Y)
        if bestFeat[0] == None:
            leaf = DesicionTreeNode()
            leaf.results = self.majorityCnt(labelList)
            leaf.labelDict = self.finalLabelDict(labelList)
            return leaf
        bestFeatIndex,bestFeatValue = bestFeat[0]
        TrueXSet,FalseXSet,TrueYSet,FalseYSet = bestFeat[1]
        root.col = bestFeatIndex
        root.val = bestFeatValue
        root.tb = self.BuildTree(TrueXSet,TrueYSet)
        root.fb = self.BuildTree(FalseXSet,FalseYSet)
        return root

    def chooseBestFeatureToSplit_MIsError(self,X, Y):
        """

                :param X:
                :param y:
                :return:
                """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        numFeat = X.shape[1]  # 获取列数目
        bestSplit = 1
        best_idx = -1
        best_val = 0
        best_res = []
        best_set = None
        best_colval = None
        for i in range(numFeat):
            featlist = X[:, i]  # 得到第i个特征对应的特征列
            uniqueVals = set(featlist)
            curEntropy = 0.0
            splitInfo = 0.0
            tmp = 0
            for value in uniqueVals:
                curEntropy = 0.0
                sub_x1, sub_x2, sub_y1, sub_y2 = self.splitDataSet(X, Y, i, value)
                prob = len(sub_y1) / float(len(Y))  # 计算某个特征的某个值的概率
                curError = prob * self.calMisClassificationError(sub_y1) + (1 - prob) * self.calMisClassificationError(sub_y2)  # 迭代计算gini指数
                if curError < bestSplit and len(sub_y1) > 0 and len(sub_y2) > 0:
                    bestSplit = curError
                    best_idx = i
                    best_colval = [best_idx, value]
                    best_set = [sub_x1, sub_x2, sub_y1, sub_y2]
        best_res = [best_colval, best_set]
        return best_res


    def chooseBestFeatureToSplit_Gini(self,X,y):
        """

        :param X:
        :param y:
        :return:
        """
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        numFeat = X.shape[1]   #获取列数目
        bestSplit = 1
        best_idx = -1
        best_val = 0
        best_res = []
        best_set = None
        best_colval = None
        for i in range(numFeat):
            featlist = X[:, i]  # 得到第i个特征对应的特征列
            uniqueVals = set(featlist)
            curEntropy = 0.0
            splitInfo = 0.0
            tmp = 0
            for value in uniqueVals:
                curEntropy = 0.0
                sub_x1, sub_x2,sub_y1,sub_y2 = self.splitDataSet(X, y, i, value)
                prob = len(sub_y1) / float(len(y))          # 计算某个特征的某个值的概率
                curGini = prob * self.calGini(sub_y1)+ (1-prob)*self.calGini(sub_y2)  # 迭代计算gini指数
                if curGini < bestSplit and len(sub_y1) > 0 and len(sub_y2) > 0:
                    bestSplit = curGini
                    best_idx = i
                    best_colval=[best_idx,value]
                    best_set = [sub_x1, sub_x2,sub_y1,sub_y2]
        best_res = [best_colval,best_set]

        return best_res


    def chooseBestFeatureToSplit_GainInfo(self, X, y):
        """ID3 & C4.5
        参数：X为特征，y为label
        功能：根据信息增益或者信息增益率来获取最好的划分特征
        输出：返回最好划分特征的下标
        """
        if not isinstance(X,np.ndarray):
            X = np.array(X)
        numFeat = X.shape[1]   #获取列数目
        baseEntropy = self.calEntropy(y)
        bestSplit = 0.0
        best_idx = -1
        best_val = 0
        best_res = []
        best_set = None
        best_colval = None
        for i in range(numFeat):
            featlist = X[:, i]  # 得到第i个特征对应的特征列
            uniqueVals = set(featlist)
            curEntropy = 0.0
            splitInfo = 0.0
            tmp = 0
            for value in uniqueVals:
                curEntropy = 0.0
                sub_x1, sub_x2,sub_y1,sub_y2 = self.splitDataSet(X, y, i, value)
                prob = len(sub_y1) / float(len(y))  # 计算某个特征的某个值的概率
                curEntropy = prob * self.calEntropy(sub_y1)	+ (1-prob)*self.calEntropy(sub_y2)  # 迭代计算条件熵
                IG = baseEntropy - curEntropy
                if IG > bestSplit and len(sub_y1) > 0 and len(sub_y2) > 0:
                    bestSplit = IG
                    best_idx = i
                    best_colval=[best_idx,value]
                    best_set = [sub_x1, sub_x2,sub_y1,sub_y2]
        best_res = [best_colval,best_set]

        return best_res

    def splitDataSet(self,  X, y, index,  value):
        """

        :param X:
        :param y:
        :param index:
        :param value:
        :return:
        """
        split_function = None
        if isinstance(value, int) or isinstance(value, float):
            split_function = lambda row: row[index] <= value
        else:
            split_function = lambda row: row[index] == value
        Xset1, Xset2,Yset1,Yset2 = [],[],[],[]
        # Divide the rows into two sets and return them
        for i in xrange(len(X)):
            if split_function(X[i]):
                Xset1.append(X[i])
                Yset1.append(y[i])
            else:
                Xset2.append(X[i])
                Yset2.append(y[i])
        return Xset1, Xset2,Yset1,Yset2

    def classfy(self,tree,observation):
        if tree.results != None:
            return tree.results
        else:
            v = observation[int(tree.col)]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v <= tree.val:
                    branch = tree.tb
                else:
                    branch = tree.fb
            else:
                if v == tree.val:
                    branch = tree.tb
                else:
                    branch = tree.fb
            return self.classfy( branch,observation)

    def predict(self,tree,observation):
        """

        :param tree:
        :param observation:
        :return:
        """
        if tree == None:
            raise Exception('未建立决策树')
            return None
        if len(observation) < 1:
            raise Exception('测试数据不对')
            return None
        if len(observation) == 1:
            return self.classfy(tree,observation[0])
        else:
            results = []

            for i in range(len(observation)):
                tmp = self.classfy(tree,observation[i])
                results.append(tmp)
            return results

