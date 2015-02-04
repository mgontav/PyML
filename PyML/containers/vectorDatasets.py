
import numpy
import os

from PyML.containers.baseDatasets import WrapperDataSet, BaseVectorDataSet
from PyML.containers.labels import Labels
from PyML.utils import arrayWrap,misc
from ext import csparsedataset,cvectordataset
from PyML.utils import myio

class BaseCVectorDataSet (WrapperDataSet, BaseVectorDataSet) :
    """A base class for vector dataset containers implemented in C++"""

    def __init__(self) :
        if self.__class__.__name__ == 'SparseDataSet' :		
            self.container = csparsedataset.SparseDataSet
        elif self.__class__.__name__ == 'VectorDataSet' :
            self.container = cvectordataset.VectorDataSet

    def copy(self, other, patterns) :
        """
        copy a wrapper dataset

        :Parameters:
          - `other` - the other dataset
          - `patternsToCopy` - a list of patterns to copy
        """
    
        if patterns is None :
            patterns = range(len(other))
        self.container.__init__(self, other, patterns)
        self.featureDict = other.featureDict.copy()
        self.featureID = other.featureID[:]

        
    def initializeDataMatrix(self, numPatterns, numFeatures) :

        self.container.__init__(self, numPatterns)


    def addPattern(self, x, i) :

        if type(x) == type({}) :
            keys,values = arrayWrap.dict2vectors(x)
        elif type(x) == type(numpy.array(1)) or type(x) == type([]) :
            keys = arrayWrap.longVector([])
            values = arrayWrap.doubleVector(x)
        else:
            raise TypeError,"data vectors must be dictionary, list or arrays"
        self.container.addPattern(self, keys, values)

    def addFeature(self, id, values) :
        """
        Add a feature to a dataset.

        :Parameters:
          - `id` - the id of the feature
          - `values` - list of values

        """
        if len(values) != self.size() :
            raise ValueError, \
                'number of values provided does not match dataset size'
        if type(id) == type(1) : 
            id = str(id)
        hashID = hash(id)
        if not hasattr(self, 'featureKeyDict') :
            self.addFeatureKeyDict()
        if hashID in self.featureKeyDict :
            raise ValueError, 'Feature already exists, or hash clash'
        if type(values) != type([]) :
            values = [v for v in values]

        self.container.addFeature(self, hashID, values)
        self.updateFeatureDict(id)
        
    def addFeatures(self, other) :
        """
        Add features to a dataset using the features in another dataset

        :Parameters:
          - `other` - the other dataset
        """

        if len(other) != len(self) :
            raise ValueError, 'number of examples does not match'
        if not hasattr(self, 'featureKeyDict') :
            self.addFeatureKeyDict()
        for id in other.featureID :
            if hash(id) in self.featureKeyDict :
                raise ValueError, 'Feature already exists, or hash clash'
        self.container.addFeatures(self, other)
        self.updateFeatureDict(other)


    def getPattern(self, i) :

        if i < 0 or i >= len(self) :
            raise ValueError, 'Index out of range'
        return self.container.getPattern(self, i)
        
    def extendX(self, other, patterns) :

        self.container.extend(self, other, patterns)

    def eliminateFeatures(self, featureList):
        """eliminate a list of features from a dataset
        INPUT:
        featureList - a list of features to eliminate; these are numbers
        between 0 and numFeatures-1 (indices of features, not their IDs)"""

        if len(featureList) == 0 : return
        if type(featureList[0]) == type('') :
            featureList = self.featureNames2IDs(featureList)
        featureList.sort()
        if type(featureList) != list :
            featureList = list(featureList)
        if max(featureList) >= self.numFeatures or min(featureList) < 0 :
            raise ValueError, 'Bad feature list'
        cfeatureList = arrayWrap.intVector(featureList)
        self.container.eliminateFeatures(self, cfeatureList)
        self.updateFeatureDict(featureList)
        
    def scale(self, w) :
        """rescale the columns of the data matrix by a weight vector w:
        set X[i][j] = X[i][j] * w[j]
        """

        if type(w) == type(1.0) :
            w = [w for i in range(self.numFeatures)]
        if type(w) != type([]) :
            w = list(w)
            #numpy.ones(self.numFeatures, numpy.float_) * w
        self.container.scale(self, w)

    def translate(self, c) :
        
        if type(c) != type([]) :
            c = list(c)
        self.container.translate(self, c)

    def mean(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.mean(self, cpatterns)

    def std(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.standardDeviation(self, cpatterns)

    def featureCount(self, feature, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)    
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.featureCount(self, feature, cpatterns)

    def featureCounts(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)        
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.featureCounts(self, cpatterns)

    def nonzero(self, feature, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)        
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index goes outside of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.nonzero(self, feature, cpatterns)

    def commonFeatures(self, pattern1, pattern2) :

        return [self.featureKeyDict[featureKey] for featureKey in
                self.container.commonFeatures(self, pattern1, pattern2)]
        
    def normalize(self, norm=2) :

        norm = int(norm)
        if norm not in [1,2] :
            raise ValueError, 'bad value for norm'
        self.container.normalize(self, norm)


class VectorDataSet (BaseCVectorDataSet, cvectordataset.VectorDataSet) :

    def __init__(self, arg = None, **args):
        BaseCVectorDataSet.__init__(self)
        BaseVectorDataSet.__init__(self, arg, **args)

    def addPattern(self, x, i) :

        if type(x) == type(numpy.array(1)) or type(x) == type([]) :
            values = arrayWrap.doubleVector(x)
        else:
            raise TypeError, "data vectors must be list or array"
        self.container.addPattern(self, values)
        

    def updateFeatureDict(self, arg = None) :

        if arg.__class__ == self.__class__ :   
            # features were extended with those in another dataset
            other = arg
            self.featureID.extend(other.featureID)
        elif type(arg) == list :
            #features were eliminated:
            eliminated = misc.list2dict(arg)
            self.featureID = [self.featureID[i] for i in range(len(self.featureID))
                              if i not in eliminated]
        elif type(arg) == type(1) or type(arg) == type('') :
            # a feature was added
            id = arg
            self.featureID.append(id)
            self.featureDict[id] = self.numFeatures - 1
            return

        self.featureDict = {}
        for i in range(len(self.featureID)) :
            self.featureDict[self.featureID[i]] = i

class SparseDataSet (BaseCVectorDataSet, csparsedataset.SparseDataSet) :

    def __init__(self, arg = None, **args):
        BaseCVectorDataSet.__init__(self)
        BaseVectorDataSet.__init__(self, arg, **args)

    def updateFeatureDict(self, arg = None) :
        
        if arg.__class__ == self.__class__ :
            other = arg
            self.featureID.extend(other.featureID)
            self.featureID.sort(cmp = lambda x,y : cmp(hash(x), hash(y)))
        elif type(arg) == list :
            #features were eliminated:
            eliminated = misc.list2dict(arg)
            self.featureID = [self.featureID[i] for i in range(len(self.featureID))
                              if i not in eliminated]
        elif type(arg) == type(1) or type(arg) == type('') :
            # a feature was added:
            id = arg
            self.featureID.append(id)
            self.featureID.sort(cmp = lambda x,y : cmp(hash(x), hash(y)))

        self.featureDict = {}
        self.featureKeyDict = {}
        for i in range(len(self.featureID)) :
            self.featureDict[self.featureID[i]] = i
            self.featureKeyDict[hash(self.featureID[i])] = i


class PyVectorDataSet (BaseVectorDataSet) :
    """A non-sparse dataset container that uses a numpy array"""

    #def __init__(self, arg = None, **args):
        #BaseVectorDataSet.__init__(self, arg, **args)

    def __len__(self) :
        """the number of patterns in the dataset"""

        if self.X is not None :
            return len(self.X)
        else :
            raise ValueError, "no data here!"

    def getNumFeatures(self) :

        return len(self.featureID)

    def setNumFeatures(self, value) :

        raise ValueError, 'do not call this function!'

    numFeatures = property (getNumFeatures, setNumFeatures,
                            None, 'The number of features in a dataset')

    def fromArrayAdd(self, X) :

        self.X = numpy.asarray(X)

    def updateFeatureDict(self) :

        pass
    
    def dotProduct(self, i, j, other = None) :

        x = self.X[i]
        if other is not None :
            y = other.X[j]
        else :
            y = self.X[j]
        return numpy.dot(x, y)
        
    def initializeDataMatrix(self, numPatterns, numFeatures) :

        self.X = numpy.zeros((numPatterns, numFeatures), numpy.float)

    def addPattern(self, x, i) :

        if type(x) == type({}) :
            raise ValueError, 'wrong type of argument for addPattern'
            for key in x :
                self.X[i][key] = x[key]
        else :
            for j in range(len(x)) :
                self.X[i][j] = x[j]

    def addFeature(self, featureID, values) :
        if len(values) != len(self) :
            raise ValueError, 'wrong number of inputs'
        newX = numpy.zeros(( len(self), self.numFeatures + 1), numpy.float)
        newX[:, :-1] = self.X
        for i in range(len(self)) :
            #newX[i, :-1] = self.X[i]
            newX[i, -1] = values[i]
        self.featureID.append(featureID)
        self.X = newX

    def getPattern(self, i) :

        return self.X[i]
    
    def extendX(self, other, patterns) :

        X = self.X
        self.X = numpy.zeros((len(self) + len(patterns), len(self.numFeatures)),
                               numpy.float)
        for i in range(len(X)) :
            self.X[i] = X[i]
        for i in patterns :
            self.X[i + len(X)] = other.X[i]

    def featureIDcompute(self) :

        pass

    def copy(self, other, patternsToCopy) :
        """performs a deepcopy from the given dataset"""

        if patternsToCopy is None :
            patternsToCopy = range(len(other))
        self.X = other.X[patternsToCopy]
        self.featureID = other.featureID[:]
        #self.featureKey = other.featureKey[:]
        #self.featureKeyDict = copy.deepcopy(other.featureKeyDict)

    def eliminateFeatures(self, featureList) :
        """eliminate a list of features from a dataset
        Input:
        featureList - a list of features to eliminate; these are numbers
        between 0 and numFeatures-1 (indices of features, not their IDs)"""

        #if len(featureList) == 0 : return
        #if type(featureList[0]) == type('') :
        #    featureList = self.featureNames2IDs(features)
        featuresToTake = misc.setminus(range(self.numFeatures), featureList)
        featuresToTake.sort()
        self.featureID = [self.featureID[i] for i in featuresToTake]
        #self.featureKey = [self.featureKey[i] for i in featuresToTake]
        #self.featureKeyDict = {}
        #for i in range(len(self.featureKey)) :
        #    self.featureKeyDict[self.featureKey[i]] = i        
        
        self.X = numpy.take(self.X, featuresToTake, 1)

    def getFeature(self, feature, patterns = None) :

        if patterns is None :
            patterns = range(len(self))
        values = numpy.zeros(len(patterns), numpy.float)
        for i in range(len(patterns)) :
            values[i] = self.X[i][feature]

        return values

    def norm(self, pattern, p = 1) :

        if p == 1 :
            return numpy.sum(numpy.absolute(self.X[pattern]))
        elif p == 2 :
            return math.sqrt(numpy.sum(numpy.dot(self.X[pattern])))
        else :
            raise ValueError, 'wrong value of p'

    def normalize(self, p = 1) :
        """normalize dataset according to the p-norm, p=1,2"""
        
        for i in range(len(self)) :
            norm = self.norm(i, p)
            if norm == 0 : continue
            self.X[i] = self.X[i] / norm

    def scale(self, w) :
        """rescale the columns of the data matrix by a weight vector w:
        set X[i][j] = X[i][j] / w[j]
        """
        
        self.X = self.X * w

    def translate(self, c) :

        self.X = self.X - numpy.resize(c, (len(self), len(c)))

    def mean(self, patterns = None) :

        if patterns is None or len(patterns) == len(self) :
            return numpy.mean(self.X, 0)
        else :
            return numpy.mean(self.X[patterns], 0)
        
    def std(self, patterns = None) :
        
        if patterns is None or len(patterns) == len(self) :
            return numpy.std(self.X, 0)
        else :
            return numpy.std(self.X[patterns], 0)

    def featureCount(self, feature, patterns = None) :

        if patterns is None :
            patterns = range(len(self))

        count = 0
        for p in patterns :
            if self.X[p][feature] != 0 : count+=1
        
        return count

    def featureCounts(self, patterns = None) :

        if patterns is None :
            patterns = range(len(self))
        
        counts = numpy.zeros(self.numFeatures)
        for i in patterns :
            counts += numpy.not_equal(self.X[i], 0)

        return counts

def load_libsvm_format(file_name, **args) :
    """
    Load a dataset from a file in libsvm format
    returns an instance of PyVectorDataSet
    If you want to use the data with a SparseDataSet, you can directly
    do it using the SparseDataSet constructor.
    """

    regression = False
    if 'regression' in args :
        regression = args['regression']
    # first extract labels and check how many features there are:
    labels = []
    num_features = 0
    if not os.path.exists(file_name) :
        raise ValueError, "file doesn't exist at %s" % file_name
    file_handle = myio.myopen(file_name)
    for line in file_handle :
        tokens = line.split()
        if regression :
            labels.append(float(tokens[0]))
        else :
            labels.append(str(int(float(tokens[0]))))
        for token in tokens[1:] :
            id,value = token.split(':')
            num_features = max(num_features, int(id))
    X = numpy.zeros((len(labels), num_features), numpy.float)
    # fill in the array:
    i = 0
    for line in open(file_name) :
        tokens = line.split()
        for token in tokens[1:] :
            id,value = token.split(':')
            id = int(id) - 1
            X[i][id] = float(value)
        i+=1
    data = PyVectorDataSet(X)
    if regression :
        labels = Labels(labels, numericLabels=True)
    else :
        labels = Labels(labels)
    data.attachLabels(labels)
    return data

