
import numpy

from PyML.classifiers.baseClassifiers import Classifier
from PyML.classifiers.svm import modelDispatcher, LinearSVModel
from PyML.evaluators import resultsObjects

class RidgeRegression (Classifier) :
    """
    A kernel ridge regression classifier

    :Keywords:
      - `ridge` -- the ridge parameter [default: 10.0]
      - `kernel` -- a kernel object [default: Linear]
      - `regression` -- whether to use the object for regression [default: False]
        in its default (False), it is used as a classifier
    """

    attributes = {'ridge' : 1.0,
                  'kernel' : None,
                  'regression' : False,
                  'fit_bias' : True}

    def __init__(self, arg = None, **args) :
    
        Classifier.__init__(self, arg, **args)
        if self.regression :
            self.resultsObject = resultsObjects.RegressionResults
            self.classify = self.decisionFunc

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'ridge: %f\n' % self.ridge
            
        return rep
                    
        
    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        self.data = data
        
        if self.kernel is not None :
            self.data.attachKernel(self.kernel)
        if data.kernel.__class__.__name__ == 'Linear' :
            self.train_linear(data)
        else :
            self.train_nonlinear(data)
            
        self.log.trainingTime = self.getTrainingTime()

    def train_linear(self, data) :

        if len(data) < data.numFeatures :
            self.train_nonlinear(data)
            return
        print 'training in primal'
        if self.fit_bias :
            data.addFeature('bias', [1.0 for i in range(len(data))])
        self.w = numpy.zeros(data.numFeatures)
        self.bias = 0.0
        Y = numpy.array(data.labels.Y)
        if not (self.regression) :
            Y = Y * 2 - 1
        X = data.getMatrix()
        self.w = numpy.linalg.solve(X.T.dot(X) + numpy.eye(data.numFeatures), X.T.dot(Y))
        # there are alternative ways of computing the weight vector which are not
        # as computationally efficient:
        #self.w = np.dot(np.linalg.inv(data.X.T.dot(data.X)), X.T.dot(Y))
        #self.w = np.dot(np.linalg.pinv(data.X), Y)
        if self.fit_bias :
            data.eliminateFeatures([data.numFeatures -1])
            self.bias = self.w[-1]
            self.w = self.w[:-1]
        self.model = LinearSVModel(data, w = self.w, b = float(self.bias))
            
    def train_nonlinear(self, data) :

        Y = numpy.array(data.labels.Y)
        if not (self.regression) :
            Y = Y * 2 - 1
        K = data.getKernelMatrix()
        K = K + self.ridge * numpy.eye(len(data))
        self.alpha = numpy.linalg.solve(K, Y)
        # you can also do it this way, but not as efficient:
        #self.alpha = numpy.dot(Y, numpy.linalg.inv(K))
        self.model = modelDispatcher(data, svID=range(len(data)), alpha=self.alpha, b = 0.0)

        
    classify = Classifier.twoClassClassify

    def decisionFunc(self, data, i) :

        return self.model.decisionFunc(data, i)


