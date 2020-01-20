import numpy as np
import ipdb
import torch
from sklearn.metrics import f1_score
class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric_mod(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def eval_score(self, outputs, target, distance):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        # try:
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        # except:
            # ipdb.set_trace()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * round( float(self.correct) / self.total, 4)

    def name(self):
        return 'Class. Acc.'

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

def compute_simpleACC(y_pred, y_true):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    ipdb.set_trace()
    thres = np.mean(y_pred)
    thres = 1.0
    pred = y_pred.ravel() < thres
    return np.mean(pred == y_true)

class SimpleSiamDistAcc(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        # self.correct = 0
        # self.total = 0
        self.pred = 0
        self.thres = 1.0
        self.correct = 0
        self.total = 0
        self.sample_feedback = []
        # ipdb.set_trace()
    

    def eval_score(self, outputs, target, distance):

        # with torch.no_grad():
        # if distance == None:
            # distance = (output[1]- output[0]).pow(2).sum(1)  # squared distances
        self.outputs_np = distance.detach().cpu().numpy()
        self.target_np = target[0].detach().cpu().numpy()
        self.pred = self.outputs_np.ravel() < self.thres
        self.correct += np.sum(self.pred == self.target_np)
        self.total += len(self.pred)
            # fscore = f1_score(self.target_np, self.pred)
            # print("fscore:", fscore)
        return self.value()

    def reset(self):
        # self.correct = 0
        # self.total = 0
        self.pred = 0
        self.target_np = None
        self.outputs_np = None

    def value(self):
        return round(float(self.correct / self.total),4)

    def name(self):
        return 'EucDist {} Accuracy'.format(self.thres)

class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
