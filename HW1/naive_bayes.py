# CS114B Spring 2022 Homework 1
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict

class NaiveBayes():

    def __init__(self):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        #self.class_dict = {'action': 0, 'comedy': 1}
        self.feature_dict = {}
        self.prior = None
        self.likelihood = None

    '''
    Trains a multinomial Naive Bayes classifier on a training set.
    Specifically, fills in self.prior and self.likelihood such that:
    self.prior[class] = log(P(class))
    self.likelihood[class][feature] = log(P(feature|class))
    '''
    def train(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # collect class counts and feature counts
                    pass
        # fill in feature_dict
        # normalize counts to probabilities, and take logs

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # create feature vectors for each document
                    pass
                # get most likely class
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict),len(self.class_dict)))
        pass

if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    nb.train('movie_reviews/train')
    #nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
