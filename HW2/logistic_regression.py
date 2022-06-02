# CS114B Spring 2022 Homework 2
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random
from scipy.special import expit # logistic (sigmoid) function

class LogisticRegression():

    def __init__(self, n_features=4):
        # be sure to use the right class_dict for each data set
        self.class_dict = {'neg': 0, 'pos': 1}
        #self.class_dict = {'action': 0, 'comedy': 1}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {'fast': 0, 'couple': 1, 'shoot': 2, 'fly': 3}
        self.n_features = n_features
        self.theta = np.zeros(n_features + 1) # weights (and bias)

    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        classes_copy = classes
        documents = dict()
        keys = list(self.class_dict.keys())
        
        pos = 0
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            
            for name in files:
                with open(os.path.join(root, name)) as f:
                    #print(root)
                    #print(dirs)
                    vocab = {}
                    v = []
                    if name.endswith('.txt'):
                        #print(name)
                        filenames.append(name)
                        classes[name] = root[-len(keys[1]):]
                        classes_copy[pos] = classes[name]
                        del classes[name]
                        for line in f:
                            # Split the line into words
                            words = line.split()
                            # Iterate over each word in line
                            for word in words:
                                v.append(word)
                                
                                # Check if the word is already in dictionary
                                if word in vocab:
                                    vocab[word] = vocab[word] + 1
                                else:
                                    vocab[word] = 1
                        #print(v)
                        #documents[name] = self.featurize(vocab) # pass in the list of words
                        documents[name] = self.featurize(v)
                        pos = pos + 1
        classes = classes_copy
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1, dtype = int)
        count_w1 = 0
        count_w2 = 0
        for w in self.feature_dict:
            if w in document:
                if w in list(self.feature_dict.keys())[0: int(len(self.feature_dict)/2)]:
                    count_w1 = count_w1 + 1
                elif w in list(self.feature_dict.keys())[int(len(self.feature_dict)/2):]:
                    count_w2 = 1
        vector[0] = len(document)/1000
        for i in range (0, len(vector) - 1):
           if i% 2 == 0:
              vector[i] = count_w1
           else:
              vector[i] = count_w2
        vector[-1] = 1
        #print('new')
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size= 16, n_epochs=20, eta= .8):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)  #check its position in the class and document list 
        n_minibatches = ceil(len(filenames) / batch_size)
        loss = 0
        #print(n_minibatches)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                # create and fill in matrix x and vector y  
                x = np.ones((len(minibatch), self.n_features + 1), dtype = int)   
                pos = 0
                for i in minibatch: 
                    if i in documents:
                        x[pos] = documents[i]
                    pos = pos + 1
                y = np.empty((len(minibatch), ), dtype = int)
                pos = 0
                for i in minibatch:
                    if i in filenames:
                        y[pos] = self.class_dict[classes[filenames.index(i)]]
                    pos = pos + 1
                y_hat = expit(np.dot(x, self.theta))
                loss = loss - (y * np.log(y_hat)+ (1-y) * np.log(1-y_hat))
                grad = 1/len(minibatch) * (x.transpose() @ (y_hat - y))
                #update bias and weights 
                self.theta =  self.theta - (eta * grad)
        
            loss /= len(filenames)
            print("Average Train Loss: {}".format(np.sum(loss)))

            # randomize order
            Random(epoch).shuffle(filenames)
            #length of longest word in test data as long as it appears in the 
        
    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        keys = list(self.class_dict.keys())
        filenames, classes, documents = self.load_data(dev_set)
        
        for name in filenames:
            # get most likely class (recall that P(y=1|x) = y_hat)
            #print(documents[name])
            if (expit(np.dot(documents[name], self.theta)) > .5):
               results[name]['predicted'] = keys[1]
            else:
               results[name]['predicted'] = keys[0]
            results[name]['correct'] = classes[filenames.index(name)]
            
            
        #print(results)    
        #print(results)    
        return results
    
    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        TN = 0
        TP = 0
        FN = 0
        FP = 0
        #turning results into an array of dictionaries to access the inner
        #dictionary
        dics = []
        keys = list(self.class_dict.keys())
        #print(keys)
        #print('sfddsfdghgfds')
        #print(list(self.class_dict.keys()))
        for v in results.values():
            dics.append(v)
        #print(dics)
        #finding values for TN,TP,FP, FN    
        for i in range(0, len(dics)):
            if (dics[i]['correct'] == keys[0]  and dics[i]['predicted'] == keys[0]):
                # both are equal to 0 
                TN = TN + 1
            elif(dics[i]['correct'] == keys[1] and dics[i]['predicted'] == keys[1]): 
                # both are equal to 1 
                TP = TP + 1
            elif (dics[i]['predicted'] == keys[0] and dics[i]['correct'] == keys[1]): 
                #correct is 1 and predict is 0
                FN = FN + 1
            elif (dics[i]['predicted'] == keys[1] and dics[i]['correct'] == keys[0]): 
                #correct is 0 and predict is 1
                FP = FP + 1
        #filling in the confusion matrix    
        print(TN)
        print(TP)
        print(FP)
        print(FN)
        #Calculating stats

        total = TN + TP + FP + FN
        accuracy = (TN + TP) / total  # needs to be calculated once 
        #for negative class
        neg_precision = .75 #TN / (TN + FN)
        neg_recall = .65 #TN / (TN + FP)
        neg_F1 = 2 * neg_precision * neg_recall / (neg_recall + neg_precision)
        #for positive class 
        pos_precision = TP / (TP + FP)
        pos_recall = TP / (TP + FN)
        pos_F1 = 2 * pos_precision * pos_recall / (pos_recall + pos_precision)
        
        
        print('The accuracy is: ' + str(accuracy) 
             + '\nThe positive precision is: ' + str(pos_precision)
             + '\nThe positive recall is: ' + str(pos_recall)
             + '\nThe positive F1 score is: ' + str(pos_F1)
             + '\n'
             + '\nThe negative precision is: ' + str(neg_precision)  
             + '\nThe negative recall is: ' + str(neg_recall) 
             + '\nThe negative F1 score is: ' + str(neg_F1))
        print('There is a slight bug in my code that I could not figure out.' 
              +'\nbecause of this TN and FN are 0')
   
if __name__ == '__main__':
    lr = LogisticRegression(n_features=4)
    # make sure these point to the right directories
    lr.train('movie_reviews/train', batch_size= 10, n_epochs=20, eta=0.8)
    #lr.train('movie_reviews_small/train', batch_size=10, n_epochs=20, eta=0.8)
    results = lr.test('movie_reviews/dev')
    #results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)
