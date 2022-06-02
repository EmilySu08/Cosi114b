# CS114B Spring 2022 Homework 1
# Naive Bayes Classifier and Evaluation

import os
import numpy as np
from collections import defaultdict
import math

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
        p_count = 0
        n_count = 0
        V_pos = {}
        V_neg = {}
        
        keys = list(self.class_dict.keys())
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            #print(files)
            #print()
            for name in files:
                with open(os.path.join(root, name), encoding="utf8", errors='ignore') as f:
                    #print(root)
                    if keys[1] in root:
                        p_count = p_count + 1
                        for line in f:
                            # Split the line into words
                            words = line.split()
                            # Iterate over each word in line
                            for word in words:
                                # Check if the word is already in dictionary
                                if word in V_pos:
                                    V_pos[word] = V_pos[word] + 1
                                else:
                                    V_pos[word] = 1
                    
                    elif keys[0] in root:
                        n_count = n_count + 1
                        for line in f:
                            # Split the line into words
                            words = line.split()
                            # Iterate over each word in line
                            for word in words:
                                # Check if the word is already in dictionary
                                if word in V_neg:
                                    V_neg[word] = V_neg[word] + 1
                                else:
                                    V_neg[word] = 1
                    
        ##number of total documents and creating self.prior
        D = p_count + n_count
        self.prior = np.zeros(len(self.class_dict))
        self.prior[0] = math.log(n_count/D)
        self.prior[1] = math.log(p_count/D)
        
        #create self.feature.dict  
        self.feature_dict = V_neg.copy()
        for k in V_pos:
            if k not in self.feature_dict:
                self.feature_dict[k] = 0
        p = 0        
        for key in self.feature_dict:
            self.feature_dict[key] = p 
            p = p + 1
        self.likelihood = np.zeros((len(self.class_dict), len(self.feature_dict)))
        
        pos_sum = sum(V_pos.values())
        neg_sum = sum(V_neg.values())
        for key in self.feature_dict:
            if key in V_pos:
                p = (V_pos[key] + 1) / (pos_sum + len(self.feature_dict))
                self.likelihood[1, self.feature_dict[key]] = math.log(p)
            else:
                p = 1 / (pos_sum  + len(self.feature_dict))
                self.likelihood[1, self.feature_dict[key]] = math.log(p)
            if key in V_neg:
                n  = (V_neg[key] + 1) / (neg_sum + len(self.feature_dict))
                self.likelihood[0,self.feature_dict[key]] = math.log(n)
            else:
                n  = 1 / (neg_sum + len(self.feature_dict))
                self.likelihood[0, self.feature_dict[key]] = math.log(n)
        return self.likelihood, self.prior, self.feature_dict

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
        # iterate over testing documents
        for root, dirs, files in os.walk(dev_set):
            for name in files:
                with open(os.path.join(root, name), encoding="utf8", errors='ignore') as f:
                    results[name]['predicted'] = ''
                    results[name]['correct'] = root[-3:]
                    #creating dictionary for each document
                    d = {}
                    for line in f:
                        # Split the line into words
                        words = line.split()
                        # Iterate over each word in line
                        for word in words:
                            # Check if the word is already in dictionary
                            if word in d:
                                d[word] = d[word] + 1
                            else:
                                d[word] = 1
                    #creating the feature vector                
                    fd_copy = self.feature_dict
                    for k in fd_copy:
                        if k in d:
                            fd_copy[k] = d[k]
                        else:
                            fd_copy[k] = 0 
    
                    #arg_max = 0
                    data = list(fd_copy.values())
                    
                    dot = np.dot(self.likelihood, data)
                    SUM = np.add(self.prior, dot)
                    arg_max = np.argmax(SUM)
                    if (arg_max == 1):
                        #print('p')
                        results[name]['predicted'] = keys[1] 
                    else : 
                        #print('n')
                        results[name]['predicted'] = keys[0] 
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
        for v in results.values():
            dics.append(v)
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
        #print(TN)
        #print(TP)
        #print(FP)
        #print(FN)
        #Calculating stats
        
        total = TN + TP + FP + FN
        accuracy = (TN + TP) / total  # needs to be calculated once 
        #for negative class
        neg_precision = TN / (TN + FN)
        neg_recall =  TN / (TN + FP)
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
        
if __name__ == '__main__':
    nb = NaiveBayes()
    # make sure these point to the right directories
    #print(len(nb.feature_dict))
    nb.train('movie_reviews/train')
    #nb.train('movie_reviews_small/train')
    results = nb.test('movie_reviews/dev')
    #results = nb.test('movie_reviews_small/test')
    nb.evaluate(results)
