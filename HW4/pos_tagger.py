# CS114B Spring 2022 Homework 4
# Part-of-speech Tagging with Structured Perceptrons

import os
import numpy as np
from collections import defaultdict
from random import Random

class POSTagger():

    def __init__(self):
        # for testing with the toy corpus from the lab 8 exercise
        #self.tag_dict = {'nn': 0, 'vb': 1, 'dt': 2}
        #self.word_dict = {'Alice': 0, 'admired': 1, 'Dorothy': 2, 'every': 3,
                          #'dwarf': 4, 'cheered': 5}
        self.tag_dict = {}
        self.word_dict = {}
        """
        self.initial = np.array([-0.3, -0.7, 0.3])
        self.transition = np.array([[-0.7, 0.3, -0.3],
                                    [-0.3, -0.7, 0.3],
                                    [0.3, -0.3, -0.7]])
        self.emission = np.array([[-0.3, -0.7, 0.3],
                                  [0.3, -0.3, -0.7],
                                  [-0.3, 0.3, -0.7],
                                  [-0.7, -0.3, 0.3],
                                  [0.3, -0.7, -0.3],
                                  [-0.7, 0.3, -0.3]])
        """
        self.unk_index = -1

    #rsplit the last / 
    '''
    Fills in self.tag_dict and self.word_dict, based on the training data.
    '''
    def make_dicts(self, train_set):
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    lines = f.read().rsplit()
                    for item in lines:
                        word, tag = item.rsplit('/', 1)
                        
                        if word in self.word_dict:
                            self.word_dict[word] = self.word_dict[word] + 1
                        else:
                            self.word_dict[word] = 1 
                        if tag in self.tag_dict:
                            self.tag_dict[tag] = self.tag_dict[tag] + 1
                        else:
                            self.tag_dict[tag] = 1 

    '''
    Loads a dataset. Specifically, returns a list of sentence_ids, and
    dictionaries of tag_lists and word_lists such that:
    tag_lists[sentence_id] = list of part-of-speech tags in the sentence
    word_lists[sentence_id] = list of words in the sentence
    '''
    def load_data(self, data_set):
        sentence_ids = []
        num_total_sentences = 0
        tag_lists = dict()
        word_lists = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    numlines = [line.rstrip('\n') for line in f]
                    for line in numlines:
                        pairs = line.rsplit()
                        words = []
                        tags =[]
                        for w_t in pairs:
                            word, tag = w_t.rsplit('/', 1)
                            if word != [] and tag != []:
                                words.append(word)
                                tags.append(tag)
                                  
                        tag_lists[num_total_sentences] = tags
                        word_lists[num_total_sentences] = words
                         
                        #store the tags and words in terms of their indices, 
                        #using self.tag dict and self.word dict to translate 
                        #between them.
                        sentence_ids.append(num_total_sentences)   
                        num_total_sentences = num_total_sentences + 1
        return sentence_ids, tag_lists, word_lists

    '''
    Implements the Viterbi algorithm.
    Use v and backpointer to find the best_path.
    '''
    def viterbi(self, sentence):
        T = len(sentence)
        N = len(self.tag_dict)
        # initialization step
        v = np.zeros((N, T))
        backpointer = np.zeros((N, T), dtype=int)
      
        if T > 0:
            v[:, 0] = self.initial + self.emission[sentence[0]]
            backpointer[:, 0] = 0
            # recursion step
            for t in range (1, T):
                v[:, t] = np.max(v[:, t - 1, None] + self.transition + self.emission[sentence[t]], axis = 0)
                backpointer[: ,t] = np.argmax(v[:, t - 1,None] + self.transition + self.emission[sentence[t]], axis = 0)
            best_path_pointer = np.argmax(v[:,T-1], axis = 0)
            #time step backward
            best_path = np.zeros(T)
            best_path[0] = best_path_pointer 
            bp_ind = 1
        
            for i in range(T-1, 0, -1):
               best_path[bp_ind] = backpointer[int(best_path_pointer),i]
               best_path_pointer = backpointer[int(best_path_pointer),i]
               bp_ind += 1
            best_path = np.flip(best_path).tolist()
        else:
            best_path = [] 
    
        return best_path

    '''
    Trains a structured perceptron part-of-speech tagger on a training set.
    '''
    def train(self, train_set):
        self.make_dicts(train_set)
        sentence_ids, tag_lists, word_lists = self.load_data(train_set)
        
        Random(0).shuffle(sentence_ids)
        self.initial = np.zeros(len(self.tag_dict)) #+ 1e-8
        self.transition = np.zeros((len(self.tag_dict), len(self.tag_dict))) #+ 1e-8
        self.emission = np.zeros((len(self.word_dict) + 1, len(self.tag_dict))) #+ 1e-8

        for i, sentence_id in enumerate(sentence_ids):
            print(i)
            for w in range(0,  len(word_lists[sentence_id])):
                if word_lists[sentence_id][w] in list(self.word_dict.keys()):
                    word_lists[sentence_id][w] = list(self.word_dict.keys()).index(word_lists[sentence_id][w])
                else:
                    word_lists[sentence_id][w] = self.unk_index
                if tag_lists[sentence_id][w] in list(self.tag_dict.keys()):
                    tag_lists[sentence_id][w] = list(self.tag_dict.keys()).index(tag_lists[sentence_id][w])
                else:
                    tag_lists[sentence_id][w] = self.unk_index
            predicted = self.viterbi(word_lists[sentence_id])
            #print(predicted)
            if len(predicted) > 0:
                sentence_w = word_lists[sentence_id]
                sentence_t = tag_lists[sentence_id]
                for j in range (0, len(word_lists[sentence_id])):
                    if j > 0 and int(predicted[j]) != sentence_t[j]:
                        self.transition[sentence_t[j] - 1][sentence_t[j]] += 1
                        self.transition[int(predicted[j]) - 1][int(predicted[j])] += -1
                        self.emission[sentence_w[j]][sentence_t[j]] += 1
                        self.emission[sentence_w[j]][int(predicted[j])] += -1
                    else:
                        self.initial[sentence_t[j]] += 1
                        self.initial[int(predicted[j])] += -1
                        self.emission[sentence_w[j]][sentence_t[j]] += 1
                        self.emission[sentence_w[j]][int(predicted[j])] += -1
                            
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids):
                print(i + 1, 'testing sentences tagged')
          
    '''
    Tests the tagger on a development or test set.
    Returns a dictionary of sentence_ids mapped to their correct and predicted
    sequences of part-of-speech tags such that:
    results[sentence_id]['correct'] = correct sequence of tags
    results[sentence_id]['predicted'] = predicted sequence of tags
    '''
    def test(self, dev_set):
       # print()
        results = defaultdict(dict)
        sentence_ids, tag_lists, word_lists = self.load_data(dev_set)

        for i, sentence_id in enumerate(sentence_ids):
            for w in range(0,  len(word_lists[sentence_id])):
                if word_lists[sentence_id][w] in list(self.word_dict.keys()):
                    word_lists[sentence_id][w] = list(self.word_dict.keys()).index(word_lists[sentence_id][w])
                else:
                    word_lists[sentence_id][w] = self.unk_index
                if tag_lists[sentence_id][w] in list(self.tag_dict.keys()):
                    tag_lists[sentence_id][w] = list(self.tag_dict.keys()).index(tag_lists[sentence_id][w])
                else:
                    tag_lists[sentence_id][w] = self.unk_index
            results[sentence_id]['correct'] = tag_lists[sentence_id]
            predicted = self.viterbi(word_lists[sentence_id])
            #print('train',predicted)
        
            results[sentence_id]['predicted'] = self.viterbi(word_lists[sentence_id])
            if (i + 1) % 1000 == 0 or i + 1 == len(sentence_ids): 
                print(i + 1, 'testing sentences tagged')

        return results

    '''
    Given results, calculates overall accuracy.
    '''
    def evaluate(self, results):
        correct = []
        predicted = []
        match = 0
        for r in results:
            correct.append(results[r]['correct'])
            predicted.append(results[r]['predicted'])
        #flatten 2d list into 1d list
        correct = [j for sub in correct for j in sub]
        predicted = [j for sub in predicted for j in sub]
        #print(correct)
       # print(predicted)
        for i in range (0, len(correct)):
            if correct[i] == int(predicted[i]): 
                match += 1 
        accuracy = match/len(correct) * 100
        return accuracy

if __name__ == '__main__':
    pos = POSTagger()
    # make sure these point to the right directories
    #pos.train('brown/train')
    pos.train('data_small/train')
    #results = pos.test('brown/dev')
    results = pos.test('data_small/test')
    print('Accuracy:', pos.evaluate(results))
