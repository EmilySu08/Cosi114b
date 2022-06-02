# -*- coding: utf-8 -*-
import numpy as np 
import scipy.linalg as scipy_linalg
import re 
#function that creates the co-occurence matrix 
def semantic_word_vector(textfile):  
    feature_dict = {}
   
    #reading in the file
    with open(textfile) as f:
        lines = f.readlines()
    #create feature dictionary
    for line in lines:
        # Split the line into words
        words = line.split()
        # Iterate over each word in line
        for word in words:
            # Check if the word is already in dictionary
            if word in feature_dict:
                feature_dict[word] = feature_dict[word] + 1
            else:
                feature_dict[word] = 1
   # print(lines)
    #initialize co-occurrence matrix
    C = np.zeros((len(feature_dict),len(feature_dict)), dtype = int) 
    #filling in  co-occurrence matrix
    features_list = list(feature_dict.keys())
    #print(feature_dict)
    for w in range(0, len(features_list)):
       for c in range(0, len(features_list)): 
           if w == c:
               C[w,c] = feature_dict[features_list[w]]
           else:    
               for j in range(len(lines)):
                   w_c = features_list[w] + ' ' + features_list[c]
                   c_w = features_list[c] + ' ' + features_list[w]
                   if (w_c in lines[j]):
                       v = r"\b(?=\w)" + re.escape(w_c) + r"\b(?!\w)"
                       C[w,c] = C[w,c] + len(re.findall(v,lines[j]))
                   elif (c_w in lines[j]):
                       v = r"\b(?=\w)" + re.escape(c_w) + r"\b(?!\w)"
                       C[w,c] = C[w,c] + len(re.findall(v,lines[j]))
    C = C * 10  + 1
    return C, features_list 
#This function creates the Positive Pointwise Mutual Information matrix 
#following the formula 
def ppmi(C):
    PPMI = np.zeros((len(C),len(C)))
    total_sum = np.sum(C)
   # print(total_sum)
    sum_row = np.sum(C, axis = 1)
   # print(sum_row)
    sum_col = np.sum(C, axis = 0)
    #print(sum_col)
    #print(C[0, :])
    for w in range(0, len(C)):
        for c in range(0, len(C)):
            num = C[w,c] / total_sum 
            #CALCULATE 
            denom = (sum_row[w] *sum_col[c]) / (total_sum*total_sum)
            PPMI[w,c] = max(np.log2(num/denom), 0)
    return PPMI
#This function that calculates the euclidean distance between the given 2 nouns 
#and returns the results in an array. The elements in the array correspond 
#to the order in which they were asked to be calculated.
def euclidean_distance(C, feature_list):
    distances = []
    women_vec = C[feature_list.index('women')]
    men_vec = C[feature_list.index('men')]
    like_vec = C[feature_list.index('like')]
    dogs_vec = C[feature_list.index('dogs')]
    feed_vec = C[feature_list.index('feed')]
    bite_vec = C[feature_list.index('bite')]
    
    w_v_m = scipy_linalg.norm(women_vec - men_vec)
    w_v_a = scipy_linalg.norm(women_vec - dogs_vec)
    m_v_a = scipy_linalg.norm(men_vec - dogs_vec)
    f_v_l = scipy_linalg.norm(feed_vec - like_vec)
    f_v_b = scipy_linalg.norm(feed_vec - bite_vec)
    l_v_b = scipy_linalg.norm(like_vec - bite_vec)
    #
    distances = [w_v_m, w_v_a, m_v_a, f_v_l, f_v_b, l_v_b]
    return distances
#This function does some minor tweaks to the PPMI matrix by finding the SVD 
#and returning a reduced PPMI matrix
def alter(p):
    #decompostion 
    U, E, Vt = scipy_linalg.svd(p, full_matrices=False)
    E = np.diag(E) # compute E
    print(np.allclose(p, U.dot(E).dot(Vt)))
    V = Vt.T # compute V = conjugate transpose of Vt
    reduced_PPMI = p.dot(V[:, 0:3])
    return reduced_PPMI
#This main method runs the code 
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    swv, fl = semantic_word_vector('dist_sim_data.txt')   
    print("The Co-occurence matrix is: ")
    print(swv)
    print()
    p = ppmi(swv)
    print("The PPMI matrix is:")
    print(p)
    print()
    ed = euclidean_distance(p, fl)
    print("The original distances are " + str(ed))
    print()
    verify = alter(p)
    #print(E)
    print("The reduced PPMI matrix is:")
    print(verify)
    print()
    #calculate euclidean distance on reduced PPMI 
    new_ed = euclidean_distance(verify, fl)
    print("The reduced distances are " + str(new_ed))