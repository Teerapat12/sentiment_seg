# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None
max_seq_len = 75
max_word_num = 75
k = 800
from gensim import models
model = models.Word2Vec.load('./sentiment_seg/model/fbcomment.w2v')


########################Set up#########################
######################################################


def comments_to_matrix(comments,verbose=False):
    comment_size = comments.shape[0]
    X = np.zeros([comment_size,max_seq_len,k+2])
    
    for i,comment in enumerate(comments):
        for j in range(max_seq_len):
            if(max_seq_len-j>len(comment)): # Should not be word yet
                if(verbose):print("PAD",end=' ')
                X[i][j][-2] = 1
            else: 
                if(verbose):print(comment[j-(max_seq_len-len(comment))],end=' ')
                try:
                    X[i][j] = np.concatenate((model[comment[j-(max_seq_len-len(comment))]],np.zeros([2])))
                except Exception:
                    if(verbose):print("UNK",end=' ')
                    X[i][j] = np.concatenate((np.zeros([k+1]),[1]))
    if(verbose):print(i)
    return X

