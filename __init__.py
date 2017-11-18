# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, './sentiment_seg/')

import preprocess

from lstm import LSTMs
model = LSTMs(4,(400,400))
model.load_model("LSTM_Default4Class_44","5800") #2280

import deepcut as dc

import numpy as np

mapping = {
    0:'Irr',
    1:'Neg',
    2:'Neu',
    3:'Pos'
}

def predict(commentString,ans=True):
    commentWords = dc.tokenize(commentString)
    if(len(commentWords)<=75):
        x = preprocess.comments_to_matrix(np.array([commentWords]))
        feed_dict = {model.input: x, model.keep_prob: [1, 1, 1]}
        y_all_preds = model.sess.run([model.all_predictions], feed_dict)
        print(len(y_all_preds[0][0]))
        print(len(commentWords))

        return [(commentWords[i],mapping[pred]) for i,pred in enumerate(np.argmax(y_all_preds[0][0][-len(commentWords):],1))]
    else:
        #TODO: use overlapping to cross check the result
        y_all_preds = []
        for i in range(0,len(commentWords),75):
            words = commentWords[i:i+75]
            x = preprocess.comments_to_matrix(np.array([words]))
            feed_dict = {model.input: x, model.keep_prob: [1, 1, 1]}
            preds = model.sess.run([model.all_predictions], feed_dict)[0][0][:len(words)]
            print(preds)
            y_all_preds.extend(np.argmax(preds,1))
        return [(commentWords[i],mapping[pred]) for i,pred in enumerate(y_all_preds)]
