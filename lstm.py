# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import preprocess
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
max_seq_len = 75
k = 800
mapping = [-1,0,1]


class LSTMs():
    """
    LSTM for sentiment classification
    """
    
    def __init__(self,num_labels=3,lstm_hidden=(400,400),rnn_type='LSTM'):
        #Create input/output and weights
        self.input = tf.placeholder(tf.float32, [None, max_seq_len,k+2])
        self.output = tf.placeholder(tf.float32, [None, num_labels])
        self.keep_prob = tf.placeholder(tf.float32,[3])
        

        cells = []
        for num_hidden in lstm_hidden:
            if(rnn_type=='LSTM'):
                cell = rnn_cell.BasicLSTMCell(num_hidden,state_is_tuple=True)
            else:
                cell = rnn_cell.GRUCell(num_hidden)        #self.keep_prob
            celldropout = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.keep_prob[0],output_keep_prob=self.keep_prob[1],state_keep_prob=self.keep_prob[2],seed=0)
            cells.append(celldropout)
        cell = rnn_cell.MultiRNNCell(cells , state_is_tuple=True)
        val, state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)
        valT = tf.transpose(val,[1,0,2])
        
        last = tf.gather(valT,int(valT.get_shape()[0])-1)
        
        weight_out = tf.Variable(tf.truncated_normal([num_hidden, int(self.output.get_shape()[1])]))
        bias_out = tf.Variable(tf.constant(0.1, shape=[self.output.get_shape()[1]]))
        
        #Get prediction at last tiem setp
        prediction = tf.nn.softmax(tf.matmul(last,weight_out)+bias_out)
        self.prediction = prediction
        
        #Get prediction at every timestep
        allpredictions = tf.transpose(tf.map_fn(lambda x: tf.nn.softmax(tf.matmul(x,weight_out)+bias_out), valT, dtype=tf.float32),[1,0,2])
        self.all_predictions = allpredictions

        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if 'bias' not in v.name ]) * 0.001

        #loss = -tf.reduce_sum(self.output * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.output,logits=prediction)) #+ lossL2
        self.loss = loss
        
        
        
        predictClass = tf.argmax(prediction, 1)
        self.predictClass = predictClass

        #Calculate accuracy
        correct = tf.equal(tf.argmax(self.output, 1), tf.argmax(prediction, 1))
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.accuracy = acc

        #Calculate confusion matrix
        confusion_matrix = tf.contrib.metrics.confusion_matrix(tf.argmax(self.output, 1),predictClass)
        self.confusion_matrix = confusion_matrix

    def load_model(self,model_name='LSTM_44_Test',step='350'):
        # with tf.Graph().as_default():

        """
        Usable model:
            -../model/checkpoint/yoyoyo/ch-2280 :[[263  14   7  11] , Validation accuracy: 0.74, default structure.  P,R,F1 = (0.72,0.74,0.72)
                                                 [ 25 258   8   9]  40 epochs
                                                 [ 16  39  35  41]  400,400
                                                 [ 30  24  17 117]]

            -../model/checkpoint/LSTM_Default
        """

        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_count={'GPU': 0})
        sess = tf.Session(config=session_conf)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        self.global_step = global_step
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './sentiment_seg/model/checkpoint/' + model_name + '/ch-' + step)
        print("Model Restored")
        self.sess = sess

    #Create function that will predict and calculate accuracy, f1, precision,recall,confusion matrix
    def evaluate(self,X_comments,y,sample_size=10,y_mapping=[-32,-1,0,1]):
        X = preprocess.comments_to_matrix(X_comments)
        #TODO: Change from default model to model specify by user.
        try:
            self.sess
        except Exception:
            print("Model not initialized yet. Please load model first using load_model() function.")
            # sess.run(tf.global_variables_initializer())

        # Restoring Model
        #
        # saver = tf.train.Saver(tf.global_variables())
        # saver.restore(self.sess, '../model/checkpoint/-411')
        # print("Model Restored")
        print("Start evaluating...")
        current_step = tf.train.global_step(self.sess, self.global_step)
        print("Global step: %d"%current_step)
        feed_dict = {self.input: X, self.output: y,self.keep_prob:[1,1,1]}
        acc,l,pred = self.sess.run([self.accuracy,self.loss,self.predictClass], feed_dict=feed_dict)

        print("Accuracy: %.2f"%acc)
        print("Loss: {}".format(l))

        #Confusion matrix

        #y_labels = np.full([y.shape[0]],32)
        y_labels = np.full([y.shape[0]],-32)
        for i, m in enumerate(y_mapping):
            y_labels[y[:, i] == 1] = m
        pred = [y_mapping[i] for i in pred]


        cm = confusion_matrix(y_labels, pred)
        print(cm)

        #Calculate precision,recall per class
        precisions,recalls,fscores,supports = precision_recall_fscore_support(y_labels,pred)
        for i,sent in enumerate(["Negative","Neutral","Positive"]):
            print("Class: %s"%sent)
            print("Precision: %.2f"%precisions[i])
            print("Recall: %.2f"%recalls[i])
            print("F1 scores: %.2f"%fscores[i])
            print("================================")
        precision, recall, fscore, support = precision_recall_fscore_support(y_labels, pred,average='weighted')
        print("Weighted Score")
        print("Precision: %.2f" % precision)
        print("Recall: %.2f" % recall)
        print("F1 scores: %.2f" % fscore)


        print("Printing Sample Predicted comments")
        print("===========================")
        idx = np.random.choice(np.arange(X.shape[0]), sample_size, replace=False)
        sample_x = X[idx]
        comments = X_comments[idx]
        sample_y = y[idx]
        test_feed_dict = {self.input:sample_x,self.output:sample_y,self.keep_prob:[1,1,1]}
        test_pred,all_preds = self.sess.run([self.predictClass,self.all_predictions],test_feed_dict)



        correctPrediction = 0
        for i in range(len(test_pred)):
            pred = test_pred[i]
            current_pred_seq = all_preds[i][-len(comments[i]):]
            print(len(comments[i]))
            #Prin word with color by prediction
            for j,word in enumerate(comments[i]):
                word_pred = current_pred_seq[j]
                printSegmentColor(word,word_pred)
            predictionLabel = y_mapping[pred]
            realLabel = y_mapping[np.argmax(sample_y[i])]
            print(", %.2f Ans: %.2f"%(predictionLabel,realLabel),end=' ')
            if(predictionLabel==realLabel):
                print("O")
                correctPrediction+=1
            else:
                print("X")
        print("Samples accuracy: {}".format(round(correctPrediction/sample_size)))

