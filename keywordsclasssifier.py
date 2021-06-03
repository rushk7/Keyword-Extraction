import pickle
from data_extraction import get_data
import string
from functions import *

import re
import string
import nltk
from nltk.tag import StanfordNERTagger
from nltk.collocations import *
from gensim import corpora, models, similarities
from collections import defaultdict
import wikiwords
import numpy as np
from nltk.corpus import stopwords
stoplist = stopwords.words('english')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def remove_punctuation(text):
    """
    Returns text free of punctuation marks
    """
    exclude = set(string.punctuation)
    return ''.join([ch for ch in text if ch not in exclude])


def to_tfidf(documents):
    """
    Returns documents transformed to tf-idf vector space
    """
    texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]for document in documents]
    
    
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    tfidf = models.TfidfModel(corpus,normalize=True)
    corpus_tfidf = tfidf[corpus]    

    return {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}


def keywords_classifier(preload) :
    
    print ("executing keyword_classifier")
    if preload == 1 :
        train_XY = pickle.load(open('saved/trainXY_crowd500.pkl','rb'))
        test_XY = pickle.load(open('saved/testXY)_crowd500.pkl','rb'))
    else :
        # get training data from crowd500 corpus
        traindata = get_data('train')
        tx_traindata = to_tfidf(traindata['documents'])
        train_XY = functions.get_feature_labels(traindata,tx_traindata['corpus'],tx_traindata['dictionary'],1)
        pickle.dump(train_XY, open('saved/trainXY_crowd500.pkl','wb'))    
  
        # get test data from crowd500 corpus
        testdata = get_data('test')

        # use tf-idf dictionary learned on training data to transform test data
        dictionary = tx_traindata['dictionary']
        tfidf = tx_traindata['tfidf_model']
        texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist]
                  for document in testdata['documents']]
        corpus = [dictionary.doc2bow(text) for text in texts]
        corpus_tfidf = tfidf[corpus]    
        tx_testdata = {'dictionary':dictionary, 'corpus':corpus_tfidf, 'tfidf_model': tfidf}

        test_XY = functions.get_feature_labels(testdata,tx_testdata['corpus'],tx_testdata['dictionary'],1)
        pickle.dump(test_XY, open('saved/testXY_crowd500.pkl','wb'))

                        
def get_keyword_classifier(preload,classifier_type) :
    print ("executing get_keyword_classifier")
    if preload == 1:
        train_XY = pickle.load(open('saved/trainXY_crowd500.pkl','rb'))
        test_XY = pickle.load(open('saved/testXY_crowd500.pkl','rb'))
        
        if classifier_type == 'logistic' :
            model = pickle.load(open('saved/logisticregression_crowd500.pkl','rb'))
        else :
            model = pickle.load(open('saved/randomforest_crowd500.pkl','rb'))
            
    else :
        
        traindata = get_data('train')
        tx_traindata = to_tfidf(traindata['documents'])
        
        train_XY = get_feature_labels(traindata,tx_traindata['corpus'],tx_traindata['dictionary'],1)
        pickle.dump(train_XY,open('saved/trainXY_crowd500.pkl','wb'))
        
        
        testdata = get_data('test')
        
        dictionary = tx_traindata['dictionary']
        
        tfidf = tx_traindata['tfidf_model']
        
        texts = [[remove_punctuation(word) for word in document.lower().split() if word not in stoplist ]
                     for document in testdata['documents'] ]
        
        corpus = [dictionary.doc2bow(text) for text in texts ]
        
        corpus_tfidf = tfidf[corpus]
        
        tx_testdata = {'dictionary' : dictionary , 'corpus' : corpus_tfidf , 'tfidf_model' : tfidf }
        
        test_XY = get_feature_labels(testdata,tx_testdata['corpus'],tx_testdata['dictionary'],1)
        pickle.dump(test_XY,open('saved/testXY_crowd500.pkl','wb'))
        
        
        
        if classifier_type == 'logistic' :
            model = LogisticRegression()
            model = model.fit(train_XY['features'],train_XY['labels'])
            pickle.dump(model,open('saved/logisticregression_crowd500.pkl','wb'))
        else :
            model = RandomForestClassifier(n_estimators = 10)
            model = model.fit(train_XY['features'],train_XY['labels'])
            
            pickle.dump(model,open('saved/randomforest_crowd500.pkl','wb'))
            
        in_sample_acc = cross_val_score(model,train_XY['features'],train_XY['labels'],cv=4)
        out_sample_acc = cross_val_score(model , test_XY['features'],test_XY['labels'],cv = 4)
        
        print ('--------------------------------------------------------------')
        
        if classifier_type == 'logistic' : 
            print ('using logistic regresision model for keyword classification(0 = non-keyword,1 = keyword)')
        else :
            print ('using random forest model for keyword classification(0 = non_keyword , 1 = keyword)' )
            
        
    return {'model' : model , 'train_XY' : train_XY , 'test_XY' : test_XY }
    
    
    
    
    
def evaluate_keywords(proposed , groundtruth ) :
    print ("executing evaluate_keywords")
    proposed_set = set(proposed)
    true_set = set(groundtruth)
    
    true_positives = len(proposed_set.intersection(true_set))
    
    if len(proposed_set) == 0 :
        precision = 0
    else :
        precision = true_positives/float(len(proposed))
        
    if len(true_set) == 0:
        recall = 0
    else :
        recall = true_positives/float(len(true_set))
        
    if precision + recall > 0:
        f1 = 2*precision*recall/float(precision + recall )
    else : 
        f1 = 0
        
        
    return (precision,recall,f1)
            
            
                                
        