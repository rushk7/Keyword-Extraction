from keywordsclasssifier import * 
from functions import *
from data_extraction import *
import pickle
import bson
import time


def main() :
    preload =1 
    classifier_type = 'logistic'
    top_k = 20
    
    with open(r"keyword_classifier.pkl","rb") as input_file :
        keyword_classifier = pickle.load(input_file)
    
    
    bs = open('Yrals_data/GistButton/0e56a30f1e3f3c1859f7c60019d22eca_1443106948987_data.bson', 'rb').read()
    
    yralsdata = []
    i = 0
    for valid_dict in bson.decode_all(bs):
    #if i == 0: 
       # print (valid_dict['text'])
        print('working')
        i += 1
        if i == 39876 or i == 35456:
            yralsdata.append(valid_dict['text'])
        if i > 40000:
            break
            
    num_docs = len(yralsdata)
    
    print (num_docs)
    for doc_idx in range(num_docs) :
        text = yralsdata[doc_idx]
        
        start_time = time.time()
        if doc_idx == 0 :
            suggested_keywords = extract_keywords(text,keyword_classifier,top_k,preload)
        else :
        
            suggested_keywords = extract_keywords(text,keyword_classifier,top_k,1)
        
        print ('article :')
        print (text)
        print ('keywords :')
        print (suggested_keywords)
    
        print ('time taken : ',time.time() - start_time)

if __name__ == '__main__' :
    main()
    
    