import os
import re
import nltk
import string 
import pickle
def remove_punctuation(text):
    """
    Returns text free of punctuation marks
    """
    exclude = set(string.punctuation)
    return ''.join([ch for ch in text if ch not in exclude])

def get_data(status) : 
    
    print ("executing get_data")
    
    documents = []
    all_keywords = []
    
    path = 'CorpusAndCrowdsourcingAnnotations/' + status + '/'
    files = [f[:-4] for f in os.listdir(path) if re.search('\.key',f)]
    
    if status == 'test' :
    
        documents = pickle.load(open(path + 'scraped_testdata.pkl','rb')) # scraped webpages in test set
        skip_these = [3,7,14,19,26,27,32,33,43,45] # these webpages no longer exist, cannot find source text
        
    
    for file_idx in range(len(files)):
        if status=='train':

            # original text
            f = open(path + files[file_idx] + '.txt',encoding = "utf8")
            text = f.read()
            f.close()  

     # encoding issues in Crowd500  
            try:
                text = text.encode('utf-8')
                sentences = nltk.sent_tokenize(text.lower())        
            except:
                text = text.decode('utf-8')
                sentences = nltk.sent_tokenize(text.lower())   
      
            documents.append(text)

            # keywords
            keywords = []
            with open(path + files[file_idx] + '.key','r') as f:
                for line in f:
                    keywords.append(line.strip('\n'))            
            keywords = [remove_punctuation(k.lower()) for k in keywords]
            all_keywords.append(keywords)

        else:
            if file_idx not in skip_these:
                keywords = []
                with open(path + files[file_idx] + '.key','r') as f:
                    for line in f:
                        keywords.append(line.strip('\n'))            
                keywords = [remove_punctuation(k.lower()) for k in keywords]
                all_keywords.append(keywords)
  
    return {'documents':documents, 'keywords':all_keywords}
        