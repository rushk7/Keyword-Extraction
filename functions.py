from gensim import corpora, models, similarities

import nltk
from nltk.corpus import stopwords

import random
from pymongo import MongoClient 
import subprocess
import multiprocessing
import concurrent.futures

import numpy as np

from nltk.tag import StanfordNERTagger
from nltk.collocations import *
from nltk import *
import string 
stoplist = stopwords.words('english')
import time

import wikiwords
from data_extraction import get_data
import pickle
from sklearn import preprocessing

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

def extract_candidate_words(text, good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    print ("executing extract_candidate_words")
    import itertools, nltk, string

    # exclude candidates that are stop words or entirely punctuation
    punct = set(string.punctuation)
    stop_words = set(nltk.corpus.stopwords.words('english'))
    # tokenize and POS-tag words
    tagged_words = itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text)))
    # filter on certain POS tags and lowercase all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags and word.lower() not in stop_words
                  and not all(char in punct for char in word)]

    return candidates
    
def score_keyphrases_by_textrank(text, n_keywords=0.5):
    """
    extracting the candiates for the articles using textrank algorithm ,
     not good enough and it aslo takes a n_keywords as a parameter to extract the keywords

    """
    print ("executing score_keyphrases_by_textrank")
    from itertools import takewhile, tee
    import networkx, nltk
    import wikiwords 
    
    # tokenize for all words, and extract *candidate* words
    words = [word.lower()
             for sent in nltk.sent_tokenize(text)
             for word in nltk.word_tokenize(sent)]
    candidates = extract_candidate_words(text)
    # build graph, each node is a unique candidate
    graph = networkx.Graph()
    graph.add_nodes_from(set(candidates))
    # iterate over word-pairs, add unweighted edges into graph
    def pairwise(iterable):
        """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
    for w1, w2 in pairwise(candidates):
        if w2:
            graph.add_edge(*sorted([w1, w2]))
    # score nodes using default pagerank algorithm, sort by score, keep top n_keywords
    ranks = networkx.pagerank(graph)
    if 0 < n_keywords < 1:
        n_keywords = int(round(len(candidates) * n_keywords))
    word_ranks = {word_rank[0]: word_rank[1]
                  for word_rank in sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:n_keywords]}
    keywords = set(word_ranks.keys())
    # merge keywords into keyphrases
    keyphrases = {}
    j = 0
    for i, word in enumerate(words):
        if i < j:
            continue
        if word in keywords:
            kp_words = list(takewhile(lambda x: x in keywords, words[i:i+10]))
            avg_pagerank = sum(word_ranks[w] for w in kp_words) / float(len(kp_words))
            keyphrases[' '.join(kp_words)] = avg_pagerank
            # counter as hackish way to ensure merged keyphrases are non-overlapping
            j = i + len(kp_words)
    
    return sorted(keyphrases.items(), key=lambda x: x[1], reverse=True) 

def get_namedentities(text) :
    """
    Returns named entities in text using StanfordNERTagger
    """
    st = StanfordNERTagger('utils/english.conll.4class.caseless.distsim.crf.ser.gz','utils/stanford-ner.jar')

    ner_tagged = st.tag(text.lower().split())     
  
    named_entities = []
    if len(ner_tagged) > 0:
        for n in ner_tagged:
            if n[1]!='O':
                named_entities.append(remove_punctuation(n[0]))

    named_entities = [n for n in named_entities if n] 
    return named_entities

def get_nounphrases(text) :
    """
    Returns noun phrases in text
    """
    grammar = r""" 
        NBAR:
            {<NN.*|JJ>*<NN.*>}  
        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}   # from Alex Bowe's nltk tutorial
    """    
    chunker = nltk.RegexpParser(grammar)
    sentences = nltk.sent_tokenize(text.lower())
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    noun_phrases = []
    for sent in sentences:
        tree = chunker.parse(sent)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP': 
                noun_phrases.extend([w[0] for w in subtree.leaves()])

    noun_phrases = [remove_punctuation(nphrase) for nphrase in noun_phrases]
    noun_phrases = [n for n in noun_phrases if n]

    return noun_phrases

def get_trigrams(text,num_trigrams):
    """
    Return all members of most frequent trigrams
    """
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(text.lower().split())
    finder.apply_freq_filter(1) # ignore trigrams that occur only once
    top_ngrams = finder.nbest(trigram_measures.pmi,num_trigrams)
  
    ngrams = []
    for ng in top_ngrams:
        ngrams.extend(list(ng))    

    ngrams = [remove_punctuation(n) for n in list(set(ngrams))]
    ngrams = [n for n in ngrams if n]
    return ngrams


def generate_keywords(text) :
    """
    Returns candidate words that occur in named entities, noun phrases, or top trigrams
    """
    num_trigrams = 5
    named_entities = get_namedentities(text)
    noun_phrases = get_nounphrases(text)
    top_trigrams = get_trigrams(text,num_trigrams)

    return list(set.union(set(named_entities),set(noun_phrases),set(top_trigrams)))

def get_capitalized(text,candidate_keywords) :
    """
    Returns a 0/1 encoding indicating if any occurence of keyword included 
    capitalization
    """
    words_original = [remove_punctuation(w) for w in text.split()]
    words_lower = [remove_punctuation(w) for w in text.lower().split()]
  
    caps = []
    for candidate in candidate_keywords:
        occurences = [pos for pos,w in enumerate(words_lower) if w == candidate]
        if len(occurences)>0:
            any_caps = sum([1 for o in occurences if words_original[o]!=words_lower[o]])
            if any_caps>0:
                caps.append(1)
            else:
                caps.append(0)
        else:
            caps.append(0)
  
    return caps

def get_tfidf(candidate_keywords,corpus_entry,dictionary) :
    """
    Returns tf-idf scores for keywords using a tf-idf transformation of 
    the text containing keywords
    """
    weights = []
    if corpus_entry:
        for candidate in candidate_keywords:
            if candidate in dictionary.token2id:
                tfidf_score = [w[1] for w in corpus_entry if w[0]==dictionary.token2id[candidate]]
                if len(tfidf_score)>0:
                    weights.append(tfidf_score[0])
                else:
                    weights.append(0)
            else:
                weights.append(0)
    else:
        weights = [0]*len(candidate_keywords)

    return weights


def get_googled_value(candidate) :
    
    url = "mongodb://localhost:27017/mydb"
    client = MongoClient(url)

    mydb = client.mydb
    
    collection =  mydb.googled_words
    google_values = []
    #print ('kvbjkxbv')
    my_collection = mydb['googled_words']
    resultant_value = 0.0
    #for candidate in candidates :
       # start_time = time.time()
    #resultant_value = 0
    found = 0
    if mydb.final_words.find({"word" : candidate}).count() == 0 :
        print ("not found in final_words")
        # if mydb.googled_words.find({"word" : candidate}).count() == 0 :
        #     print("not found in googled_words")

        #     #inserting the word in the not_found_yet database
        #     mydb.not_found_yet.insert_one({"word" : candidate})
        #     #calling the javascript file to get the word trend from google
        #     subprocess.call(['./call_js.sh'])
            
        #     mydb.not_found_yet.delete_many({})
        #     #google_values.append()
        #     length = 0


        #     for word in collection.find({"word" : candidate}) :
        #         print ("this is happeniong ")
        #         length = len(word['data'])
        #         print (length)
        #         for i in range(length) :
        #             resultant_value += word['data'][i]['value'][0]
        #         #print (word['value'][0])
        #     if length != 0 :    
        #         resultant_value = resultant_value / length
        #     else :
        #         resultant_value = 0
        #     #print (resultant_value)
        #     google_values.append(resultant_value)
        #     print ("")
            
            
        # else :
        #     print("found in googled_words")
        #     length = 0
        #     resultant_value = 0
        #     x = 0

        #     for word in collection.find({"word" : candidate}) :
        #         if x == 0 :
        #             length = len(word['data'])
        #             print("is this happening")
        #             print (length)
        #             for i in range(length) :
        #                 resultant_value += word['data'][i]['value'][0]
        #                 print (word['data'][i]['value'][0])
        #         x += 1

        #     if length != 0 :
        #         resultant_value = resultant_value /length
        #     else :
        #         resultant_value = 0
        #     #print (resultant_value)
        #     google_values.append(resultant_value)
        # mydb.final_words.insert_one({"word" : candidate,"resultant_value" : resultant_value});
    else :
        resultant_value = 0.0
        print("found in final_words")
        found = 1
        for words in mydb.final_words.find({"word" : candidate}) :
            if 'resultant_value' in words.keys() :
                resultant_value = words['resultant_value']
            #print ("mil gaya re")
        
        #print('Time Tane for this word : ' % (time.time() - start_time))  
        # if resultant_value == 0 :
            
            
        #     print("not found in googled_words")

        #     #inserting the word in the not_found_yet database
        #     mydb.not_found_yet.insert_one({"word" : candidate})
        #     #calling the javascript file to get the word trend from google
        #     subprocess.call(['./call_js.sh'])
            
        #     mydb.not_found_yet.delete_many({})
        #     #google_values.append()
        #     length = 0


        #     for word in collection.find({"word" : candidate}) :
        #         print ("this is happeniong ")
        #         length = len(word['data'])
        #         print (length)
        #         for i in range(length) :
        #             resultant_value += word['data'][i]['value'][0]
        #         #print (word['value'][0])
        #         if length != 0 :    
        #             resultant_value = resultant_value / length
        #         else :
        #             resultant_value = 0
        #         #print (resultant_value)
        #     google_values.append(resultant_value)
        #     print ("")
            
            
        
        #     #print (resultant_value)
        #     #google_values.append(resultant_value)
        #     mydb.final_words.insert_one({"word" : candidate,"resultant_value" : resultant_value})
            
            
    #print (resultant_value)``
    return found,resultant_value


            
def extract_candidate_features(text ,candidates, corpus_entry , dictionary):

    print ("executing extract_candidate_features")
    import collections, math, nltk, re
    
    
   
    
    # get word counts for document
    doc_word_counts = collections.Counter(word.lower()
                                          for sent in nltk.sent_tokenize(text)
                                          for word in nltk.word_tokenize(sent))
    """
    # wiki frequency goes like this ... 
    fname = 'word_freq.txt'

    with open(fname) as f:
        content = f.readlines()
    data = {}
    for words in content :
        stars = words.split()
        data[stars[0]] = stars[1]
    
    """
    
    candidate_features  =  np.zeros((len(candidates),12))
    weights = get_tfidf(candidates,corpus_entry,dictionary)
    caps = get_capitalized(text,candidates)


    executor = concurrent.futures.ProcessPoolExecutor(10)
    futures = [executor.submit(get_googled_value, candidate) for candidate in candidates]
    concurrent.futures.wait(futures)
    num_candidates = len(candidates)
    got = 0
    total_got = 0
    for i ,candidate in enumerate(candidates):
        
        pattern = re.compile(r'\b'+re.escape(candidate) + r'(\b|[,;.!?]|\s)', re.IGNORECASE)
        
        # frequency-based
        # number of times candidate appears in document
        cand_doc_count = len(pattern.findall(text))
        # count could be 0 for multiple reasons; shit happens in a simplified example
        if not cand_doc_count:
            #print ('**WARNING:', candidate, 'not found!')
            continue
    
        # statistical
        candidate_words = candidate.split()
        if candidate_words :
            max_word_length = np.max((len(w) for w in candidate_words )) 
        else : 
            max_word_length = None 
            
        
        term_length = len(candidate_words)
        # get frequencies for term and constituent words
        sum_doc_word_counts = float(sum(doc_word_counts[w] for w in candidate_words))
        try:
            # lexical cohesion doesn't make sense for 1-word terms
            if term_length == 1:
                lexical_cohesion = 0.0
            else:
                lexical_cohesion = term_length * (1 + math.log(cand_doc_count, 10)) * cand_doc_count / sum_doc_word_counts
        except (ValueError, ZeroDivisionError) as e:
            lexical_cohesion = 0.0
        
        # positional
        # found in title, key excerpt
        
        # first/last position, difference between them (spread)
        doc_text_length = float(len(text))
        first_match = pattern.search(text)
        abs_first_occurrence = first_match.start() / doc_text_length
        
        
        if cand_doc_count == 1:
            spread = 0.0
            abs_last_occurrence = abs_first_occurrence
        else:
            for last_match in pattern.finditer(text):
                pass
            abs_last_occurrence = last_match.start() / doc_text_length
            spread = abs_last_occurrence - abs_first_occurrence
      
        wiki_freq = wikiwords.freq(candidate)
       
        candidate_features[i][0] =  cand_doc_count # term_count
        candidate_features[i][1] = term_length #term_length
      # candidate_features[i][2] = max_word_length #max_word_length
        candidate_features[i][3] = spread #spread
        candidate_features[i][4] = lexical_cohesion #lexical_cohesion
      # candidate_features[i][5] = abs_first_occurrence #abs_first_occurence
        candidate_features[i][6] = abs_last_occurrence#abs_last_occurence
        candidate_features[i][7] = wiki_freq # wiki_frequency
        candidate_features[i][8] = weights[i] #tfidf
        candidate_features[i][9] = caps[i] # capitalisation
        got,candidate_features[i][10] = get_googled_value(candidate)

        total_got += got

    

        
                                               
    print("the ratio of googled_words is " , (total_got/num_candidates))                                                            
  #   candidate_features[:][8]= get_capitalized(text,candidates)
  #   candidate_features[:][9]= get_tfidf(candidates,corpus_entry,dictionary)                     

    return candidate_features


# to implement as soon as possible
def extract_keywords(text,keyword_classifier,top_k,preload) :
    print("executing extract_keywords")
    if preload == 1:
        preprocessing = pickle.load(open('saved/tfidf_preprocessing.pkl','rb'))
        dictionary = preprocessing['dictionary']
        tfidf = preprocessing['tfidf_model']
        
    else :
        traindata = get_data('train')
        tx_traindata = to_tfidf(traindata['documents'])
        dictionary = tx_traindata['dictionary']
        tfidf = tx_traindata['tfidf_model']
        pickle.dump({'dictionary' : dictionary,'tfidf_model' : tfidf},open('saved/tfidf_preprocessing.pkl','wb'))

    text_processed = [remove_punctuation(word) for word in text.lower().split() if word not in stoplist]
    corpus = [dictionary.doc2bow(text_processed)]
    corpus_entry = tfidf[corpus][0]
    
    candidate_keywords_tuple =  score_keyphrases_by_textrank(text,0.9)
    length = len(candidate_keywords_tuple)

    candidate_keywords = []

    for idx in range(length) :
        candidate_keywords.append(candidate_keywords_tuple[idx][0])

    candidate_keywords = list(set(candidate_keywords))

    if len(candidate_keywords) < top_k :
        candidate_keywords = text_processed
    

    #candidate_keywords = generate_keywords(text)
    feature_set = extract_candidate_features(text,candidate_keywords,corpus_entry,dictionary)
    predicted_prob = keyword_classifier.predict_proba(feature_set)
    this_column = np.where(keyword_classifier.classes_ == 1)[0][0]
    sorted_indices = [i[0] for i in sorted(enumerate(predicted_prob[:,this_column]),key = lambda x :x[1],reverse = True)]
    chosen_keywords = [candidate_keywords[j] for j in sorted_indices[:top_k]]
    chosen_keywords = list(set(chosen_keywords))

    return chosen_keywords
        
    

def get_feature_labels(data,corpus,dictionary,verbose) :

    print ("executing get_feature_labels")
    num_docs = len(data['documents'])
        
    for doc_idx in range(num_docs) :
        text = data['documents'][doc_idx]
        keywords = data['keywords'][doc_idx]
            
        corpus_entry = corpus[doc_idx]
            
        separate_keywords = []
        for k in keywords :
            separate_keywords.extend(remove_punctuation(k.lower()).split())
                
                
        positive_examples = separate_keywords
        
        num_positive = len(positive_examples)
            
        all_words = [remove_punctuation(w) for w in text.lower().split()]
        
        negative_examples = [w for w in all_words if( w not in stoplist) and (w not in positive_examples) ]
        
        if len(negative_examples) > num_positive :
            negative_examples = random.sample(negative_examples,num_positive)
                
        num_negative = len(negative_examples)
            
        if num_positive < num_negative :
            candidate_keywords = positive_examples + random.sample(negative_examples,num_positive)
            labels = np.array([1]*num_positive + [0]*num_postive)
        elif num_positive > num_negative :
            candidate_keywords = random.sample(positive_examples,num_negative) + negative_examples
            labels = np.array([1]*num_negative + [0]*num_negative)
        else :
            candidate_keywords = positive_examples + negative_examples
            labels = np.array([1]*num_positive + [0]*num_negative)
            
        if doc_idx == 0 :
            all_labels = labels
        else :
            all_labels= np.concatenate((all_labels , labels),axis = 0)
                
        feature_set = extract_candidate_features(text ,candidate_keywords,corpus_entry,dictionary)
            
        if doc_idx == 0 :
            all_features = feature_set
        else :
            all_features = np.vstack((all_features,feature_set))
                
    #if verbose :
    #   print ( 'get_features_labels : extracted %d samples from document %d of %d' % (len(labels),doc_idx + 1,num_docs)
                       
                       
    return    {'features' : all_features , 'labels' : all_labels}
                                                                                                                                   
               
        
        
                                                                                                                                    