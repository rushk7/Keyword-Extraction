from keywordsclasssifier import * 
from functions import *
from data_extraction import *
import pickle
import bson
import time
import nltk



def main() :
    preload = 1
    classifier_type = 'logistic'
    top_k  = 50
    verbose = 1
    
    
    keyword_classifier = get_keyword_classifier(preload, classifier_type)['model']
    
    testdata = get_data('test')
    num_docs = len(testdata['documents'])
    
    performance_data = np.zeros((num_docs,3))
    
    # for doc_idx in range(num_docs) :
    #     text = testdata['documents'][doc_idx]
    #     true_keyphrases = testdata['keywords'][doc_idx]
        
        
    #     true_keywords = []
    #     for phrase in true_keyphrases :
    #         true_keywords.extend(phrase.lower().split())
            
    #     if doc_idx == 0 :
    #         suggested_keywords = extract_keywords(text,keyword_classifier,top_k,preload)
    #     else :
    #         suggested_keywords = extract_keywords(text,keyword_classifier,top_k,1)
            
    #     (precision,recall ,f1score) = evaluate_keywords(suggested_keywords,true_keywords)
        
        
    #     performance_data[doc_idx,0] = precision
    #     performance_data[doc_idx,1] =recall
    #     performance_data[doc_idx,2] = f1score
        
    #     if verbose == 1:
    #         print ('Document %d of %d : f1-score for top- %d keywords extracted by model = %.4f' % (doc_idx+ 1,num_docs,top_k,f1score))
                   
    # print('-------------------------------------------------------')
    # print('Evaluation of keyword extraction model on Crowd500 test set')
    # print('Number of documents = %d, keywords extracted per document = %d ' % (num_docs,top_k))
    # print('Precision : Mean = %.4f , SEM = %.4f' % (np.mean(performance_data[:,0]),np.std(performance_data[:,0])/float(np.sqrt(num_docs))))
    # print('Recall : Mean = %.4f, SEM = %.4f' % (np.mean(performance_data[:,1]),np.std(performance_data[:,1])/float(np.sqrt(num_docs))))
    # print('F-1 score : Mean = %.4f , SEM = %.4f' % (np.mean(performance_data[:,2]),np.std(performance_data[:,2]/float(np.sqrt(num_docs)))))
                                                     
    # pickle.dump(performance_data,open('saved' + classifier_type + '_mode_evaluation.pkl','wb'))

    data = bson.BSON.encode({'a': 1})
    bs = open('Yrals_data/GistButton/0e56a30f1e3f3c1859f7c60019d22eca_1443106948987_data.bson', 'rb').read()
    
    english_vocab = set(w.lower() for w in nltk.corpus.words.words())

    yralsdata = []
    i = 0
    # for valid_dict in bson.decode_all(bs):
    # #if i == 0: 
    #    # print (valid_dict['text'])
    #     i = i +1
    #     if i % 3 == 0 :
    #         yralsdata.append(valid_dict['text'])
    #     if i > 50 :
    #         break
            
    # num_docs = len(yralsdata)
    
    
    # for doc_idx in range(num_docs) :
    #     text = yralsdata[doc_idx]
        
        
    #     print ('article :')
    #     print (text)
    #     print ('keywords :')
    #     for keyword in suggested_keywords :
    #         print (keyword)
    
    #     print ('time taken : ',time.time() - start_time)

    #     # for keyword in suggested_keywords :
    #     #     words  = keyword.split()

    #     #     length = len(words)
    #     #     flag = True
    #     #     for i in range(length) :
    #     #         if flag == False :
    #     #             continue
    #     #         if words[i] not in english_vocab :
    #     #             if i + 1 < length :
    #     #                 if words[i + 1] not in english_vocab :
    #     #                     print(words[i]," ",words[i+1])
    #     #                     flag = False
    #     #             else :
    #     #                 print(words[i])
    #     #                 flag = True


    # text_1 = 


    from sklearn.externals import joblib
    joblib.dump(keyword_classifier,'keyword_classifier.pkl')

   
    #s = pickle.dumps(keyword_classifier,'keyword_classifier.pkl')
    start_time = time.time()
    text_1 = "LONDON: Euro zone inflation and US jobs data will offer clues to the health of major developed economies in the coming week while the malaise gripping emerging markets is expected to prompt India to cut interest rates.China may release monthly foreign exchange reserve data indicating how much more the central bank has spent on steadying the yuan following Aug. 11's surprise devaluation.Catalans vote on Sunday in a regional election which separatist parties are framing as a proxy referendum on independence from Spain while polls point to no clear winner in Portugal's Oct. 4 election.Wednesday's flash reading of September's annual euro zone inflation is expected at zero, although core inflation, which excludes volatile energy prices, is seen at 0.9 per cent for a third consecutive month.A negative headline inflation reading, which would be the first since March, would fuel speculation about further European Central Bank stimulus, six months after the euro zone's central bank launched a 1 trillion-euro-plus asset-purchase programme.On Wednesday, however, a surprisingly hawkish-sounding Mario Draghi said the ECB needed more time to assess whether China's slowdown, particularly its impact on commodity prices, cheap oil and a rising euro, would slow inflation further.Even if inflation turns negative again, deflation risks remain low, Unicredit analysts said in a note, with a fading of the base effect from 2014's plunge in energy prices likely to push the headline rate higher by year-end.Friday's non-farm payrolls data is expected to show the US economy added 203,000 jobs in September with the unemployment rate holding steady after falling in August to 5.1 per cent, its lowest since April 2008. Wage growth, a focus for Federal Reserve policymakers, also accelerated last month.Buoyant labour market data would revive expectations of a first US interest rate rise in nearly a decade, after a sharp selloff in global financial markets sparked by worries about China's economy prompted the Fed to hold fire this month.Fed Chair Janet Yellen said on Thursday she expects the US central bank to begin raising rates this year as long as inflation remains stable and the US economy is strong enough to boost employment. Her comments lifted the dollar on Friday.The Fed has two more chances to hike this year, at meetings in October and December. A Reuters poll this week found 72 of 93 economists expected a rise in December. Only nine foresaw a move next month and eight predicted the decision would be deferred to the first quarter of 2016.We continue to favour a December rate hike, wrote Commonwealth Bank of Australia economists in a note. But on the back of Yellen's comments today the risk is that the Fed pulls the trigger in October.A number of Fed policymakers are due to speak next week, which could help hone views on the likely timing of a hike.The Reserve Bank of India is seen cutting interest rates for the fourth time this year when it meets on Tuesday, as falling energy prices have cooled inflation and the economy has slowed.A Reuters poll forecast a 25 basis point reduction to 7.0 per cent, a four-year low.Purchasing Managers' surveys on Thursday will give further clues to the strength of China's economy, after a similar release this week showed factory activity at a 6-1/2 year low, while the central bank may also release FX reserves data.The $93.9 billion decline in China's reserves in August, reflecting central bank intervention around the yuan's devaluation, was the biggest monthly fall on record and marked an 11 per cent drop from a June 2014 peak.Capital outflows have escalated as fears grow that the world's second-largest economy is slowing as US interest rates look set to rise, although at $3.557 trillion in August, China's FX reserves remain the world's biggest.In an interview published this week, Chinese President Xi Jinping described the drop in reserves as moderate and manageable and said there was no need to overreact to it"    
    print(text_1)
    suggested_keywords = extract_keywords(text_1,keyword_classifier,top_k,preload)
    for keyword in suggested_keywords :
        print (keyword)
    print(time.time() - start_time)
                                                     
if __name__ == '__main__' :
    main()
                                                     
                                                     
                                                     
                                               
      
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                                     
                                        
    
        