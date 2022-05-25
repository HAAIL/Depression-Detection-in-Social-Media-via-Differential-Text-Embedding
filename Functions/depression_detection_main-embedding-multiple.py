import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datas import Dataset
from sentence_similarity import SentenceSimilarity
from pandas import *
from sklearn.decomposition import PCA
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from CleanData import clean_text
from models import run_exps, lr_cv, ROS_pipeline,original_pipeline, SMOTE_pipeline
from imblearn.pipeline import Pipeline, make_pipeline
import re
import time
import scipy
import logging
from typing import List, AnyStr
from datas import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer


import csv
import json
import time
import scipy
import pickle
import logging
from typing import List, AnyStr
from datas import Dataset
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer, InputExample, losses, SentencesDataset, evaluation
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SentenceSimilarity2():
    def __init__(self, dataset: Dataset, model: SentenceTransformer = None, n_docs: int = -1):
        self.dataset = dataset
        self.model = model if model else SentenceTransformer("bert-base-nli-stsb-mean-tokens")
        print("embedder loaded...")
        self.sentences = []
        self.doc_id_to_sentence_ids = {}

        self.sentence_pattern = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        for d in dataset.get_documents(n=n_docs):
            doc_id = d.get('id')
            text = d.get('text', None)

            sentence_ids = []
            if text:
                sentences = re.split(self.sentence_pattern, text)
                for s in sentences:
                    sentence_ids.append(len(self.sentences))
                    self.sentences.append(s)

            # Map from document to all its sentences (One-to-Many)
            self.doc_id_to_sentence_ids[doc_id] = sentence_ids
        # end for

        logger.debug(f"doc_to_sentence_ids: {self.doc_id_to_sentence_ids}")

        # Map from sentence to the document it came from (Many-to-One)
        self.sentence_id_to_doc_id = {}
        for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
            for s_id in sentence_ids:
                self.sentence_id_to_doc_id[s_id] = doc_id

        logger.debug(f"sentence_id_to_doc_id: {self.sentence_id_to_doc_id}")
        # Embedd extracted sentences using SentenceTransformer model.
        start = time.time()

        self.embedded_sentences = self.model.encode(self.sentences)
        logger.info(f"It took {round(time.time() - start, 3)} s to embedd {len(self.sentences)} sentences.")

        # temp  = []
        # for doc_id, sentence_ids in self.doc_id_to_sentence_ids.items():
        #     logger.debug((doc_id, sentence_ids))
        #     temp.append(f"document: {self.dataset.get_documents_by_id([doc_id])} - sentences: {[self.sentences[sid] for sid in sentence_ids]}")
        # logger.debug("\n\n".join(temp))

    # end def

    def get_most_similar(self, query: AnyStr, threshold: float = 1, limit: int = 63) -> List[int]:
        query_sentences = re.split(self.sentence_pattern, query)
        query_embeddings = self.model.encode(query_sentences)

        # logger.info(f"Extracted {len(query_sentences)} sentences from query")
        logger.debug(f"Sentences: {' -- '.join(query_sentences)}")

        # Calculate cosine distance between requested sentences and all sentences
        cosine_dist = scipy.spatial.distance.cdist(query_embeddings, self.embedded_sentences, "cosine")
        # Extract column values where distance is below threshold
        below_threshold = cosine_dist < threshold
        doc_ids, matched_column_ids = np.where(below_threshold)

        # Extract x (input sentence id), y (dataset sentence id) and distance between these.
        x_y_dist = []
        for x, y in zip(doc_ids, matched_column_ids):
            x_y_dist.append([x, y, cosine_dist[x][y]])

        # Sort list based on distance and remove duplicates, keeping the one with lowest distance.
        sorted_x_y_dist = sorted(x_y_dist, key=lambda x: x[2])
        sorted_sentence_ids = [doc[1] for doc in sorted_x_y_dist]
        sorted_doc_ids = [self.sentence_id_to_doc_id[sent_id] for sent_id in sorted_sentence_ids]
        top_similar_dict = {}
        top_similar = []
        top_dist = []
        # logger.info(f"Distance for top documents: {[round(x[2],3) for x in sorted_x_y_dist[:limit]]}")

        top_dist              = [round(x[2], 3) for x in sorted_x_y_dist[:limit]]
        top_similar           = self.dataset.get_documents_by_id(list(dict.fromkeys(sorted_doc_ids).keys())[:limit])
        # top_similar_sentences = self.dataset.get_documents_list(list(dict.fromkeys(sorted_doc_ids).keys())[:limit])
        # print(sorted_sentence_ids)
        if (len(top_dist) != len(top_similar)):
            print(len(top_dist))
            print(len(top_similar))
        anchors_ids = []
        for i in range(len(top_dist)):
            top_similar_dict[top_similar[i]] = top_dist[i]

        return top_similar_dict, self.embedded_sentences, query_embeddings, sorted_doc_ids

# using SBERT model and get BERT embeddings of the tweets

all_tweet = pd.read_csv("experiment5-final-set.csv")
all_tweets = all_tweet['tweet']
ard = all_tweet[all_tweet['Depressed'] == 1]
ard = np.array(ard['tweet'])


# twee = Dataset('tweets_text_experiment5.txt')
# sentence_sim = SentenceSimilarity(twee)
# emb_tweets = sentence_sim.embedded_sentences
# emb_tweets_df = pd.DataFrame.from_records(emb_tweets)
# emb_tweets_df.to_csv('final-set5-emb.csv', index=False)
#
# data = pd.read_csv("experiment7-final-set.csv")
examples = Dataset("dysfunctional_thoughts_examples.txt")
sentence_sim2 = SentenceSimilarity2(examples)
# most_similar ,embds,qembed,c = sentence_sim2.get_most_similar("I always fail")  # Most similar contains the top 7 sentences (as i Specefied)  and its cosine distances as key,valu
# # print(type(most_similar))
# print(type(most_similar[0]))

# print(len(embds))
# df = pd.DataFrame(embds)
# # Declare a list that is to be converted into a column
# id = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,
# 60,61,62,63]
# #
# df['cluidster'] = id
# df.to_csv(r'examples-embeddings.csv')
df1 = pd.read_csv('examples-embeddings.csv')
#
# df1.rename( columns={'Unnamed: 0':'example id'}, inplace=True )
# closest_anchor = []
# for i, entity in data.iterrows():
#     t = entity['tweet']
#     a, b, c, d = sentence_sim2.get_most_similar(t)
#     closest_anchor.append(d)

# print(closest_anchor)

# closest_anchor   = pd.DataFrame.from_records(closest_anchor)
# closest_anchor.to_csv('closest_anchor1.csv', index=False)

# close_emb = []
# for i in closest_anchor:
#     close_emb.append(df[df['example id']==i])
#
# close_emb   = pd.DataFrame.from_records(close_emb)
# close_emb.to_csv('close_emb7.csv', index=False)

# tweets_embedding = pd.read_csv("experiment1-emb.csv")
# diff_embed = []
#
# close_emb = pd.read_csv("close_emb1.csv")
# for i, entity in tweets_embedding.iterrows():
#     tweet_vector  = pd.to_numeric(tweets_embedding.iloc[i])
#     anchor_vector = pd.to_numeric(list(close_emb.iloc[i, 1:-1]))
#     diff_embed.append(np.subtract(list(tweet_vector), anchor_vector))

# print(len(diff_embed))
# diff_embed   = pd.DataFrame.from_records(diff_embed)
# diff_embed.to_csv('diff_embed1.csv', index=False)

# #
# #------------------------------------------------#
# #-----------------Build dataset------------------#
# #------------------------------------------------#
#
#
thoughts_dict = {}
xls           = ExcelFile('thoughts_categories.xlsx')
df            = xls.parse(xls.sheet_names[0])
df.columns    = ['Category', 'Example', 'ID']
columns = list(df)

for r in range (0,df.shape[0]):
    for i in columns:
        if i =='Category':
            if df[i][r] not in thoughts_dict.keys():
                thoughts_dict[df[i][r]] = [df['Example'][r]]
                break
            else:
                thoughts_dict[df[i][r]].append(df['Example'][r])
                break
        else:
            break
# #
# # thoughts_dict2 = {}
# # xls           = ExcelFile('thoughts_categories2.xlsx')
# # df2            = xls.parse(xls.sheet_names[0])
# # df2.columns    = ['Category', 'Example']
# # columns2 = list(df2)
# #
# # for r in range (0,df2.shape[0]):
# #     for i in columns2:
# #         if i =='Category':
# #             if df2[i][r] not in thoughts_dict2.keys():
# #                 thoughts_dict2[df2[i][r]] = [df2['Example'][r]]
# #                 break
# #             else:
# #                 thoughts_dict2[df2[i][r]].append(df2['Example'][r])
# #                 break
# #         else:
# #             break
#
# # An example of using the get_most_similar function
# most_similar ,embds, q_embed = sentence_sim.get_most_similar("I always fail")  # Most similar contains the top 7 sentences (as i Specefied)  and its cosine distances as key,value
#                                                                       # embds contains the embeddings for the examples
# # most_similar2 ,embds2, q_embed2 = sentence_sim2.get_most_similar("I always fail")  # Most similar contains the top 7 sentences (as i Specefied)  and its cosine distances as key,value
#
# # Getting the top similar sentences
# most_similar_by_c = {}
# for i in most_similar.keys():
#     for dystype, e in thoughts_dict.items():
#         if i in e:
#             if dystype not in most_similar_by_c:
#                 most_similar_by_c[dystype] = [i]
#             else:
#                 most_similar_by_c[dystype].append(i)
# print ('The nearset categories: ' )
# print (most_similar_by_c)
#
# # # Getting the top similar sentences
# most_similar_by_c2 = {}
# for i in most_similar.keys():
#     for dystype, e in thoughts_dict.items():
#         if i in e:
#             if dystype not in most_similar_by_c2:
#                 most_similar_by_c2[dystype] = [most_similar[i]]
#             else:
#                 most_similar_by_c2[dystype].append(most_similar[i])
# print ('The nearset categories 2: ' )
# print (most_similar_by_c2)
# #
#
#
# # A function that utlizes get_most_similar to be used later
def get_thoughts_features(tweet):
    most_similar_types, emb,q, ids = sentence_sim2.get_most_similar(tweet)
    most_similar_by_c = {}
    for i in most_similar_types.keys():
        for c, e in thoughts_dict.items():
            if i in e:
                if c not in most_similar_by_c:
                    most_similar_by_c[c] = [i]
                else:
                    most_similar_by_c[c].append(i)

    return most_similar_by_c
#
# # A function that utlizes get_most_similar to be used later
# # def get_thoughts_features2(tweet):
# #     most_similar_types, emb,q = sentence_sim2.get_most_similar(tweet)
# #     most_similar_by_c = {}
# #     for i in most_similar_types.keys():
# #         for c, e in thoughts_dict2.items():
# #             if i in e:
# #                 if c not in most_similar_by_c:
# #                     most_similar_by_c[c] = [most_similar_types[i]]
# #                 else:
# #                     most_similar_by_c[c].append(most_similar_types[i])
# #
# #     return most_similar_by_c
#
# # go over all tweets and generate a list that contains the features vector for every tweet
df_example = pd.read_csv('thoughts_categories.csv')
df_tweets = pd.read_csv('experiment5-final-set.csv')
tweets_emb = pd.read_csv('final-set5-emb.csv')


featuresList = []
#
for tweet in all_tweets[:2]:
    words_count = 0
    featureDict = get_thoughts_features(tweet)  # get 7 top types of thinking and distances

    words_count = len(str(tweet).split(" "))  # calculate words count feature

    text = TextBlob(tweet)
    pol = text.sentiment.polarity  # calculate polarity score
    sub = text.sentiment.subjectivity  # calculate subjectivity score

    featureDict['words count'] = words_count
    featureDict['polarity'] = pol
    featureDict['subjectivity'] = sub

    tweet_features = []
    tweet_features.append(tweet)
    with open('features_names_c.txt', 'r') as feat_order:
        for line in feat_order:
            feature_name = line.strip()
            if feature_name == 'tweet':
                continue
            elif feature_name in featureDict:
                tweet_features.append(featureDict[feature_name])
            else:
                tweet_features.append(1)

    # add the calss (1) for showing depression sympotoms (0) for no symptoms
    if tweet in ard:
        tweet_features.append(1)
    else:
        tweet_features.append(0)

    # Append this features vector of the tweet onto the master featuresList
    featuresList.append(tweet_features)

# A function to output the calculated features to file
def output(outputFilename, featuresList):
    # open output file
    outFile = open(outputFilename, 'w', encoding='utf8')

    # Output header
    # outFile.write('tweet\n')
    with open('features_names_c.txt', 'r') as feat_order:
        for line in feat_order:
            feature_name = line.strip()
            outFile.write(feature_name)
            outFile.write(',')
    outFile.write('Depressed\n')

    # iterate over networks
    for features in featuresList:
        print(features)
        # iterate over each feature
        for i in range(len(features)):
            print(i)
            if i > 0 and isinstance(features[i], list):
                # get example embedding
                ex = features[i][0]
                ex_row = df_example[df_example['Example'] == ex]
                ex_id  = ex_row['ID']

                emb_vect = df1.loc[df1['example id'] == ex_id.values[0]]
                emb_vect = emb_vect.values
                emb_vect =emb_vect[0][1:]

                diff_embed= np.subtract(tweet_vect, emb_vect)
                outFile.write(str(diff_embed))
            elif i == 0:
                # get tweets embedding
                print(features[i])
                tw_row = df_tweets[df_tweets['tweet'] == features[i]]
                tw_id = tw_row['tweet id']
                # print(tw_id.values[0])
                tweet_vect = tweets_emb.loc[tweets_emb['tweet id'] == tw_id.values[0]]
                tweet_vect = tweet_vect.values
                tweet_vect = tweet_vect[0][1:]
            else:
                outFile.write(str(features[i]))
            if i < len(features) - 1:
                outFile.write(',')
        outFile.write('\n')
    print("Final dataset is created")
    outFile.close()

# output file for features
outfilename = 'experiment5-final-set-emb.csv'
# Output features to file
output(outfilename, featuresList)

# featuresList2 = []
#
# for tweet in all_tweets:
#     words_count = 0
#     featureDict = get_thoughts_features2(tweet)  # get 7 top types of thinking and distances
#
#     words_count = len(str(tweet).split(" "))  # calculate words count feature
#
#     text = TextBlob(tweet)
#     pol = text.sentiment.polarity  # calculate polarity score
#     sub = text.sentiment.subjectivity  # calculate subjectivity score
#
#     featureDict['words count'] = words_count
#     featureDict['polarity'] = pol
#     featureDict['subjectivity'] = sub
#
#     tweet_features = []
#     tweet_features.append(tweet)
#     with open('features_names.txt', 'r') as feat_order:
#         for line in feat_order:
#             feature_name = line.strip()
#             if feature_name == 'tweet':
#                 continue
#             elif feature_name in featureDict:
#                 tweet_features.append(featureDict[feature_name])
#             else:
#                 tweet_features.append(1)
#
#     # add the calss (1) for showing depression sympotoms (0) for no symptoms
#     if tweet in ard:
#         tweet_features.append(1)
#     else:
#         tweet_features.append(0)
#
#     # Append this features vector of the tweet onto the master featuresList
#     featuresList2.append(tweet_features)
#
#
# # A function to output the calculated features to file
# def output2(outputFilename, featuresList):
#     # open output file
#     outFile = open(outputFilename, 'w', encoding='utf8')
#
#     # Output header
#     # outFile.write('tweet\n')
#     with open('features_names.txt', 'r') as feat_order:
#         for line in feat_order:
#             feature_name = line.strip()
#             outFile.write(feature_name)
#             outFile.write(',')
#     outFile.write('Depressed\n')
#
#     # iterate over networks
#     for features in featuresList2:
#         # iterate over each feature
#         for i in range(len(features)):
#             if i > 0 and isinstance(features[i], list):
#                 minn = min(features[i])
#                 outFile.write(str(minn))
#             else:
#                 outFile.write(str(features[i]))
#             if i < len(features) - 1:
#                 outFile.write(',')
#         outFile.write('\n')
#     print("Final dataset is created")
#     outFile.close()
#
# # output file for features
# outfilename = 'experiment9-final-set-1anchor.csv'
# # Output features to file
# output(outfilename, featuresList2)

#------------------------------------------------#
#-----------------Experiments--------------------#
#------------------------------------------------#
#
#
#
# vectorizer           = TfidfVectorizer()
data1                 = pd.read_csv('experiment1-final-set.csv')
# data1= data1.dropna()
# #
# # # data2                 = pd.read_csv('experiment9-final-set-LIWC-Analysis-paperC.csv')
# # # data2= data2.dropna()
# #
# data2                 = pd.read_csv('experiment11-final-set.csv')
# data2= data2.dropna()
# #
# #
# # tfidf_features      = vectorizer.fit_transform(data1['tweet'].apply(lambda s: np.str_(s))).toarray()
# # # # #
# # # Perform PCA on tf-idf features
# # # pca = PCA()
# # # pca.fit(tfidf_features)
# # # tfidf_features_pca = pca.transform(tfidf_features)
# # # tfidf_df   = pd.DataFrame.from_records(tfidf_features_pca)
# # # tfidf_df.to_csv('tfidf7.csv', index=False)
# #
# # #
# # # print(data.Depressed.value_counts())
# # # # Plot the cumulative sum to choose how many PC to use.
# # # # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# # # # plt.plot(range(0,tfidf_features_pca.shape[1]),cumulative_variance)
# # # # plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# # # # plt.xlabel('Number of Principal Components')
# # # # plt.ylabel('% of total variance accounted for')
# # # # plt.legend()
# # # # plt.show()
# # # # #
# # # #
# # #
# # #
# # #
# # #
# # #
# bert                 = pd.read_csv('experiment7-emb.csv')
# # # # #
# d1                    = data1.to_numpy()
# x_d1                  = d1[:,1:11]                                                     # 1:7   -> the proposed features, 7:11   -> the other common features
# #
# d2                    = data2.to_numpy()
# x_d2                  = d2[:,1:19]
# #
# # # d3                    = data3.to_numpy()
# # # x_d3                  = d3[:,1:11]
# #
# #
# # # tfidf_features_pcaed =tfidf_features_pca[:,0:6500]                                   # first 3500 PCs represnting tf-idf features
# # # tfidf_features_pcaed = pd.read_csv("tfidf7.csv")
# # # tfidf_features_pcaed = np.array(tfidf_features_pcaed)
# # # tfidf_features_pcaed =tfidf_features_pcaed[:,0:3500]
# # # # #
# pca = PCA()
# pca.fit(bert)
# bert_pca = pca.transform(bert)
# #
# # # # # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# # # # # plt.plot(range(0,bert_pca.shape[1]),cumulative_variance)
# # # # # plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# # # # # plt.xlabel('Number of Principal Components')
# # # # # plt.ylabel('% of total variance accounted for')
# # # # # plt.legend()
# # # # # plt.show()
# # # # #
# # # # # # 1- Combine the features in final set with tf-idf features
# # # all_features     = np.concatenate((x_d1,tfidf_features_pcaed), axis=1)
# #
# #
# # print("--------------------------------------------------------------------------")
# #
# # # all_features2     = np.concatenate((x_d2,tfidf_features_pcaed), axis=1)
# #
# # # tfliwc     = np.concatenate((d1[:,12:] ,tfidf_features_pcaed), axis=1)
# # # all     = np.concatenate((d1[:,12:] ,all_features), axis=1)
# # # all2     = np.concatenate((d1[:,12:] ,all_features2), axis=1)
# #
# # #
# # # # #
# # # # # # 2- tf-idf features + other common features   (Excluding the proposed features)
# # # # # # expert_crafted   = np.concatenate((tfidf_features_pcaed, x_d[:,8:11]), axis=1)
# # # # #
# # # # # # 3- tf-idf features + the proposed features   (Excluding other common features)
# # # # #
# # # # # # 4- bert embeddings
# x       = tweets_embedding.to_numpy()
# y=d1[:,11].astype('int')
# #
# # # run_exps(all_features,y, 'The report for using tfidf dys')
# # # run_exps(tfidf_features_pcaed,y, 'The report for using tfidf ')
# #
# # # # #
# liwc_new1        = np.concatenate((x_d1[:,0:7], d1[:,12:] ), axis=1)
# liwc_new_bert1        = np.concatenate((liwc_new1, bert_pca[:,:300]), axis=1)
# liwc_bert1        = np.concatenate((d1[:,12:], x), axis=1)
# new_bert1        = np.concatenate((x_d1[:,0:7], x), axis=1)
# #
# bert_diff        = np.concatenate((x, diff_embed ), axis=1)
# liwc_new_bert2        = np.concatenate((liwc_new2, x ), axis=1)
# liwc_bert2        = np.concatenate((d2[:,21:], x), axis=1)
# new_bert2        = np.concatenate((x_d2[:,0:18],x), axis=1)
# #
# # # liwc_new2        = np.concatenate((x_d2[:,0:7], d2[:,12:] ), axis=1)
# #
# # # run_exps(d1[:,12:],y, 'The report for using tfidf')
# # # run_exps(tfliwc,y, 'The report for using tf liwc')
# #
# # run_exps(x , y, 'The report for using bert')
# #
# # #
# # # #
# # run_exps(d1[:,12:],y, 'The report for using liwc ')
# # run_exps(x_d1[:,0:7],y, 'The report for using new')
# # run_exps(new_bert1,y, 'The report for using new bert')
# # run_exps(liwc_new_bert1,y, 'The report for using liwc new bert ')
# # run_exps(liwc_bert1,y, 'The report for using liwc bert ')
# # # run_exps(all_features,y, 'The report for using tfidf dys')
# # # run_exps(all,y, 'The report for using all')
#
# print("-----------------------------------------")
# # run_exps(d2[:,12:],y, 'The report for using liwc paper')
# # run_exps(x_d2[:,0:16],y, 'The report for using new ')
# # run_exps(new_bert2,y, 'The report for using new bert')
# # run_exps(liwc_new_bert2,y, 'The report for using liwc new bert ')
# # run_exps(liwc_bert2,y, 'The report for using liwc bert ')
# # run_exps(all_features2,y, 'The report for using tfidf dys')
# # run_exps(all2,y, 'The report for using all')
#
# print("--------------------custom--------------------------")
#
# # run_exps(d2[:,12:],y, 'The report for using liwc and custom')
# # run_exps(x_d2[:,0:7],y, 'The report for using new ')
# # run_exps(liwc_new_bert2,y, 'The report for using liwc new bert')
# # run_exps(liwc_new2,y, 'The report for using liwc new ')
# # run_exps(liwc_bert2,y, 'The report for using liwc bert ')
#
# # # print("original ")
# # # print("bert")
# # # lr_cv(5, bert_pca[:,:300], y,original_pipeline, 'micro')
# # # print("bert liwc")
# # # lr_cv(5, liwc_bert, y,original_pipeline, 'micro')
# # # print("bert liwc dys")
# # # lr_cv(5, liwc_new_bert, y,original_pipeline, 'micro')
# # # print("over sampler")
# # # print("bert")
# # # lr_cv(5, bert_pca[:,:300], y,ROS_pipeline, 'micro')
# # # print("bert liwc")
# # # lr_cv(5, liwc_bert, y,ROS_pipeline, 'micro')
# # # print("liwc bert dys")
# # # lr_cv(5, liwc_new_bert, y,ROS_pipeline, 'micro')
# # # print("smote")
# # print("bert")
# #  run_exps(d1[:,12:],y, 'The report for using liwc ')
# # run_exps(x_d1[:,0:7],y, 'The report for using new')
# # run_exps(new_bert1,y, 'The report for using new bert')
# # run_exps(liwc_new_bert1,y, 'The report for using liwc new bert ')
# # run_exps(liwc_bert1,y, 'The report for using liwc bert ')
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# from numpy import interp
# import pandas as pd
# from sklearn.datasets import make_blobs
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
# from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
# from sklearn.svm import SVC


def draw_cv_roc_curve(classifier, cv, X, y, title='ROC Curve'):
    """
    Draw a Cross Validated ROC Curve.
    Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series
    Example largely taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
    """
    # Creating ROC Curve with Cross Validation
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()


def draw_cv_pr_curve(classifier, cv, X, y, title='PR Curve'):
    """
    Draw a Cross Validated PR Curve.
    Keyword Args:
        classifier: Classifier Object
        cv: StratifiedKFold Object: (https://stats.stackexchange.com/questions/49540/understanding-stratified-cross-validation)
        X: Feature Pandas DataFrame
        y: Response Pandas Series

    Largely taken from: https://stackoverflow.com/questions/29656550/how-to-plot-pr-curve-over-10-folds-of-cross-validation-in-scikit-learn
    """
    y_real = []
    y_proba = []

    i = 0
    for train, test in cv.split(X, y):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        precision, recall, _ = precision_recall_curve(y[test], probas_[:, 1])

        # Plotting each individual PR Curve
        plt.plot(recall, precision, lw=1, alpha=0.3,
                 label='PR fold %d (AUC = %0.3f)' % (i, average_precision_score(y[test], probas_[:, 1])))

        y_real.append(y[test])
        y_proba.append(probas_[:, 1])

        i += 1

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)

    precision, recall, _ = precision_recall_curve(y_real, y_proba)

    plt.plot(recall, precision, color='b',
             label=r'Precision-Recall (AUC = %0.3f)' % (average_precision_score(y_real, y_proba)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()

# # print('bert')
# # lr_cv(10, x, y, 'bert')
# # print("liwc")
# # lr_cv(10, d1[:,12:], y,'liwc')
# # print("new")
# # lr_cv(10, x_d1[:,0:7], y, 'new')
# # print("bert liwc")
# # lr_cv(10, liwc_bert1, y, 'bert liwc')
# # print("new bert")
# # lr_cv(10, new_bert1, y, 'bert new')
# # print("liwc bert dys")
# # lr_cv(10, liwc_new_bert1, y, 'liwc bert new')
# #
# # print("affitional")
# # # print("liwc")
# # # lr_cv(10, d2[:,21:], y)
# # print("new")
# # lr_cv(10, x_d2[:,0:16], y, 'new+add')
# # # print("bert liwc")
# # # lr_cv(10, liwc_bert2, y)
# # print("new bert")
# # lr_cv(10, new_bert2, y, 'new add bert')
# # print("liwc bert dys")
# # lr_cv(10, liwc_new_bert2, y, 'liwc bert new add')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

#
# kfold = StratifiedKFold(n_splits=10, shuffle=False)
# model = LogisticRegression(penalty='l2', max_iter=10000000, solver='liblinear')
#
# draw_cv_roc_curve(model, kfold,x, y, title='Cross Validated ROC Of BERT exp 5')
# draw_cv_pr_curve(model, kfold, x, y, title='Cross Validated PR Curve Of BERT exp 5')
# #
# draw_cv_roc_curve(model, kfold, diff_embed, y, title='Cross Validated ROC Of DIFF')
# draw_cv_pr_curve(model, kfold, diff_embed, y, title='Cross Validated PR Curve Of DIFF')
#
# draw_cv_roc_curve(model, kfold, bert_diff, y, title='Cross Validated ROC Of BERT & DIFF exp 5')
# draw_cv_pr_curve(model, kfold, bert_diff, y, title='Cross Validated PR Curve Of BERT & DIFF exp 5')
# #
# # draw_cv_roc_curve(model, kfold, liwc_bert1, y, title='Cross Validated ROC Of LIWC + BERT')
# # draw_cv_pr_curve(model, kfold, liwc_bert1, y, title='Cross Validated PR Curve Of LIWC + BERT')
#
# draw_cv_roc_curve(model, kfold, new_bert1, y, title='Cross Validated ROC Of Dys. Thoughts + BERT pca - exp 11')
# draw_cv_pr_curve(model, kfold, new_bert1, y, title='Cross Validated PR Curve Of Dys. Thoughts + BERT pca - exp 11')
#
# # draw_cv_roc_curve(model, kfold, liwc_new_bert1, y, title='Cross Validated ROC Of LIWC + Dys. Thoughts + BERT')
# # draw_cv_pr_curve(model, kfold, liwc_new_bert1, y, title='Cross Validated PR Curve Of LIWC +Dys. Thoughts + BERT')
# #
# draw_cv_roc_curve(model, kfold, x_d2[:,0:16], y, title='Cross Validated ROC Of Dys. Thoughts + Additional Depression Symptoms')
# draw_cv_pr_curve(model, kfold, x_d2[:,0:16], y, title='Cross Validated PR Curve Of Dys. Thoughts + Additional Depression Symptoms')
# #
#
# draw_cv_roc_curve(model, kfold, new_bert2, y, title='Cross Validated ROC Of Dys. Thoughts + Additional Depression Symptoms  + BERT pca - exp 11')
# draw_cv_pr_curve(model, kfold, new_bert2, y, title='Cross Validated PR Curve Of Dys. Thoughts +  Additional Depression Symptoms +BERT pca - exp 11')
#
# # draw_cv_roc_curve(model, kfold, liwc_new_bert2, y, title='Cross Validated ROC Of LIWC + Dys. Thoughts + Additional Depression Symptoms + BERT')
# # draw_cv_pr_curve(model, kfold, liwc_new_bert2, y, title='Cross Validated PR Curve Of LIWC +Dys. Thoughts + Additional Depression Symptoms + BERT')
#
# #
# # # run_exps(tfidf_features_pcaed,y, 'The report for using tfidf')
# # # run_exps(all_features,y, 'The report for using all')
# # # run_exps(tfliwc,y, 'The report for using tf liwc')
# XTrain, XTest, yTrain, yTest = train_test_split(x,y, random_state=123, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(XTrain, yTrain)
# KNeighborsClassifier(...)
# p = neigh.predict(XTest)
# cm = confusion_matrix(yTest, p)
# cm
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in p])
# def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
#
#     matrix = confusion_matrix(actual_classes, predicted_classes)
#     print(matrix)
#     plt.figure(figsize=(12.8, 6))
#     sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
#     plt.xlabel('Predicted');
#     plt.ylabel('Actual');
#     plt.title('Confusion Matrix')
#     plt.savefig("unsupervised-bert")
#     plt.show()
# plot_confusion_matrix(yTest, p,["1", "0"])
# print(roc_auc_score(yTest,y_pred_))
# print(average_precision_score(yTest,y_pred_))
# print(precision_recall_curve(yTest,y_pred_))

# XTrain, XTest, yTrain, yTest = train_test_split(bert_diff,y, random_state=123, test_size=0.2)
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(XTrain, yTrain)
# # KNeighborsClassifier(...)
# p = neigh.predict(XTest)
# cm = confusion_matrix(yTest, p)
# cm
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in p])
# def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
#
#     matrix = confusion_matrix(actual_classes, predicted_classes)
#     print(matrix)
#     plt.figure(figsize=(12.8, 6))
#     sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
#     plt.xlabel('Predicted');
#     plt.ylabel('Actual');
#     plt.title('Confusion Matrix')
#     plt.savefig("unsupervised-bert-new1")
#     plt.show()
# plot_confusion_matrix(yTest, p,["1", "0"])
# print(roc_auc_score(yTest,y_pred_))
# print(average_precision_score(yTest,y_pred_))
# print(precision_recall_curve(yTest,y_pred_))