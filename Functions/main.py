#-----------------------------------------------------
import pandas as pd
import numpy as np
import re
import nltk
import torch
from datas import Dataset
from sentence_similarity import SentenceSimilarity
from pandas import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from textblob import TextBlob,Word
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from ReplaceAbbr import ReplaceCommonAbbr
from CleanData import clean_text
from sklearn.feature_selection import chi2
from models import run_exps
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
#------------------------------------------------
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

        return top_similar_dict, self.embedded_sentences, query_embeddings, sorted_doc_ids[0]


# Dataset contains random tweets for sentiments analysis purpose - the tweets with lable '0' indicate posivity are going to be control users

# Dataset contains tweets from users who participated in mydepressionlookslike hashtag
DEPRESSIVE_TWEETS_CSV = 'MyDepressionLooksLike.csv'
#
# Dataset contains random tweets for sentiments analysis purpose - the tweets with lable '0' negative, "4" postitve and neutral
RANDOM_TWEETS_CSV = 'training.1600000.processed.noemoticon.csv'



depressive_tweets_df  = pd.read_csv(DEPRESSIVE_TWEETS_CSV)
random_tweets_df      = pd.read_csv(RANDOM_TWEETS_CSV, encoding='latin-1')
random_tweets_df.columns =['polarity', 'id', 'date', 'flag', 'user', 'text']

random_tweets_df_negative = random_tweets_df[random_tweets_df['polarity'] == 0]
random_tweets_df_negative = random_tweets_df_negative.sample(900)

random_tweets_df_random = random_tweets_df[random_tweets_df['polarity'] == 4]
random_tweets_df_random = random_tweets_df_random.sample(1500)

depressive_tweets     = np.array(depressive_tweets_df['Tweet Text'])
random_tweets_n         = np.array(random_tweets_df_negative['text'])
random_tweets_r         = np.array(random_tweets_df_random['text'])

arr_d = clean_text(depressive_tweets)
arr_n = clean_text(random_tweets_n)
arr_r = clean_text(random_tweets_r)
print (str(len(arr_d)) +" tweets that show depression symtpoms are ready")
print (str(len(arr_n)) +" tweets with no depression symtpoms (negative) are ready")
print (str(len(arr_r)) +" tweets with no depression symtpoms (postitve, neutral) are ready")

print ("------------------------------------------------------------------")

all_tweets = arr_d + arr_n + arr_r
print (str(len(all_tweets)) +" tweets in the dataset are ready")


all_tweetss = pd.DataFrame(all_tweets)
all_tweets = all_tweetss.dropna()
all_tweets.index = range(len(all_tweets))
all_tweets =  list(all_tweets[0])
print(all_tweets)

textfile = open("tweets_text2.txt", "w", encoding='utf8')
for element in all_tweets:
    textfile.write(element + "\n")
textfile.close()

data_ex = Dataset('dysfunctional_thoughts_examples-c.txt')

#using SBERT model to get depression symptoms examples embedding
sentence_sim1 = SentenceSimilarity2(data_ex)
emb_examples = sentence_sim1.embedded_sentences
# emb_examples_df = pd.DataFrame.from_records(emb_examples)
# emb_examples_df.to_csv('emb_examples-bert-c.csv', index=False)

#using SBERT model to get tweets embedding

twee = Dataset('tweets_text2.txt')
sentence_sim2 = SentenceSimilarity(twee)
emb_tweets = sentence_sim2.embedded_sentences
emb_tweets_df = pd.DataFrame.from_records(emb_tweets)
emb_tweets_df.to_csv('final-set2-bert.csv', index=False)


thoughts_dict = {}
xls           = ExcelFile('thoughts_categories_c.xlsx')
df            = xls.parse(xls.sheet_names[0])
df.columns    = [ 'Category', 'Example','id','subset id', 'category id' ]
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

# print(thoughts_dict)
# An example of using the get_most_similar function
# most_similar ,embds, q_embed, doc = sentence_sim1.get_most_similar("I always fail")  # Most similar contains the top 7 sentences (as i Specefied)  and its cosine distances as key,value
#                                                                       # embds contains the embeddings for the examples
# # Getting the top similar sentences
# most_similar_by_c = {}
# for i in most_similar.keys():
#     print(most_similar)
#     for dystype, e in thoughts_dict.items():
#
#         if i in e:
#             if dystype not in most_similar_by_c:
#                 most_similar_by_c[dystype] = [most_similar[i]]
#             else:
#                 most_similar_by_c[dystype].append(most_similar[i])
# print ('The nearset categories: ' )
# print (most_similar_by_c)


# # # # ---------------------------
# # # # Building the dataset
# # # # ---------------------------


# Cosine Distance

# A function that utlizes get_most_similar to be used later
def get_thoughts_features(tweet):
    most_similar_types, emb,q, doc = sentence_sim1.get_most_similar(tweet)
    most_similar_by_c = {}
    for i in most_similar_types.keys():
        for c, e in thoughts_dict.items():
            if i in e:
                if c not in most_similar_by_c:
                    most_similar_by_c[c] = [most_similar_types[i]]
                else:
                    most_similar_by_c[c].append(most_similar_types[i])

    return most_similar_by_c
def get_thoughts_features1(tweet):

    most_similar_types, emb,q, ids = sentence_sim1.get_most_similar(tweet)
    most_similar_by_c = {}
    for i in most_similar_types.keys():
        for c, e in thoughts_dict.items():
            if i in e:
                if c not in most_similar_by_c:
                    # print(i)
                    most_similar_by_c[c] = [i]
                    # print(most_similar_types[i])
                else:
                    most_similar_by_c[c].append(i)

    return most_similar_by_c
# # #


# go over all tweets and generate a list that contains the features vector for every tweet
featuresList = []

for tweet in all_tweets:
    featureDict = get_thoughts_features(tweet)  # get 7 top types of thinking and distances



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
    if tweet in arr_d:
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
        # iterate over each feature
        for i in range(len(features)):
            if i > 0 and isinstance(features[i], list):
                minn = min(features[i])
                outFile.write(str(minn))
            else:
                outFile.write(str(features[i]))
            if i < len(features) - 1:
                outFile.write(',')
        outFile.write('\n')
    print("Final dataset is created")
    outFile.close()


# output file for features
outfilename = 'final1.csv'
# Output features to file
output(outfilename, featuresList)


# Differential Embedding concatenation

# df_example = pd.read_csv('thoughts_categories_c.csv')
# df_tweets = pd.read_csv('final1.csv')
# df1 = pd.read_csv('emb_examples-bert-c.csv')
#
# all_tweets = df_tweets['tweet']
# all_tweets = all_tweets.dropna()
# all_tweets.index = range(len(all_tweets))
#
# textfile = open("tweets_text1.txt", "w", encoding='utf8')
# for element in all_tweets:
#     textfile.write(element + "\n")
# textfile.close()
#
# twee = Dataset('tweets_text1.txt')
# sentence_sim2 = SentenceSimilarity(twee)
# emb_tweets = sentence_sim2.embedded_sentences
# emb_tweets_df = pd.DataFrame.from_records(emb_tweets)
# emb_tweets_df.to_csv('final-set1-bert.csv', index=False)
#
# arr_d = df_tweets[df_tweets['Depressed'] == 1]
# arr_d = np.array(arr_d['tweet'])
# tweets_emb = pd.read_csv('final-set1-bert.csv')

featuresList = []

# for tweet in all_tweets:
#     featureDict = get_thoughts_features1(tweet)  # get 7 top types of thinking and distances
#     # print(featureDict)
#     tweet_features = []
#     tweet_features.append(tweet)
#     with open('features_names_c.txt', 'r') as feat_order:
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
#     if tweet in arr_d:
#         tweet_features.append(1)
#     else:
#         tweet_features.append(0)
#     # print(tweet_features)
#     # Append this features vector of the tweet onto the master featuresList
#     featuresList.append(tweet_features)
#     # print(featuresList)
#
# # # A function to output the calculated features to file
# def output(outputFilename, featuresList):
#     # open output file
#     outFile = open(outputFilename, 'w', encoding='utf8')
#
#     # Output header
#     # outFile.write('tweet\n')
#     # with open('features_names_c.txt', 'r') as feat_order:
#     #     for line in feat_order:
#     #         feature_name = line.strip()
#     #         outFile.write(feature_name)
#     #         outFile.write(',')
#     # outFile.write('Depressed\n')
#
#     # iterate over networks
#     for features in featuresList:
#         # iterate over each feature
#         for i in range(len(features)):
#             if i > 0 and isinstance(features[i], list):
#                 # get example embedding
#                 ex = features[i][0]
#                 # print(ex)
#                 ex_row = df_example[df_example['Example'] == ex]
#                 # print(ex_row)
#                 ex_id  = ex_row['ID']
#                 # print(ex_id)
#                 # print(ex_id.values[0])
#
#                 # print(len(ex_id.values[0]))
#                 emb_vect = df1.loc[df1['example id'] == ex_id.values[0]]
#                 emb_vect = emb_vect.values
#                 emb_vect =emb_vect[0][:-1]
#                 # print(len(emb_vect))
#                 # emb_vect = list(emb_vect)
#
#                 diff_embed= np.subtract(tweet_vect, emb_vect)
#                 diff_embed = list(diff_embed)
#                 diff_embed = ','.join(map(str, diff_embed))
#                 # diff_embed = ', '.join(diff_embed)
#                 # print(diff_embed)
#                 outFile.write(diff_embed)
#             elif i == 0:
#                 outFile.write(str(features[i]))
#                 # get tweets embedding
#                 tw_row = df_tweets[df_tweets['tweet'] == features[i]]
#                 print(tw_row)
#
#                 tw_id = tw_row['tweet id']
#                 print(tw_id)
#
#                 # print(tw_id.values[0])
#                 tweet_vect = tweets_emb.loc[tweets_emb['tweet id'] == tw_id.values[0]]
#                 tweet_vect = tweet_vect.values
#                 print(len(tweet_vect))
#                 tweet_vect = tweet_vect[0][:-1]
#                 # print(tweet_vect)
#             else:
#                 outFile.write(str(features[i]))
#             if i < len(features) - 1:
#                 outFile.write(',')
#
#         outFile.write('\n')
#     print("Final dataset is created")
#     outFile.close()
#
# # output file for features
# outfilename = 'final1-embed-co.csv'
# # # Output features to file
# output(outfilename, featuresList)

# embed_df = pd.read_csv('final1-embed-co.csv')
# embed_df['Depressed'] = df_tweets['Depressed']
# embed_df = embed_df.fillna(0)
# embed_df.to_csv('final1-embed-co-filled.csv')

# Average embeddings

# result = pd.read_csv('final1-embed-co-filled.csv', delimiter=',')
# result.dropna(inplace=True)
# print(result.shape)
# result.index = range(len(result))
# mean_emb = []
# cl = []
# for i,entity in result.iterrows():
#     dys = []
#     for d in range (0,7):
#         array = result.iloc[i,(d * 768) + 2: (d * 768 + 768) + 2]
#         dys.append(array.values)
#     m = np.mean(dys, axis = 0)
#     mean_emb.append(m)
#
# emb_mean_df = pd.DataFrame.from_records(mean_emb)
# emb_mean_df.to_csv('dys-emb-mean1.csv', index=False)