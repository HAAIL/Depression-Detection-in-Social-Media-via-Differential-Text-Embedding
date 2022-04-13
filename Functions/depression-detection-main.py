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
from models import run_exps




#------------------------------------------------#
# -------------Data pre-processing---------------#
#------------------------------------------------#

# Dataset contains tweets from users who participated in mydepressionlookslike hashtag
DEPRESSIVE_TWEETS_CSV = 'MyDepressionLooksLike.csv'

# Dataset contains random tweets for sentiments analysis purpose ( 1=negative , 0=positive and neutral)
RANDOM_TWEETS_CSV = 'sentiment_tweets3.csv'

depressive_tweets_df  = pd.read_csv(DEPRESSIVE_TWEETS_CSV)
random_tweets_df      = pd.read_csv(RANDOM_TWEETS_CSV)
depressive_tweets     = np.array(depressive_tweets_df['Tweet Text'])
random_tweets         = np.array(random_tweets_df['message'])


# Clean depressive tweets and random tweets sets

arr_d = clean_text(depressive_tweets)
arr_r = clean_text(random_tweets.sample(2204))


ard = [x for x in arr_d if x]
arr = [x for x in arr_r if x]
print (str(len(ard)) +" tweets that show depression symtpoms")
print (str(len(arr)) +" random tweets with no depression symtpoms")
all_tweets = ard  +   arr
print (str(len(all_tweets)) +" total tweets are ready")
print ("------------------------------------------------------------------")



# Write tweets into text file
textfile = open("tweets_text.txt", "w", encoding='utf8')
for element in all_tweets:
    textfile.write(element + "\n")
textfile.close()




#------------------------------------------------#
#----------------------SBERT---------------------#
#------------------------------------------------#

# Get dysfunctional thoughts examples embeddings
data_ex         = Dataset('dysfunctional_thoughts_examples.txt')
sentence_sim    = SentenceSimilarity(data_ex)
emb_examples    = sentence_sim.embedded_sentences
emb_examples_df = pd.DataFrame.from_records(emb_examples)
emb_examples_df.to_csv('examples-emb.csv', index=False)

# Get tweets embeddings
twee            = Dataset('tweets_text.txt')
sentence_sim2   = SentenceSimilarity(twee)
emb_tweets      = sentence_sim2.embedded_sentences
emb_tweets_df   = pd.DataFrame.from_records(emb_tweets)
emb_tweets_df.to_csv('final-set-emb.csv', index=False)


#------------------------------------------------#
#-----------------Build dataset------------------#
#------------------------------------------------#


thoughts_dict = {}
xls           = ExcelFile('thoughts_categories.xlsx')
df            = xls.parse(xls.sheet_names[0])
df.columns    = ['Category', 'Example']
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


# An example of using the get_most_similar function
most_similar ,embds = sentence_sim.get_most_similar("I always fail")  # Most similar contains the top 7 sentences (as i Specefied)  and its cosine distances as key,value
                                                                      # embds contains the embeddings for the examples
# Getting the top similar sentences
most_similar_by_c = {}
for i in most_similar.keys():
    for dystype, e in thoughts_dict.items():
        if i in e:
            if dystype not in most_similar_by_c:
                most_similar_by_c[dystype] = [most_similar[i]]
            else:
                most_similar_by_c[dystype].append(most_similar[i])
print ('The nearset categories: ' )
print (most_similar_by_c)


# A function that utlizes get_most_similar to be used later
def get_thoughts_features(tweet):
    most_similar_types, emb,q = sentence_sim.get_most_similar(tweet)
    most_similar_by_c = {}
    for i in most_similar_types.keys():
        for c, e in thoughts_dict.items():
            if i in e:
                if c not in most_similar_by_c:
                    most_similar_by_c[c] = [most_similar_types[i]]
                else:
                    most_similar_by_c[c].append(most_similar_types[i])

    return most_similar_by_c


# go over all tweets and generate a list that contains the features vector for every tweet
featuresList = []

for tweet in all_tweets:
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
    with open('features_names.txt', 'r') as feat_order:
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
    with open('features_names.txt', 'r') as feat_order:
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
outfilename = 'final-set.csv'
# Output features to file
output(outfilename, featuresList)




#------------------------------------------------#
#-----------------Experiments--------------------#
#------------------------------------------------#



vectorizer           = TfidfVectorizer()
data                 = pd.read_csv('final-set.csv')
data= data.dropna()
tfidf_features      = vectorizer.fit_transform(data['tweet'].apply(lambda s: np.str_(s))).toarray()

# Perform PCA on tf-idf features
pca = PCA()
pca.fit(tfidf_features)
tfidf_features_pca = pca.transform(tfidf_features)


# Plot the cumulative sum to choose how many PC to use.
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(0,tfidf_features_pca.shape[1]),cumulative_variance)
plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('% of total variance accounted for')
plt.legend()
plt.show()



d                    = data.to_numpy()
x_d                  = d[:,1:11]                                                     # 1:7   -> the proposed features, 7:11   -> the other common features
tfidf_features_pcaed =tfidf_features_pca[:,0:3500]                                   # first 3500 PCs represnting tf-idf features

# 1- Combine the features in final set with tf-idf features
all_features     = np.concatenate((x_d,tfidf_features_pcaed), axis=1)

# 2- tf-idf features + other common features   (Excluding the proposed features)
expert_crafted   = np.concatenate((tfidf_features_pcaed, x_d[:,8:11]), axis=1)

# 3- tf-idf features + the proposed features   (Excluding other common features)
tfidf_new        = np.concatenate((x_d[:,0:7], tfidf_features_pcaed), axis=1)

# 4- bert embeddings
bert                 = pd.read_csv('final-set-emb.csv')
x       = bert.to_numpy()
y=d[:,11].astype('int')



run_exps(all_features,y, 'The report for using all')
run_exps(tfidf_new,y, 'The report for using new-tfidf')
run_exps(tfidf_features_pcaed,y, 'The report for using tfidf')
run_exps(expert_crafted,y, 'The report for using expert-crafted')
run_exps(x,y, 'The report for using bert')