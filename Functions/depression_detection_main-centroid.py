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
# from models import run_exps, lr_cv, ROS_pipeline,original_pipeline, SMOTE_pipeline
from imblearn.pipeline import Pipeline, make_pipeline




#------------------------------------------------#
# -------------Data pre-processing---------------#
#------------------------------------------------#
# # # #
# # # Dataset contains tweets from users who participated in mydepressionlookslike hashtag
# DEPRESSIVE_TWEETS_CSV = 'MyDepressionLooksLike.csv'
# #
# # # Dataset contains random tweets for sentiments analysis purpose ( 1=negative , 0=positive and neutral)
# RANDOM_TWEETS_CSV = 'sentiment_tweets3.csv'
# #
# # # Dataset contains random tweets for sentiments analysis purpose ( 0=negative , 2=neutral and 4=neutral)
# NEGATIVE_TWEETS_CSV = 'negative-tweets.csv'
#
# depressive_tweets_df  = pd.read_csv(DEPRESSIVE_TWEETS_CSV)
#
# random_tweets_df      = pd.read_csv(RANDOM_TWEETS_CSV)
# random_tweets_df      = random_tweets_df[random_tweets_df['label'] == 0].sample(1000)
#
# negative_tweets_df      = pd.read_csv(NEGATIVE_TWEETS_CSV, encoding = "ISO-8859-1")
# negative_tweets_df      = negative_tweets_df[negative_tweets_df['target'] == 0].sample(3000)
#
#
#
# depressive_tweets     = np.array(depressive_tweets_df['Tweet Text'])
# random_tweets         = np.array(random_tweets_df['message'])
# negative_tweets         = np.array(negative_tweets_df['tweet'])
#
# #
# # Clean depressive tweets and random tweets sets
#
# arr_d = clean_text(depressive_tweets)
# arr_r = clean_text(random_tweets)
# arr_n = clean_text(negative_tweets)
#
# #
# ard = [x for x in arr_d if x]
# arr = [x for x in arr_r if x]
# arn = [x for x in arr_n if x]
#
# print (str(len(ard)) +" tweets that show depression symtpoms")
# print (str(len(arr)) +" random tweets with no depression symtpoms")
# print (str(len(arn)) +" negative tweets with no depression symtpoms")

# all_tweets = ard +  arn + arr
# print (str(len(all_tweets)) +" total tweets are ready")
# print ("------------------------------------------------------------------")
#
# all_tweet = pd.read_csv("experiment7-final-set.csv")
# all_tweets = all_tweet['tweet']
# ard = all_tweet[all_tweet['Depressed'] == 1]
# ard = np.array(ard['tweet'])
#
# # Write tweets into text file
# textfile = open("tweets_text_experiment5-polarity.txt", "w", encoding='utf8')
# for element in all_tweets:
#     textfile.write(element + "\n")
# textfile.close()
# #
#
#
#
# #------------------------------------------------#
# #----------------------SBERT---------------------#
# #------------------------------------------------#
#
# data_ex         = Dataset('dysfunctional_thoughts_examples.txt')
# sentence_sim    = SentenceSimilarity(data_ex)
# emb_examples    = sentence_sim.embedded_sentences
# emb_examples_df = pd.DataFrame.from_records(emb_examples)
# emb_examples_df.to_csv('examples-emb.csv', index=False)
#
# # #Get dysfunctional thoughts examples embeddings
# # data_ex2         = Dataset('dysfunctional_thoughts_examples2.txt')
# # sentence_sim2    = SentenceSimilarity(data_ex2)
# # emb_examples2    = sentence_sim2.embedded_sentences
# # emb_examples_df2 = pd.DataFrame.from_records(emb_examples2)
# # emb_examples_df2.to_csv('examples2-emb.csv', index=False)
#
# # Get tweets embeddings
# twee            = Dataset('tweets_text_experiment5-polarity.txt')
# sentence_sim2   = SentenceSimilarity(twee)
# emb_tweets      = sentence_sim2.embedded_sentences
# emb_tweets_df   = pd.DataFrame.from_records(emb_tweets)
# emb_tweets_df.to_csv('experiment5-polarity-tweets-emb.csv', index=False)
# #
#
# #------------------------------------------------#
# #-----------------Build dataset------------------#
# #------------------------------------------------#
#
#
# thoughts_dict = {}
# xls           = ExcelFile('thoughts_categories.xlsx')
# df            = xls.parse(xls.sheet_names[0])
# df.columns    = ['Category', 'Example']
# columns = list(df)
#
# for r in range (0,df.shape[0]):
#     for i in columns:
#         if i =='Category':
#             if df[i][r] not in thoughts_dict.keys():
#                 thoughts_dict[df[i][r]] = [df['Example'][r]]
#                 break
#             else:
#                 thoughts_dict[df[i][r]].append(df['Example'][r])
#                 break
#         else:
#             break

# thoughts_dict2 = {}
# xls           = ExcelFile('thoughts_categories2.xlsx')
# df2            = xls.parse(xls.sheet_names[0])
# df2.columns    = ['Category', 'Example']
# columns2 = list(df2)
#
# for r in range (0,df2.shape[0]):
#     for i in columns2:
#         if i =='Category':
#             if df2[i][r] not in thoughts_dict2.keys():
#                 thoughts_dict2[df2[i][r]] = [df2['Example'][r]]
#                 break
#             else:
#                 thoughts_dict2[df2[i][r]].append(df2['Example'][r])
#                 break
#         else:
#             break

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
#                 most_similar_by_c[dystype] = [most_similar[i]]
#             else:
#                 most_similar_by_c[dystype].append(most_similar[i])
# print ('The nearset categories: ' )
# print (most_similar_by_c)
#
# # # Getting the top similar sentences
# # most_similar_by_c2 = {}
# # for i in most_similar2.keys():
# #     for dystype, e in thoughts_dict2.items():
# #         if i in e:
# #             if dystype not in most_similar_by_c2:
# #                 most_similar_by_c2[dystype] = [most_similar2[i]]
# #             else:
# #                 most_similar_by_c2[dystype].append(most_similar2[i])
# # print ('The nearset categories 2: ' )
# # print (most_similar_by_c2)
#
#
#
# # A function that utlizes get_most_similar to be used later
# def get_thoughts_features(tweet):
#     most_similar_types, emb,q = sentence_sim.get_most_similar(tweet)
#     most_similar_by_c = {}
#     for i in most_similar_types.keys():
#         for c, e in thoughts_dict.items():
#             if i in e:
#                 if c not in most_similar_by_c:
#                     most_similar_by_c[c] = [most_similar_types[i]]
#                 else:
#                     most_similar_by_c[c].append(most_similar_types[i])
#
#     return most_similar_by_c
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
# featuresList = []
#
# for tweet in all_tweets:
#     words_count = 0
#     featureDict = get_thoughts_features(tweet)  # get 7 top types of thinking and distances
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
#     if tweet in ard:
#         tweet_features.append(1)
#     else:
#         tweet_features.append(0)
#
#     # Append this features vector of the tweet onto the master featuresList
#     featuresList.append(tweet_features)
#
# # A function to output the calculated features to file
# def output(outputFilename, featuresList):
#     # open output file
#     outFile = open(outputFilename, 'w', encoding='utf8')
#
#     # Output header
#     # outFile.write('tweet\n')
#     with open('features_names_c.txt', 'r') as feat_order:
#         for line in feat_order:
#             feature_name = line.strip()
#             outFile.write(feature_name)
#             outFile.write(',')
#     outFile.write('Depressed\n')
#
#     # iterate over networks
#     for features in featuresList:
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
# outfilename = 'experiment11-final-set.csv'
# # Output features to file
# output(outfilename, featuresList)

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
# data1                 = pd.read_csv('experiment7-final-set.csv')
# data1= data1.dropna()
# #
# # # data2                 = pd.read_csv('experiment9-final-set-LIWC-Analysis-paperC.csv')
# # # data2= data2.dropna()

# evaluate model performance with outliers removed using isolation forest
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
data = pd.read_csv('experiment8-bert-original-emb.csv')
examples = pd.read_csv('examples-emb.csv')
examples.iloc[:,:-2]
examples = examples.to_numpy()
iso = IsolationForest(contamination=0.2)
yhat = iso.fit_predict(examples)
print(yhat)
mask = yhat != -1
examples = examples[mask, :]
print(examples[:,-2:])
centroids = []
# d = 0
#
examples_no_outliers = examples
new_examples = pd.DataFrame(examples_no_outliers)
new_examples.to_csv('examples_no_outliers.csv')

# examples = pd.read_csv('examples_no_outliers.csv')
# examples = examples.iloc[:,1:-1]
# for i in range(1,8):
#
#     exs = []
#     sub = examples[examples['category id'] == i]
#     for j, entity in sub.iterrows():
#         array = examples.iloc[j,:-1]
#         # array = array[0]
#         exs.append(array.values)
#     # print(result.iloc[i, 1:5377])
#     m = np.mean(exs, axis = 0)
#     centroids.append(m)
#
# emb_mean_df = pd.DataFrame.from_records(centroids)
# emb_mean_df.to_csv('new_examples-centroids.csv', index=False)
# import itertools
#
# centroids = pd.read_csv('new_examples-centroids.csv')
# diff_centroids = []
# for t, entity in data.iterrows():
#     diff_embedding= []
#     for i in range(len(centroids)):
#         # get example embedding
#         emb_vect = centroids.iloc[i,:]
#         emb_vect   = emb_vect.values
#         tweet_vect = data.iloc[t,:].values
#         diff_embed= np.subtract(tweet_vect, emb_vect )
#         diff_embedding.append(diff_embed)
#
#     merged = list(itertools.chain(*diff_embedding))
#     diff_centroids.append(merged)
#
# data_centroids_diff = pd.DataFrame(diff_centroids)
# data_centroids_diff.to_csv('dataset8-new-centroids.csv', index=False)

# #
# data2                 = pd.read_csv('experiment5-final-set - LIWC Analysis.csv')
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


# mean_emb = []
# for i,entity in dataset.iterrows():
#     dys = []
#     for d in range (0,num_cat):
#         array = dataset.iloc[i,(d*768)+1:(d*768)+769]
#         dys.append(array.values)
#
#     m = np.mean(dys, axis = 0)
#     mean_emb.append(m)
#
# emb_mean_df = pd.DataFrame.from_records(mean_emb)
# emb_mean_df.to_csv(title +'-set8-emb-mean.csv', index=False)

# kfold = StratifiedKFold(n_splits=10, shuffle=False)
# model = LogisticRegression(penalty='l2', max_iter=10000000, solver='liblinear')
# name = title + 'exp8'
# emb_mean = emb_mean_df.to_numpy()
# mean_auc, std = draw_cv_roc_curve(model, kfold, emb_mean, ys, name, title='Cross Validated ROC')
# pr            = draw_cv_pr_curve(model, kfold, emb_mean, ys, title='Cross Validated PR')
#



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

import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC


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

from sklearn.linear_model import LogisticRegression
#
# d= pd.read_csv('set8-new-centroid-emb-mean.csv')
d=d.to_numpy()
print(d)
y = pd.read_csv('experiment8-final-set.csv')
y = y.iloc[:,-2].astype('int')
print(y)
kfold = StratifiedKFold(n_splits=10, shuffle=False)
model = LogisticRegression(penalty='l2', max_iter=10000000, solver='liblinear')

# draw_cv_roc_curve(model, kfold,d, y, title='Cross Validated ROC Of diff  new centroid mean - exp 8')
# draw_cv_pr_curve(model, kfold, d, y, title='Cross Validated PR Curve Of diff new centroid mean- exp 8')
#
# draw_cv_roc_curve(model, kfold, d1[:,12:], y, title='Cross Validated ROC Of LIWC')
# draw_cv_pr_curve(model, kfold, d1[:,12:], y, title='Cross Validated PR Curve Of LIWC')

# draw_cv_roc_curve(model, kfold, x_d1[:,0:7], y, title='Cross Validated ROC Of Dys. Thoughts')
# draw_cv_pr_curve(model, kfold, x_d1[:,0:7], y, title='Cross Validated PR Curve Of Dys. Thoughts')
# #
# # draw_cv_roc_curve(model, kfold, liwc_bert1, y, title='Cross Validated ROC Of LIWC + BERT')
# # draw_cv_pr_curve(model, kfold, liwc_bert1, y, title='Cross Validated PR Curve Of LIWC + BERT')
#
# draw_cv_roc_curve(model, kfold, new_bert1, y, title='Cross Validated ROC Of Dys. Thoughts + BERT pca - exp 11')
# draw_cv_pr_curve(model, kfold, new_bert1, y, title='Cross Validated PR Curve Of Dys. Thoughts + BERT pca - exp 11')

# draw_cv_roc_curve(model, kfold, liwc_new_bert1, y, title='Cross Validated ROC Of LIWC + Dys. Thoughts + BERT')
# draw_cv_pr_curve(model, kfold, liwc_new_bert1, y, title='Cross Validated PR Curve Of LIWC +Dys. Thoughts + BERT')
#
# draw_cv_roc_curve(model, kfold, x_d2[:,0:16], y, title='Cross Validated ROC Of Dys. Thoughts + Additional Depression Symptoms')
# draw_cv_pr_curve(model, kfold, x_d2[:,0:16], y, title='Cross Validated PR Curve Of Dys. Thoughts + Additional Depression Symptoms')
# #
#
# draw_cv_roc_curve(model, kfold, new_bert2, y, title='Cross Validated ROC Of Dys. Thoughts + Additional Depression Symptoms  + BERT pca - exp 11')
# draw_cv_pr_curve(model, kfold, new_bert2, y, title='Cross Validated PR Curve Of Dys. Thoughts +  Additional Depression Symptoms +BERT pca - exp 11')

# draw_cv_roc_curve(model, kfold, liwc_new_bert2, y, title='Cross Validated ROC Of LIWC + Dys. Thoughts + Additional Depression Symptoms + BERT')
# draw_cv_pr_curve(model, kfold, liwc_new_bert2, y, title='Cross Validated PR Curve Of LIWC +Dys. Thoughts + Additional Depression Symptoms + BERT')

#
# # run_exps(tfidf_features_pcaed,y, 'The report for using tfidf')
# # run_exps(all_features,y, 'The report for using all')
# # run_exps(tfliwc,y, 'The report for using tf liwc')
