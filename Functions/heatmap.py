import pandas as pd
import numpy as np
from datas import Dataset
import re
import scipy
import logging
from typing import  List, AnyStr
from sentence_transformers import SentenceTransformer
import time
from pandas import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
import csv

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

    def get_most_similar(self, query: AnyStr, threshold: float = 100, limit: int = 1000) -> List[int]:
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


###################################################################################################
##                                           Metrics                                             ##
###################################################################################################

def draw_cv_roc_curve(classifier, cv, X, y, name,  title='ROC Curve'):
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
    plt.savefig(title + name + '.png', bbox_inches='tight')
    # plt.show()
    return mean_auc, std_auc

def draw_cv_pr_curve(classifier, cv, X, y, name, title='PR Curve'):
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
    plt.savefig(title + name +'.png', bbox_inches='tight')
    # plt.show()
    return average_precision_score(y_real, y_proba)


#############################################################################################
##                           Data Processing and Models settings                           ##
#############################################################################################

def get_thoughts_features(tweet, sentence_sim, thoughts_dict):
    most_similar_types, emb,q, ids = sentence_sim.get_most_similar(tweet)
    most_similar_by_c = {}
    for i in most_similar_types.keys():
        for c, e in thoughts_dict.items():
            if i in e:
                if c not in most_similar_by_c:
                    most_similar_by_c[c] = [i]
                else:
                    most_similar_by_c[c].append(i)
    return most_similar_by_c

df_tweets = pd.read_csv('experiment8-final-set.csv')

tweets_emb = pd.read_csv('experiment8-emb.csv')
dt = pd.read_csv('thoughts_categories.csv')

def output(outputFilename, featuresList, title, df_example, df_emb_example):
    # open output file
    outFile = open(outputFilename, 'w', encoding='utf8')

    # # Output header
    # # outFile.write('tweet\n')
    # with open('%s-features_names.txt' % title, 'r') as feat_order:
    #     for line in feat_order:
    #         feature_name = line.strip()
    #         outFile.write(feature_name)
    #         outFile.write(',')
    # outFile.write('Depressed\n')

    # iterate over networks
    for features in featuresList:
        # iterate over each feature
        for i in range(len(features)):
            if i > 0 and isinstance(features[i], list):
                # get example embedding
                ex = features[i][0]
                ex_row = df_example[df_example['Example'] == ex]
                ex_id  = ex_row['ID']
                emb_vect = df_emb_example.loc[df_emb_example['example id'] == ex_id.values[0]]
                emb_vect = emb_vect.values
                emb_vect =emb_vect[0][:-1]
                # emb_vect = list(emb_vect)

                diff_embed= np.subtract(tweet_vect, emb_vect)
                diff_embed = list(diff_embed)
                diff_embed = ','.join(map(str, diff_embed))
                # diff_embed = ', '.join(diff_embed)
                # print(diff_embed)
                outFile.write(diff_embed)
            elif i == 0:
                outFile.write(str(features[i]))
                # get tweets embedding
                tw_row = df_tweets[df_tweets['tweet'] == features[i]]
                tw_id = tw_row['tweet id']
                # print(tw_id.values[0])
                tweet_vect = tweets_emb.loc[tweets_emb['tweet id'] == tw_id.values[0]]
                tweet_vect = tweet_vect.values
                tweet_vect = tweet_vect[0][:-1]
            else:
                outFile.write(str(features[i]))
            if i < len(features) - 1:
                outFile.write(',')

        outFile.write('\n')
    print("%s Final dataset is created" %title)
    outFile.close()

# Creates a list containing 5 lists, each of 8 items, all set to 0
cat, ex = 8, 10
aucs = [[0 for x in range(1,cat)] for y in range(1,ex)]
stds = [[0 for x in range(1,cat)] for y in range(1,ex)]
prs = [[0 for x in range(1,cat)] for y in range(1,ex)]


def compute_perfomance(num_cat,num_ex,dysfunc_dataset, df_tweets):
    title  = str(num_cat) + str(num_ex)

    sub_df          = dysfunc_dataset[(dysfunc_dataset['category id']<=num_cat) & (dysfunc_dataset['example subset'] <= num_ex)]
    sub_df = sub_df[['Category', 'Example', 'ID']]
    sub_df.index = range(len(sub_df))
    sub_df['ID'] = sub_df.index
    sub_df_examples = sub_df['Example']
    sub_df_features_names = sub_df['Category'].unique()
    sub_df.to_excel(title + '-thoughts_categories.xlsx',  index=False)
    sub_df.to_csv(title + '-thoughts_categories.csv',  index=False)

    textfile = open("%s-features_names.txt" % title, "w", encoding='utf8')
    for element in sub_df_features_names:
        textfile.write(element + "\n")
    textfile.close()

    textfile = open("%s-dysfunctional_thoughts_examples.txt" % title, "w", encoding='utf8')
    for element in sub_df_examples:
        textfile.write(element + "\n")
    textfile.close()

    data_ex = Dataset('%s-dysfunctional_thoughts_examples.txt' % title)
    sentence_sim = SentenceSimilarity2(data_ex)
    emb_examples    = sentence_sim.embedded_sentences
    emb_examples_df = pd.DataFrame.from_records(emb_examples)
    emb_examples_df.to_csv(title + '-examples-emb.csv', index=False)
    df_emb_example = pd.read_csv(title + '-examples-emb.csv')
    df_emb_example['example id'] = sub_df['ID']
    thoughts_dict = {}
    xls = ExcelFile(title + '-thoughts_categories.xlsx')
    df = xls.parse(xls.sheet_names[0])
    df.columns = ['Category', 'Example', 'ID']
    columns = list(df)
    for r in range(0, df.shape[0]):
        for i in columns:
            if i == 'Category':
                if df[i][r] not in thoughts_dict.keys():
                    thoughts_dict[df[i][r]] = [df['Example'][r]]
                    break
                else:
                    thoughts_dict[df[i][r]].append(df['Example'][r])
                    break
            else:
                break

    df_example = pd.read_csv(title + '-thoughts_categories.csv')


    all_tweets = df_tweets['tweet']
    all_tweets = all_tweets.dropna()
    ard = df_tweets[df_tweets['Depressed'] == 1]
    ard = np.array(ard['tweet'])

    textfile = open("experiment8-tweets.txt", "w", encoding='utf8')
    for element in all_tweets:
        textfile.write(element + "\n")
    textfile.close()

    twee            = Dataset('experiment8-tweets.txt')
    sentence_sim2   = SentenceSimilarity2(twee)
    emb_tweets      = sentence_sim2.embedded_sentences
    emb_tweets_df   = pd.DataFrame.from_records(emb_tweets)
    emb_tweets_df.to_csv('experiment8-bert-original-emb.csv', index=False)

    featuresList = []
    ys = []
    for tweet in all_tweets:
        words_count = 0
        featureDict = get_thoughts_features(tweet, sentence_sim, thoughts_dict)  # get 7 top types of thinking and distances

        tweet_features = []
        tweet_features.append(tweet)
        with open('%s-features_names.txt' % title, 'r') as feat_order:
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
            # tweet_features.append(1)
            ys.append(1)
        else:
            # tweet_features.append(0)
            ys.append(0)

        # Append this features vector of the tweet onto the master featuresList
        featuresList.append(tweet_features)

    # output file for features
    outfilename = title + '-experiment8-final-set-emb.csv'
    # Output features to file
    output(outfilename, featuresList, title, df_example, df_emb_example)

    dataset = pd.read_csv(title + '-experiment8-final-set-emb.csv', header=None)
    dataset.fillna(0)
    dataset['Depressed'] = ys
    dataset.to_csv(title + '-experiment8-final-set-emb.csv',header=None)
    y = dataset['Depressed']
    bert = pd.read_csv('experiment8-bert-original-emb.csv')

    mean_emb = []
    for i,entity in dataset.iterrows():
        dys = []
        for d in range (0,num_cat):
            array = dataset.iloc[i,(d*768)+1:(d*768)+769]
            dys.append(array.values)

        m = np.mean(dys, axis = 0)
        mean_emb.append(m)

    emb_mean_df = pd.DataFrame.from_records(mean_emb)
    emb_mean_df.to_csv(title +'-set8-emb-mean.csv')
    mean_emb = pd.read_csv(title +'-set8-emb-mean.csv')
    mean_emb = mean_emb.iloc[:,1:]

    mean_emb = mean_emb.to_numpy()
    kfold = StratifiedKFold(n_splits=10, shuffle=False)
    model = LogisticRegression(penalty='l2', max_iter=10000000, solver='liblinear')
    name = title + 'exp8'
    # mean_emb = np.array(mean_emb)
    mean_auc, std = draw_cv_roc_curve(model, kfold, mean_emb, y, name, title='Cross Validated ROC')
    pr            = draw_cv_pr_curve(model, kfold, mean_emb, y, name, title='Cross Validated PR')
    aucs[num_cat][num_ex] = mean_auc
    stds[num_cat][num_ex] = std
    prs[num_cat][num_ex] = pr


for i in range(1,8):
    for j in range(1,10):
        compute_perfomance(i,j,dt,df_tweets)


with open("aucs-exp8.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(aucs)
f.close()


with open("stds-exp8.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(stds)
f.close()


with open("prs-exp8.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(prs)
f.close()


# compute_perfomance(2,3,dt, df_tweets)