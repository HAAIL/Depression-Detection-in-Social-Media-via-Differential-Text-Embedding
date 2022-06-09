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
import re
import time
import scipy
import logging
from typing import List, AnyStr
from datas import Dataset
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC

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


#------------------------------------------------#
#-----------------Experiments--------------------#
#------------------------------------------------#
#
#

data1 = pd.read_csv("set7-original-emb-mean.csv")

d1                    = data1.to_numpy()
# data1= data1.dropna()


result = pd.read_csv("experiment7-final-set-original-emb.csv")
result = result.dropna()
result.index = range(len(result))
y = result['Depressed'].astype('int')



from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve



import numpy as np
from sklearn.neighbors import KNeighborsClassifier


XTrain, XTest, yTrain, yTest = train_test_split(x,y, random_state=123, test_size=0.2)

neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(XTrain, yTrain)
KNeighborsClassifier(...)
p = neigh.predict(XTest)
cm = confusion_matrix(yTest, p)
cm
cm_argmax = cm.argmax(axis=0)
cm_argmax
y_pred_ = np.array([cm_argmax[i] for i in p])
def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):

    matrix = confusion_matrix(actual_classes, predicted_classes)
    print(matrix)
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted');
    plt.ylabel('Actual');
    plt.title('Confusion Matrix')
    plt.savefig("unsupervised-bert")
    plt.show()
plot_confusion_matrix(yTest, p,["1", "0"])
print(roc_auc_score(yTest,y_pred_))
print(average_precision_score(yTest,y_pred_))
print(precision_recall_curve(yTest,y_pred_))

XTrain, XTest, yTrain, yTest = train_test_split(bert_diff,y, random_state=123, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(XTrain, yTrain)
# KNeighborsClassifier(...)
p = neigh.predict(XTest)
cm = confusion_matrix(yTest, p)
cm
cm_argmax = cm.argmax(axis=0)
cm_argmax
y_pred_ = np.array([cm_argmax[i] for i in p])
def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):

    matrix = confusion_matrix(actual_classes, predicted_classes)
    print(matrix)
    plt.figure(figsize=(12.8, 6))
    sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
    plt.xlabel('Predicted');
    plt.ylabel('Actual');
    plt.title('Confusion Matrix')
    plt.savefig("unsupervised-bert-new1")
    plt.show()
plot_confusion_matrix(yTest, p,["1", "0"])
print(roc_auc_score(yTest,y_pred_))
print(average_precision_score(yTest,y_pred_))
print(precision_recall_curve(yTest,y_pred_))