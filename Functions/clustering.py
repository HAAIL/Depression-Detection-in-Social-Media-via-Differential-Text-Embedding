from sklearn.cluster import KMeans
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer



data1                 = pd.read_csv('experiment5-final-set-LIWC-Analysis3.csv')
data1= data1.dropna()
##
data2                 = pd.read_csv('experiment10-final-set-LIWC-Analysis3.csv')
data2= data2.dropna()

bert                 = pd.read_csv('experiment5-emb.csv')
x       = bert.to_numpy()

d1                    = data1.to_numpy()
x_d1                  = d1[:,1:11]                                                     # 1:7   -> the proposed features, 7:11   -> the other common features
#
d2                    = data2.to_numpy()
x_d2                  = d2[:,1:17]

y=d1[:,11].astype('int')

new_bert1        = np.concatenate((x_d1[:,0:7],x ), axis=1)

new_bert2        = np.concatenate((x_d2[:,0:16],x ), axis=1)

XTrain, XTest, yTrain, yTest = train_test_split(x,y, random_state=123, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(XTrain, yTrain)
# KNeighborsClassifier(...)
p = neigh.predict(XTest)
cm = confusion_matrix(yTest, p)
cm
cm_argmax = cm.argmax(axis=0)
cm_argmax
y_pred_ = np.array([cm_argmax[i] for i in p])
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
print(roc_auc_score(yTest,y_pred_))
print(average_precision_score(yTest,y_pred_))
print(precision_recall_curve(yTest,y_pred_))

XTrain, XTest, yTrain, yTest = train_test_split(new_bert1,y, random_state=123, test_size=0.2)
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(XTrain, yTrain)
# KNeighborsClassifier(...)
p = neigh.predict(XTest)
cm = confusion_matrix(yTest, p)
cm
cm_argmax = cm.argmax(axis=0)
cm_argmax
y_pred_ = np.array([cm_argmax[i] for i in p])
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
print(roc_auc_score(yTest,y_pred_))
print(average_precision_score(yTest,y_pred_))
print(precision_recall_curve(yTest,y_pred_))

XTrain, XTest, yTrain, yTest = train_test_split(new_bert2,y, random_state=123, test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(XTrain, yTrain)
# KNeighborsClassifier(...)
p = neigh.predict(XTest)
cm = confusion_matrix(yTest, p)
cm
cm_argmax = cm.argmax(axis=0)
cm_argmax
y_pred_ = np.array([cm_argmax[i] for i in p])
# def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
#
#     matrix = confusion_matrix(actual_classes, predicted_classes)
#     print(matrix)
#     plt.figure(figsize=(12.8, 6))
#     sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
#     plt.xlabel('Predicted');
#     plt.ylabel('Actual');
#     plt.title('Confusion Matrix')
#     plt.savefig("unsupervised-bert-new2")
#     plt.show()
# plot_confusion_matrix(yTest, p,["1", "0"])
print(roc_auc_score(yTest,y_pred_))
print(average_precision_score(yTest,y_pred_))
print(precision_recall_curve(yTest,y_pred_))
