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
from numpy import where
from sklearn.preprocessing import StandardScaler

result = pd.read_csv("bert-dys-emb-combined.csv")
result.drop(result[(result['polarity'] >= 0 ) & (result['Depressed'] == 0)].index, inplace=True)

# result = pd.read_csv("experiment5-final-set-emb.csv")

bert       = result.iloc[:,5382:]
# bert = result.sample(5)
y = result['Depressed'].astype('int')
# bert = bert.iloc[:,5382:]
print(bert)
data1                 = result.iloc[:,1:5377]
# data1 = data1.sample(100)
print(data1)
print(result['Depressed'])
# bert                 = pd.read_csv('experiment5-emb.csv')
x       = bert.to_numpy()
# print(bert)
# data1 = data1.sample(5)
# data1 = data1.iloc[:,:-1]
print(data1)
d1                    = data1.to_numpy()

x = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, result[['Depressed']].astype('int')], axis = 1)
print(finalDf)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA - BERT', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Depressed'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

x = StandardScaler().fit_transform(d1)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, result[['Depressed']].astype('int')], axis = 1)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA - DIFF', fontsize = 20)
targets = [1, 0]
colors = ['r', 'g']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Depressed'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# y=result['Depressed'].astype('int')
#
# for class_value in range(2):
# 	# get row indexes for samples with this class
# 	row_ix = where(y == class_value)
# 	# create scatter of these samples
# 	plt.scatter(d1[row_ix, 0], d1[row_ix, 1])
# # show the plot
# plt.show()

from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2).fit_transform(d1)
#
# df = pd.DataFrame()
# df['tweet'] = result['tweet']
# df['class'] = result['Depressed']
# # Add to dataframe for convenience
# df['x'] = tsne[:, 0]
# df['y'] = tsne[:, 1]
# color= ['red' if l == 0 else 'green' for l in labels]
# FS = (10, 8)
# fig, ax = plt.subplots(figsize=FS)
# # Make points translucent so we can visually identify regions with a high density of overlapping points
# plt.scatter(df.x, df.y, alpha=.1, color =color)
# plt.savefig("tsne-diff-mean7")
# plt.show()
#
# tsne = TSNE(n_components=2).fit_transform(x)
#
# df = pd.DataFrame()
# df['tweet'] = result['tweet']
# df['class'] = result['Depressed']
# # Add to dataframe for convenience
# df['x'] = tsne[:, 0]
# df['y'] = tsne[:, 1]
# color= ['red' if l == 0 else 'green' for l in labels]
# FS = (10, 8)
# fig, ax = plt.subplots(figsize=FS)
# # Make points translucent so we can visually identify regions with a high density of overlapping points
# plt.scatter(df.x, df.y, alpha=.1, color =color)
# plt.savefig("tsne-bert7")
# plt.show()
#
# def plot_bg(bg_alpha=.01, figsize=(13, 9), emb_2d=None):
#     """Create and return a plot of all our movie embeddings with very low opacity.
#     (Intended to be used as a basis for further - more prominent - plotting of a
#     subset of movies. Having the overall shape of the map space in the background is
#     useful for context.)
#     """
#     if emb_2d is None:
#         emb_2d = tsne
#     fig, ax = plt.subplots(figsize=figsize)
#     X = emb_2d[:, 0]
#     Y = emb_2d[:, 1]
#     ax.scatter(X, Y, alpha=bg_alpha)
#     return ax
#
#
# def plot_with_annotations(label_indices, text=True, labels=None, alpha=1, **kwargs):
#     ax = plot_bg(**kwargs)
#     Xlabeled = tsne[label_indices, 0]
#     Ylabeled = tsne[label_indices, 1]
#     if labels is not None:
#         for x, y, label in zip(Xlabeled, Ylabeled, labels):
#             ax.scatter(x, y, alpha=alpha, label=label, marker='1',
#                        s=90,
#                        )
#         fig.legend()
#     else:
#         ax.scatter(Xlabeled, Ylabeled, alpha=alpha, color='green')
#
#
#     return ax











# XTrain, XTest, yTrain, yTest = train_test_split(x,y, random_state=123, test_size=0.2)

from numpy import unique
# from numpy import where
# from sklearn.datasets import make_classification
# from sklearn.cluster import MiniBatchKMeans
# from matplotlib import pyplot
# # define dataset
# # define the model
# from numpy import unique
# model = MiniBatchKMeans(n_clusters=2)
# # fit the model
# model.fit(x)
# # assign a cluster to each example
# yhat = model.predict(x)
# # retrieve unique clusters
# clusters = unique(yhat)
# # create scatter plot for samples from each cluster
# for cluster in clusters:
# 	# get row indexes for samples with this cluster
# 	row_ix = where(yhat == cluster)
# 	# create scatter of these samples
# 	pyplot.scatter(x[row_ix, 0], x[row_ix, 1])
# # show the plot
# plt.savefig("mini-batch-kmeans-diff7")
#
# pyplot.show()
#
# model = MiniBatchKMeans(n_clusters=2)
# # fit the model
# model.fit(x)
# # assign a cluster to each example
# yhat = model.predict(x)
# # retrieve unique clusters
# clusters = unique(yhat)
# # create scatter plot for samples from each cluster
# for cluster in clusters:
# 	# get row indexes for samples with this cluster
# 	row_ix = where(yhat == cluster)
# 	# create scatter of these samples
# 	pyplot.scatter(d1[row_ix, 0], d1[row_ix, 1])
# # show the plot
# plt.savefig("mini-batch-kmeans-bert7")
#
# pyplot.show()




# # x_d1                  = d1[:,1:11]                                                     # 1:7   -> the proposed features, 7:11   -> the other common features
# #
# # d2                    = data2.to_numpy()
# # x_d2                  = d2[:,1:17]
# print(data1)
# y=result['Depressed'].astype('int')
#
new_bert1        = np.concatenate((d1,x ), axis=1)
#


# from sklearn.cross_validation import cross_val_score
# # use the same model as before
# knn = KNeighborsClassifier(n_neighbors = 5)
# # X,y will automatically devided by 5 folder, the scoring I will still use the accuracy
# scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
# print(scores.mean())
# # 0.973333333333
#
#
# # # new_bert2        = np.concatenate((x_d2[:,0:16],x ), axis=1)
# #
from sklearn.neighbors import KNeighborsClassifier
#
# ############################################################################################
# XTrain, XTest, yTrain, yTest = train_test_split(x,y, random_state=123, test_size=0.2)
#
# neigh = KNeighborsClassifier(n_neighbors=12)
# neigh.fit(XTrain, yTrain)
# # KNeighborsClassifier(...)
# p = neigh.predict(XTest)
# cm = confusion_matrix(yTest, p)
# cm
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in p])
from sklearn.metrics.cluster import adjusted_rand_score
#
# print("adjusted")
# print(adjusted_rand_score(yTest, y_pred_))
# print("roc")
# print(roc_auc_score(yTest,y_pred_))
# # print(average_precision_score(yTest,y_pred_))
# precision, recall, _ = precision_recall_curve(yTest,y_pred_)
# plt.plot(recall, precision, color='b',
#              label=r'Precision-Recall (AUC = %0.3f)' % (average_precision_score(yTest, y_pred_)),
#              lw=2, alpha=.8)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('bert5')
# plt.legend(loc="lower right")
# plt.savefig('bert5' + '.png', bbox_inches='tight')
# plt.show()
#
#
# # XTrain, XTest, yTrain, yTest = train_test_split(d1,y, random_state=123, test_size=0.2)
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=12)
# neigh.fit(XTrain, yTrain)
# # KNeighborsClassifier(...)
# p = neigh.predict(XTest)
# cm = confusion_matrix(yTest, p)
# cm
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in p])
# print("diff")
# print("adjusted")
# print(adjusted_rand_score(yTest, y_pred_))
#
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
# print("roc")
# print(roc_auc_score(yTest,y_pred_))
# # print(average_precision_score(yTest,y_pred_))
# precision, recall, _ = precision_recall_curve(yTest,y_pred_)
# plt.plot(recall, precision, color='b',
#              label=r'Precision-Recall (AUC = %0.3f)' % (average_precision_score(yTest, y_pred_)),
#              lw=2, alpha=.8)
#
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('diff52')
# plt.legend(loc="lower right")
# plt.savefig('diff52' + '.png', bbox_inches='tight')
# plt.show()

# ####################################################################################################
# XTrain, XTest, yTrain, yTest = train_test_split(new_bert1,y, random_state=123, test_size=0.2)
# #
# # from sklearn.neighbors import KNeighborsClassifier
# # neigh = KNeighborsClassifier(n_neighbors=10)
# # neigh.fit(XTrain, yTrain)
# # # KNeighborsClassifier(...)
# # p = neigh.predict(XTest)
# # cm = confusion_matrix(yTest, p)
# # cm
# # cm_argmax = cm.argmax(axis=0)
# # cm_argmax
# # y_pred_ = np.array([cm_argmax[i] for i in p])
# # # def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, sorted_labels: list):
# # #
# # #     matrix = confusion_matrix(actual_classes, predicted_classes)
# # #     print(matrix)
# # #     plt.figure(figsize=(12.8, 6))
# # #     sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
# # #     plt.xlabel('Predicted');
# # #     plt.ylabel('Actual');
# # #     plt.title('Confusion Matrix')
# # #     plt.savefig("unsupervised-bert-new2")
# # #     plt.show()
# # # plot_confusion_matrix(yTest, p,["1", "0"])
# # print(roc_auc_score(yTest,y_pred_))
# # print(average_precision_score(yTest,y_pred_))
# # print(precision_recall_curve(yTest,y_pred_))
#
# error_rate = []
# for i in range(1,40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(XTrain,yTrain)
#     pred_i = knn.predict(XTest)
#     error_rate.append(np.mean(pred_i != yTest))
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.savefig("exp5-k")
# req_k_value = error_rate.index(min(error_rate)) + 1
# print("Minimum error:-", min(error_rate), "at K =", req_k_value)
#
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
dbscan = DBSCAN(eps=0.3)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(x)
dbscan.fit(x)
kmeans = KMeans(
    init="k-means++",
    n_clusters=2,
    n_init=10,
    max_iter=300,
    random_state=42
 )
kmeans.fit(x)
ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
# print('bert8 - kmeans')
# print(ari_kmeans)
ari_dbscan = adjusted_rand_score(y, dbscan.labels_)
# print('bert8 - dbscan')
# print(ari_dbscan)
kmeans = KMeans(n_clusters=2, random_state=30)
labels = kmeans.fit_predict(x)
print(calinski_harabasz_score(x, labels))
print(davies_bouldin_score(x, labels))
print(adjusted_rand_score(y, labels))

print(roc_auc_score(y,labels))

scaled_features = scaler.fit_transform(d1)
dbscan.fit(d1)

kmeans = KMeans(
    init="k-means++",
    n_clusters=2,
    n_init=10,
    max_iter=300,
    random_state=42
 )
# kmeans.fit(d1)
# ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
# print('diff8 - kmeans')
# print(ari_kmeans)
# ari_dbscan = adjusted_rand_score(y, dbscan.labels_)
# print(ari_dbscan)
kmeans2 = KMeans(n_clusters=2, random_state=30)

labels = kmeans2.fit_predict(d1)

print(calinski_harabasz_score(d1, labels))
print(davies_bouldin_score(d1, labels))
print(adjusted_rand_score(y, labels))

print(roc_auc_score(y,labels))

# create kmeans object
# kmeans = KMeans(n_clusters=2)
# # fit kmeans object to data
# kmeans.fit(x)
# # print location of clusters learned by kmeans object
# print(kmeans.cluster_centers_)
# # save new clusters for chart
# y_km = kmeans.fit_predict(x)
# plt.scatter(x[y_km ==0,0], x[y_km == 0,1], s=100, c='red')
# plt.scatter(x[y_km ==1,0], x[y_km == 1,1], s=100, c='black')
# plt.show()

# kmeans = KMeans(n_clusters=2)
# # fit kmeans object to data
# kmeans.fit(d1)
# # print location of clusters learned by kmeans object
# print(kmeans.cluster_centers_)
# # save new clusters for chart
# y_km = kmeans.fit_predict(x)
# plt.scatter(d1[y_km ==0,0], d1[y_km == 0,1], s=100, c='red')
# plt.scatter(d1[y_km ==1,0], d1[y_km == 1,1], s=100, c='black')
# plt.show()