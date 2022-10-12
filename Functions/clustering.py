from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


result = pd.read_csv("bert-dys-emb-combined8.csv")
result.dropna(inplace=True)
result.index = range(len(result))

# bert
bert       = result.iloc[:,5380:]
# bert = pd.read_csv('experiment8-tweets-original-emb-cut-dep.csv')
bert       = bert.iloc[:,:-1]

x       = bert.to_numpy()

#diff dys


data                 = result.iloc[:,1:5377]
data = data.to_numpy()

#mean
mean = pd.read_csv('set8-emb-mean2.csv')
mean = mean.iloc[:,:-1]
mean.to_numpy()

y = result['Depressed'].astype('int')




more = pd.read_csv('experiment8-final-set-dif-emb-allmore-filled.csv')
# print(more.sample())
#
print(len(more))

more.dropna(inplace=True)
# more.to_csv('experiment8-final-set-dif-emb-allmore-filled.csv')
print(len(more))
more.index = range(len(more))
# print(more.sample())

more_all = more.iloc[:,1:]
more_all = more_all.to_numpy()
y2 = more['Depressed'].astype('int')
more = more.iloc[:,1:-4609]
more = more.to_numpy()

# ymore = pd.read_csv('experiment8-final-set-emb.csv')
# ymore.dropna(inplace=True)
# ymore.index = range(len(ymore))


#
#
from sklearn.decomposition import PCA
# pca = PCA()
# principalComponents = pca.fit_transform(x)
# import matplotlib.pyplot as plt
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(0,principalComponents.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components BERT')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()
#
pca = PCA()
principalComponents2 = pca.fit_transform(data)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cm = list(cumulative_variance)
pcindex1 = cm.index(max(cm))
print(pcindex1)

# plt.plot(range(0,principalComponents2.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components DIFF')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()

pca = PCA()
principalComponents_more = pca.fit_transform(more)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cm = list(cumulative_variance)
pcindex2 = 0
i = 0
for c in cm:

    c = round(c,2)
    print(c)
    if c == 0.90:
        print(c)
        pcindex2 = i
        break
    i = i +1
print(pcindex2)

# plt.plot(range(0,principalComponents_more.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components MORE')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()

pca = PCA()
principalComponents_more_all = pca.fit_transform(more_all)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cm = list(cumulative_variance)
pcindex3 = cm.index(max(cm))
print(pcindex3)

# plt.plot(range(0,principalComponents_more_all.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components MORE all')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()

def clustering_exp(data,l, s):

    kmeans = KMeans(init="k-means++",n_clusters=2)
    print(s)
    labels = kmeans.fit_predict(data)
    print('Adjusted Rand Score: %.3f' % adjusted_rand_score(l,labels))
    print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(data,labels))
    print('Davies Bouldin Score: %.3f' % davies_bouldin_score(data,labels))
    print('Silhouetter Score: %.3f' % silhouette_score(data,labels, metric='euclidean'))
    print('ROC AUC Score: %.3f' % roc_auc_score(l,labels))


print("bert")
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(x,y))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(x,y))
print('Silhouetter Score: %.3f' % silhouette_score(x,y, metric='euclidean'))
# clustering_exp(x,y,"BERT 8")
# clustering_exp(more,y2,"DIFF 8")



# y2 = more['Depressed'].astype('int')
# for i in range(1,17):
#     new = more.iloc[:, :-(i*768)]
#     # print(more)
#     new = new.to_numpy()
#     print(i)
#     print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(new,y2))
#     print('Davies Bouldin Score: %.3f' % davies_bouldin_score(new,y2))
#     print('Silhouetter Score: %.3f' % silhouette_score(new,y2, metric='euclidean'))


# clustering_exp(principalComponents_mean[:,:200],y2,"DIFF and more 8")

print('mean')
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(mean,y))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(mean,y))
print('Silhouetter Score: %.3f' % silhouette_score(mean,y, metric='euclidean'))

print('original diff')
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(principalComponents2[:,:pcindex1],y))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(principalComponents2[:,:pcindex1],y))
print('Silhouetter Score: %.3f' % silhouette_score(principalComponents2[:,:pcindex1],y, metric='euclidean'))



# from sklearn.decomposition import PCA
# pca = PCA()
# morepca = pca.fit_transform(more)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(0,morepca.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()
lda = LinearDiscriminantAnalysis()
morelda = lda.fit_transform(more,y2)
print("subset of symtpoms")
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(principalComponents_more[:,:300],y2))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(principalComponents_more[:,:300],y2))
print('Silhouetter Score: %.3f' % silhouette_score(principalComponents_more[:,:300],y2, metric='euclidean'))

print("all symptoms")
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(principalComponents_more_all[:,:pcindex3],y2))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(principalComponents_more_all[:,:pcindex3],y2))
print('Silhouetter Score: %.3f' % silhouette_score(principalComponents_more_all[:,:pcindex3],y2, metric='euclidean'))

# def plot_confusion_matrix(actual_classes: np.array, predicted_classes: np.array, title, sorted_labels: list):
#
#     matrix = confusion_matrix(actual_classes, predicted_classes)
#     print(matrix)
#     plt.figure(figsize=(12.8, 6))
#     sns.heatmap(matrix, annot=True, xticklabels=sorted_labels, yticklabels=sorted_labels, cmap="Blues", fmt="g")
#     plt.xlabel('Predicted');
#     plt.ylabel('Actual');
#     plt.title('Confusion Matrix')
#     plt.savefig(title)
#     plt.show()
#
# X_train, X_test, y_train, y_test = train_test_split(x,y, random_state=123, test_size=0.2)
#
# kmeans = KMeans(
#      init="random",
#       n_clusters=2,
#      n_init=10,
#       max_iter=300,
#     random_state=42
#     )
#
# kmeans.fit(x)
# ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
# print(ari_kmeans)
# error_rate = []
# for i in range(1,40):
#  knn = KNeighborsClassifier(n_neighbors=i)
#  knn.fit(X_train,y_train)
#  pred_i = knn.predict(X_test)
#  error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
#          marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value BERT')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
#
# from sklearn import metrics
# #Train Model and Predict
# k = 33
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# Pred_y = neigh.predict(X_test)
# print("F1 of bert model at K=35 is",metrics.roc_auc_score(y_test, Pred_y))
#
# cm = confusion_matrix(y_test, Pred_y)
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in Pred_y])
# plot_confusion_matrix(y_test, y_pred_,'bert', ["1", "0"])
#
# X_train, X_test, y_train, y_test = train_test_split(mean,y, random_state=123, test_size=0.2)
# kmeans = KMeans(
#      init="random",
#       n_clusters=2,
#      n_init=10,
#       max_iter=300,
#     random_state=42
#     )
#
# kmeans.fit(mean)
# ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
# print(ari_kmeans)
# error_rate = []
# for i in range(1,40):
#  knn = KNeighborsClassifier(n_neighbors=i)
#  knn.fit(X_train,y_train)
#  pred_i = knn.predict(X_test)
#  error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
#          marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value Mean')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
#
# from sklearn import metrics
# #Train Model and Predict
# k = 17
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# Pred_y = neigh.predict(X_test)
# print("F1 of mean model at K=17 is",metrics.roc_auc_score(y_test, Pred_y))
#
# cm = confusion_matrix(y_test, Pred_y)
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in Pred_y])
# plot_confusion_matrix(y_test, y_pred_,'mean', ["1", "0"])
#
#
# X_train, X_test, y_train, y_test = train_test_split(principalComponents2[:,:200],y, random_state=123, test_size=0.2)
# kmeans = KMeans(
#      init="random",
#       n_clusters=2,
#      n_init=10,
#       max_iter=300,
#     random_state=42
#     )
#
# kmeans.fit(data)
# ari_kmeans = adjusted_rand_score(y, kmeans.labels_)
# print(ari_kmeans)
# error_rate = []
# for i in range(1,40):
#  knn = KNeighborsClassifier(n_neighbors=i)
#  knn.fit(X_train,y_train)
#  pred_i = knn.predict(X_test)
#  error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
#          marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value Mean')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
#
# from sklearn import metrics
# #Train Model and Predict
# k = 6
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# Pred_y = neigh.predict(X_test)
# print("F1 of dys model at K=7 is",metrics.roc_auc_score(y_test, Pred_y))
#
# cm = confusion_matrix(y_test, Pred_y)
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in Pred_y])
# plot_confusion_matrix(y_test, y_pred_,'dys', ["1", "0"])
#
# X_train, X_test, y_train, y_test = train_test_split(principalComponents_more[:,:230],y2, random_state=123, test_size=0.2)
# kmeans = KMeans(
#      init="random",
#       n_clusters=2,
#      n_init=10,
#       max_iter=300,
#     random_state=42
#     )
#
# kmeans.fit(more)
# ari_kmeans = adjusted_rand_score(y2, kmeans.labels_)
# print(ari_kmeans)
# error_rate = []
# for i in range(1,40):
#  knn = KNeighborsClassifier(n_neighbors=i)
#  knn.fit(X_train,y_train)
#  pred_i = knn.predict(X_test)
#  error_rate.append(np.mean(pred_i != y_test))
#
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',
#          marker='o',markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value Mean')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
#
# from sklearn import metrics
# #Train Model and Predict
# k = 11
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# Pred_y = neigh.predict(X_test)
# print("F1 of more model at K=10 is",metrics.roc_auc_score(y_test, Pred_y))
# cm = confusion_matrix(y_test, Pred_y)
# cm_argmax = cm.argmax(axis=0)
# cm_argmax
# y_pred_ = np.array([cm_argmax[i] for i in Pred_y])
# plot_confusion_matrix(y_test, y_pred_,'more', ["0", "1"])
#
#
#
