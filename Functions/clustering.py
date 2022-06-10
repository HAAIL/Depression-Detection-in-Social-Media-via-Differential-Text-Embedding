from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import numpy as np
result = pd.read_csv("bert-dys-emb-combined8.csv")
result.dropna(inplace=True)
result.index = range(len(result))

bert       = result.iloc[:,5379:]
y = result['Depressed'].astype('int')
data1                 = result.iloc[:,1:5377]
# data1 = pd.read_csv('set7-emb-mean2.csv')
# print(bert)
# print(data1)
x       = bert.to_numpy()
more = pd.read_csv('exp8-diff-emb0all-sym.csv')
# more = pd.read_csv('experiment7-final-set-original-emb.csv')

# more.dropna(inplace=True)
# more.index = range(len(more))
#
#
y2 = more['Depressed'].astype('int')
more = more.iloc[:,2:-3844]
# print(more)
more = more.to_numpy()
d1                    = data1.to_numpy()
# # new_bert1        = np.concatenate((d1,x ), axis=1)
#
#
#
# from sklearn.decomposition import PCA
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
# pca = PCA()
# principalComponents2 = pca.fit_transform(d1)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(0,principalComponents2.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components DIFF')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()
#
# pca = PCA()
# principalComponents_mean = pca.fit_transform(more)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(0,principalComponents_mean.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components MORE')
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


print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(x,y))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(x,y))
print('Silhouetter Score: %.3f' % silhouette_score(x,y, metric='euclidean'))
# clustering_exp(x,y,"BERT 8")
# clustering_exp(more,y2,"DIFF 8")
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(more,y2))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(more,y2))
print('Silhouetter Score: %.3f' % silhouette_score(more,y2, metric='euclidean'))
# clustering_exp(principalComponents_mean[:,:200],y2,"DIFF and more 8")
print('Calinski Harabasz Score: %.3f' % calinski_harabasz_score(d1,y))
print('Davies Bouldin Score: %.3f' % davies_bouldin_score(d1,y))
print('Silhouetter Score: %.3f' % silhouette_score(d1,y, metric='euclidean'))