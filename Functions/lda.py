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
from sklearn import metrics


result = pd.read_csv("bert-dys-emb-combined8.csv")
result.dropna(inplace=True)
result.index = range(len(result))

# bert
bert       = result.iloc[:,5380:]
bert       = bert.iloc[:,:-1]
x       = bert.to_numpy()

#diff dys
data                 = result.iloc[:,1:5377]
data = data.to_numpy()

#mean
mean = pd.read_csv('set8-emb-mean2.csv')
mean = mean.iloc[:,:-1]
mean = mean.to_numpy()

y = result['Depressed'].astype('int')




more = pd.read_csv('experiment8-final-set-dif-emb-allmore-filled.csv')
more.dropna(inplace=True)
more.index = range(len(more))
y2 = more['Depressed'].astype('int')

# All depression symtoms
more_all = more.iloc[:,1:]
more_all = more_all.to_numpy()

# A subset of
more = more.iloc[:,1:-4609]
more = more.to_numpy()




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
plt.plot(range(0,principalComponents2.shape[1]),cumulative_variance)
plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
plt.xlabel('Number of Principal Components DIFF')
plt.ylabel('% of total variance accounted for')
plt.legend()
plt.show()
#
# pca = PCA()
# principalComponents_more = pca.fit_transform(more)
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(range(0,principalComponents_more.shape[1]),cumulative_variance)
# plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
# plt.xlabel('Number of Principal Components MORE')
# plt.ylabel('% of total variance accounted for')
# plt.legend()
# plt.show()
lda = LinearDiscriminantAnalysis()
lda.fit_transform(more,y2)

pca = PCA()
principalComponents_more_all = pca.fit_transform(more_all)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(0,principalComponents_more_all.shape[1]),cumulative_variance)
plt.plot([0,len(pca.components_)],[0.9,0.9], '--', label='90% Variance')
plt.xlabel('Number of Principal Components MORE all')
plt.ylabel('% of total variance accounted for')
plt.legend()
plt.show()



print("bert")
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42, stratify=y)
lda = LinearDiscriminantAnalysis()
X_train1 = np.array(X_train)
y_train1 = np.array(y_train)
lda.fit(X_train1, y_train1)
X_test1 = np.array(X_test)
y_test1 = np.array(y_test)
y_pred = lda.predict(X_test1)
print('Accuracy of Linear Discriminant Analysis Model on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print("F1 of bert model  is",metrics.roc_auc_score(y_test, y_pred))
confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix1)


print("mean")
X_train, X_test, y_train, y_test = train_test_split(mean, y, test_size=0.33, random_state=42, stratify=y)
lda = LinearDiscriminantAnalysis()
X_train1 = np.array(X_train)
y_train1 = np.array(y_train)
lda.fit(X_train1, y_train1)
X_test1 = np.array(X_test)
y_test1 = np.array(y_test)
y_pred = lda.predict(X_test1)
print('Accuracy of Linear Discriminant Analysis Model on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print("F1 of mean model  is",metrics.roc_auc_score(y_test, y_pred))
confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix1)

print("subset")
X_train, X_test, y_train, y_test = train_test_split(more
, y2, test_size=0.33, random_state=42, stratify=y)
lda = LinearDiscriminantAnalysis()
X_train1 = np.array(X_train)
y_train1 = np.array(y_train)
lda.fit(X_train1, y_train1)
X_test1 = np.array(X_test)
y_test1 = np.array(y_test)
y_pred = lda.predict(X_test1)
print('Accuracy of Linear Discriminant Analysis Model on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print("F1 of subset model  is",metrics.roc_auc_score(y_test, y_pred))
confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix1)

print("all symptoms")
X_train, X_test, y_train, y_test = train_test_split(more_all
, y2, test_size=0.33, random_state=42, stratify=y)
lda = LinearDiscriminantAnalysis()
X_train1 = np.array(X_train)
y_train1 = np.array(y_train)
lda.fit(X_train1, y_train1)
X_test1 = np.array(X_test)
y_test1 = np.array(y_test)
y_pred = lda.predict(X_test1)
print('Accuracy of Linear Discriminant Analysis Model on test set: {:.2f}'.format(accuracy_score(y_test, y_pred)))
print("F1 of all model  is",metrics.roc_auc_score(y_test, y_pred))
confusion_matrix1 = confusion_matrix(y_test, y_pred)
print(confusion_matrix1)
