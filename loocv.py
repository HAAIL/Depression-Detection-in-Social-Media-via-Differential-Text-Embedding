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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import ConfusionMatrixDisplay,precision_score,recall_score,confusion_matrix
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn import svm
from numpy import interp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
from numpy import absolute
result = pd.read_csv("bert-dys-emb-combined8.csv")
result.dropna(inplace=True)
result.index = range(len(result))

# bert
bert       = result.iloc[:,5380:]
bert       = bert.iloc[:,:-1]
x       = bert.to_numpy()
y = result['Depressed'].astype('int')
bert8 = pd.read_csv('experiment8-tweets-original-emb-cut-dep.csv')
bert8 = bert8.iloc[:,:-1]
x = bert8.to_numpy()




#diff dys
data                 = result.iloc[:,1:5377]
data = data.to_numpy()

#mean
# mean = pd.read_csv('all-emb-mean2.csv')
# mean = mean.iloc[:,:-1]
# mean = mean.to_numpy()

# more
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

dfm = pd.read_csv('experiment8-final-set-dif-emb-allmore-filled.csv', delimiter=',')
dfm.dropna(inplace=True)
print(len(dfm))
dfm.index = range(len(dfm))
df_headersName=pd.read_csv('experiment8-final-set-dif-emb-allmore-filled.csv', nrows=1).columns.tolist()

df_attrName = [
'Mind reading',
'Labelling',
'Fortune telling',
'Overgeneralising',
'Emotional Reasoning',
'Personalising',
'Shoulds and Musts',
'Loss of insight',
'Pleasure loss',
'Interest loss',
'Feeling bothered',
'Energy loss',
'Libido',
'Inability to feel',
'Feeling needed',
'Inner tension']
df_total = pd.DataFrame()

for d in range(0, 7):
    # new_f = df_headersName[d]
    # if d == 0 or d == 1 or d ==7:
    new_f = df_headersName[(d * 768) + 1: (d * 768 + 768) + 1]
    print(new_f)

    trainX = dfm[new_f]
        # pca = PCA()
        # pc = pca.fit_transform(trainX)
        # cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        # cm = list(cumulative_variance)
        # pcindex = 0
        # n = 0
        # for c in cm:
        #     c = round(c, 2)
        #     # print(c)
        #     if c == 0.90 or c > 0.9:
        #         # print(c)
        #         pcindex = n
        #         break
        #     n = n + 1
        #
        # df = pd.DataFrame(pc[:, :pcindex])
        # print(df.shape)
    df_pcas = pd.concat([df_total, trainX], axis=1)
    df_total = df_pcas


print(df_total.shape)
pca = PCA()
pc = pca.fit_transform(df_total)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
cm = list(cumulative_variance)
pcindex = 0
n = 0
for c in cm:
    c = round(c, 2)
    # print(c)
    if c == 0.90 or c > 0.9:
        # print(c)
        pcindex = n
        break
    n = n + 1

df_total = pd.DataFrame(pc[:, :230])
# print(df.shape)
df_total = df_total.to_numpy()



# BERT -> a classifier

print("BERT -> a classifier")

model = LogisticRegression(penalty = 'l2', max_iter=10000000)

cv = LeaveOneOut()

scores1 = cross_val_score(model, df_total, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
print(scores1)
print('LogisticRegression Mean auc: %.3f (%.3f)' % (mean(scores1), std(scores1)))
for i in absolute(scores1):
    print(i)


# LDA to reduce dimentionality -> a classifier

print("PCA -> a classifier")

model =LogisticRegression(penalty = 'l2', max_iter=10000000)

scores1 = cross_val_score(data, df_total, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)
print(scores1)
print('LogisticRegression Mean auc: %.3f (%.3f)' % (mean(scores1), std(scores1)))
for i in absolute(scores1):
    print(i)


