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
# df_total = pd.read_csv('all-emb-mean.csv')
# df_total = df_total.to_numpy()
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
print(df_total.shape)
df_total = df_total.to_numpy()



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

    LDA = LinearDiscriminantAnalysis()
    i = 0
    for train, test in cv.split(X, y):
        Xtrain = LDA.fit_transform(X[train], y[train])
        Xtest = LDA.transform(X[test])

        probas_ = classifier.fit(Xtrain, y[train]).predict_proba(Xtest)
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
    return aucs





from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std

# BERT -> a classifier

# print("BERT -> a classifier")
# XTrain1, XTest1, yTrain1, yTest1 = train_test_split(x,y, random_state=123, test_size=0.2)
#
# # data = DecisionTreeClassifier()
# # data = svm.SVC(kernel='rbf', probability = True)
# data = LogisticRegression(penalty = 'l2', max_iter=10000000)
# # data.fit(XTrain1, yTrain1)
# # y_pred = data.predict(XTest1)
# # confusion_matrix(yTest1, y_pred)
# #
# # ConfusionMatrixDisplay.from_predictions(yTest1, y_pred)
# # plt.show()
# # print("Precision score",precision_score(yTest1,y_pred))
# # print("auc score",roc_auc_score(yTest1,y_pred))
# #
# kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1000, random_state=1)
#
# scores1 = cross_val_score(data, x, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
# print(scores1)
# print('LogisticRegression Mean auc: %.3f (%.3f)' % (mean(scores1), std(scores1)))
#
# # set_mean_all = draw_cv_roc_curve(data, kfold,x, y, title='Cross Validated ROC Of bert log exp 8')
# # print(set_mean_all)
# import json
#
# with open('aucs-bert.txt', 'w') as filehandle:
#     json.dump(scores1.toList(), filehandle)
# for i in scores1:
#     print(i)

# LDA to reduce dimentionality -> a classifier

print("PCA -> LDA -> a classifier")

# LDA = LinearDiscriminantAnalysis()
# X_lda = LDA.fit_transform(df_total, y2)
# print(X_lda.shape)
XTrain2, XTest2, yTrain2, yTest2 = train_test_split(df_total,y2, random_state=123, test_size=0.2)
#
# # data = svm.SVC(kernel='rbf', probability = True)
# data = DecisionTreeClassifier()
# data = LogisticRegression(penalty = 'l2', max_iter=10000000)

# apply Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis()
X_train = lda.fit_transform(XTrain2, yTrain2)
X_test = lda.transform(XTest2)


data =LogisticRegression(penalty = 'l2', max_iter=10000000)
# data.fit(XTrain2, yTrain2)
# y_pred = data.predict(XTest2)
# confusion_matrix(yTest2, y_pred)
#
# ConfusionMatrixDisplay.from_predictions(yTest2, y_pred)
# plt.show()
# print("Precision score",precision_score(yTest2,y_pred))
# print("auc score",roc_auc_score(yTest2,y_pred))
#
# kfold = StratifiedKFold(n_splits=1000, shuffle=False)
# set_mean_all = draw_cv_roc_curve(data, kfold,df_total, y2, title='Cross Validated ROC Of 3 pca- lda - log exp 8')
# print(set_mean_all)
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=1000, random_state=1)

scores1 = cross_val_score(data, df_total, y, scoring='roc_auc', cv=kfold, n_jobs=-1)
print(scores1)
print('LogisticRegression Mean auc: %.3f (%.3f)' % (mean(scores1), std(scores1)))
for i in scores1:
    print(i)
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save(outfile, scores1)


# LDA as a classifier

print("LDA as a classifier - BERT")
# BERT
# XTrain1, XTest1, yTrain1, yTest1 = train_test_split(x,y, random_state=123, test_size=0.2)
#
# #fitting the LDA model
# LDA = LinearDiscriminantAnalysis()
# LDA.fit(XTrain1, yTrain1)
# LDA_pred=LDA.predict(XTest1)
#
# ConfusionMatrixDisplay.from_predictions(yTest1, LDA_pred)
# plt.show()
# print("Precision score",precision_score(yTest1,LDA_pred))
# print("auc score",roc_auc_score(yTest1,y_pred))
#
#
# print("LDA as a classifier - all 0-10")
#
# XTrain2, XTest2, yTrain2, yTest2 = train_test_split(df_total,y2, random_state=123, test_size=0.2)
#
# #fitting the LDA model
# LDA = LinearDiscriminantAnalysis()
# LDA.fit(XTrain2, yTrain2)
# LDA_pred=LDA.predict(XTest2)
#
# ConfusionMatrixDisplay.from_predictions(yTest2, LDA_pred)
# plt.show()
# print("Precision score",precision_score(yTest2,LDA_pred))
# print("auc score",roc_auc_score(yTest2,y_pred))
