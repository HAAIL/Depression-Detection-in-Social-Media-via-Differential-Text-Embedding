from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.decomposition import PCA

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
    return aucs


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
    return precision, recall, y_real, y_proba


embed_df = pd.read_csv('final1-embed-co.csv')
y = embed_df['Depressed'].astype('int')

# embed_df = embed_df.fillna(0)
# embed_df.to_csv('final1-embed-co-filled.csv')

dfm = pd.read_csv('final1-embed-co-filled.csv', delimiter=',')
print(len(dfm))
print(len(y))
df_headersName=pd.read_csv('final1-embed-co-filled.csv', nrows=1).columns.tolist()
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
'Inability to feel',
'Feeling needed']

df_total = pd.DataFrame()

for d in range(0, 5):
    # new_f = df_headersName[d]
    if d == 0 or d == 3 :
        new_f = df_headersName[(d * 768) + 2: (d * 768 + 768) + 2]
        print(new_f)

        trainX = dfm[new_f]
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

df = pd.DataFrame(pc[:, :pcindex])
print(df.shape)
df_total = df.to_numpy()



kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
model = LogisticRegression(penalty = 'l2', max_iter=10000000)
#
#
#
#
#
#
#
#
# # Cosine Distance
# df_total = pd.read_csv('final1.csv')
# df_total= df_total.dropna()
# df_total.index = range(len(df_total))
#
# y2 = df_total['Depressed'].astype('int')
# df_total = df_total.iloc[:,1:8]
# df_total = df_total.to_numpy()
#
#Bert
bert = pd.read_csv('final-set1-bert.csv')
bert = bert.iloc[:,:-1]
x = bert.to_numpy()
#
#
# df_total = pd.read_csv('dys-emb-mean1.csv')
# df_total = df_total.to_numpy()
set_mean_all = draw_cv_roc_curve(model, kfold,df_total, y, title='Cross Validated ROC of using the Mean of Depression Symptoms Embeddings 0-3 - Dataset1')
print(set_mean_all)
p1, r1, yy1,pr1  = draw_cv_pr_curve(model, kfold, df_total, y, title='Cross Validated PR Curve Of using the Mean of Depression Symptoms Embeddings 0-3 - Dataset1')

# set_mean_all = draw_cv_roc_curve(model, kfold,x, y, title='Cross Validated ROC Of using SentBert - Dataset1')
# print(set_mean_all)
# p2, r2, y2,pr2  = draw_cv_pr_curve(model, kfold, x, y, title='Cross Validated PR Curve Of using SentBert - Dataset1')
