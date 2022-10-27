

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,accuracy_score, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from numpy import interp
import pandas as pd
from sklearn.decomposition import PCA
# ***** Reading the csv file  *****
final = pd.read_csv('final1.csv')
y = final['Depressed'].astype('int')
dfm = pd.read_csv('final1-embed-co-filled.csv', delimiter=',')
print(len(dfm))
# ***** Reading headers *****
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
print(len(df_headersName))
# ***** class variable *****
targetName= 'Depressed'
oldPerf= 0
# ***** classifier object *****
forest = LogisticRegression(penalty='l2',  max_iter=10000000, solver='saga')
fclss=[]
new_fclss=[]
# ***** Setting Variables for Sequencial feature selection (SFS) *****
sfsFeaturesSelected=[]
newPerf = oldPerf + 1
totalFolds=5

# ***** Function for making 5 Stratified fold with same proportion of class variables *****
def runSCV(trainX,testX):
    # ***** Initializing local Variables *****
    allFoldX={"0": [], "1": [], "2": [], "3": [], "4": [] }
    allFoldY={"0": [], "1": [], "2": [], "3": [], "4": [] }
    ftable=testX.value_counts()
    for sn in (range((len(testX.unique())))):#loop for differet possible class variable values
        # ***** Generating frequency table for each class variable *****
        featureVal= ftable.index.values
        # ***** Finding the indices of the unique class variable values *****
    for featureValIndx in range(len(featureVal)):
        current_class= featureVal[featureValIndx]
        inx= testX.index[testX==current_class].tolist()
        fclss=[{0 : current_class, 1 : inx}]
        new_fclss.append(fclss)
    foldX_All=dict.fromkeys(["0", "1", "2", "3", "4"])

    # ***** making proportional splits *****
    for i in range(len(featureVal)):
        inx_append=new_fclss[i][0][1]
        for item in range(len(inx_append)):
            X_list= trainX.iloc[inx_append[item]]
            Y_list= testX.iloc[inx_append[item]]
            allFoldX[str(item%totalFolds)].append(X_list)
            allFoldY[str(item%totalFolds)].append(Y_list)
            valX=allFoldX[str(item%5)]
            foldX_All[str(item%totalFolds)]=pd.concat([valX[i] for i in range(len(valX))],axis=1).T
    # ***** creating combinations for 5 fols *****
    foldsAllFive=[foldX_All["0"], foldX_All["1"], foldX_All["2"], foldX_All["3"]]
    fold_Xtrain= pd.concat(foldsAllFive)
    fold_Xtest=np.concatenate((allFoldY["0"], allFoldY["1"], allFoldY["2"], allFoldY["3"]), axis=0)
# ***** Returning Splits for training & testing *****
    return fold_Xtrain, fold_Xtest, foldX_All["4"], allFoldY["4"]

def fiveFoldPerf(performance,trainX,testX):
    dx, lx, dy, ly= runSCV(trainX,testX)
    forest.fit(dx, lx)
    prediction = forest.predict(dy)
    #calculating Mean Accuracy
    roc = roc_auc_score(np.array(ly), forest.predict_proba(dy)[:, 1])
    newPerf_temp= 100-(np.square(np.subtract(np.array(ly), prediction)).mean())
    performance.append(roc)
    return performance










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
    # plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()
    return aucs


opsubset = []
best_attr = []
subset = []
print("Running ...")
# ***** loop to incrementally add features *****
oldPerf = 0
# df = pd.DataFrame()
number_attr = len(df_headersName)-2
print(number_attr/768)
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
model = LogisticRegression(penalty = 'l2', max_iter=10000000)

for i in range(0,len(df_attrName)):
    print('first')
    print(i)
    if len(opsubset) == 0 and i == 0:
        best_x = []
        performance = []

        fs = df_headersName[(i * 768)+2: (i * 768 + 768)+2]
        # print(fs)
        for f in fs:
            best_x.append(f)
        # f = df_headersName[i]
        # best_x.append(f)

        trainX = dfm[best_x]
        pca = PCA()
        pc = pca.fit_transform(trainX)
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
        # print(pcindex)
        # print(dfm[best_x])
        # print(pc[:,:pcindex])
        # print(dfm[best_x])
        df = pd.DataFrame(pc[:,:pcindex])
        # df_original = pd.DataFrame(trainX)

        testX = dfm[targetName]
        set_mean_all = draw_cv_roc_curve(model, kfold, df, testX,
                                         title='Cross Validated ROC of using the Mean of Depression Symptoms Embeddings 0-3 - Dataset1')
        # print(set_mean_all)
        performance = np.array(set_mean_all)
        print(newPerf)
        # print(newPerf)
        if (newPerf> oldPerf):
            best_f = fs
            print(best_f)
            # print(best_f)
            oldPerf = newPerf
            best= df_attrName[i]
            df_b = df
            original = dfm[best_x]

        if i == 0:
            op = oldPerf
            print(op)
            for f in best_f:
                opsubset.append(f)
                df_headersName.remove(f)
            print(opsubset)
            best_attr.append(best)
            df_attrName.remove(best)
            print(df_attrName)
            print(best_attr)
            df = df_b
            # subset.append(df.values)
            # for f in best_f:
                # sfsFeaturesSelected.append(f)
            # print(sfsFeaturesSelected)
            # df = df_b





    overall = oldPerf
    if len(opsubset) > 0:
        while (True):
            print('second')
            op = 0
            print('if')
            print(len(df_attrName))
            for d in range(0,len(df_attrName)):
                performance = []
                # new_f = df_headersName[d]
                new_f = df_headersName[(d * 768)+2: (d * 768 + 768)+2]
                print(new_f)


                if df_attrName[d] not in best_attr:
                    for f in new_f:
                        sfsFeaturesSelected.append(f)

                    trainX = dfm[sfsFeaturesSelected]
                    # print(dfm[sfsFeaturesSelected])
                    # df_or = pd.concat([df_original, trainX], axis=1)
                    # print(df_or.shape)
                    df_pcas = pd.concat([original, trainX], axis=1)

                    pca = PCA()
                    pc = pca.fit_transform(df_pcas)
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

                    print(pcindex)
                    # print(pc[:, :pcindex])
                    df2 = pd.DataFrame(pc[:, :pcindex])
                    # print(df2.shape)
                    # df_pcas = pd.concat([df, df2], axis=1)
                    # print(df_pcas.shape)

                    testX = dfm[targetName]
                    # ***** Running 5 fold S-CV *****
                    # for iterations in range(0, totalFolds):
                    #     fiveFoldPerf(performance, df2, testX)

                    set_mean_all = draw_cv_roc_curve(model, kfold, df2, testX,
                                                     title='Cross Validated ROC of using the Mean of Depression Symptoms Embeddings 0-3 - Dataset1')
                    # print(set_mean_all)
                    performance = np.array(set_mean_all)
                    newp = performance.mean()
                    print(newp)
                    if (newp > op):
                        # ***** Adding feature *****
                        best_f = new_f
                        # overall = newp
                        for f in new_f:
                            sfsFeaturesSelected.remove(f)
                        df_pcas.drop(trainX, axis = 1, inplace = True)
                        print('drop')
                        print(df_pcas.shape)
                        op = newp
                        best = df_attrName[d]
                        # df_best = pd.DataFrame(trainX)
                        df_b = trainX


                    else:
                        for f in new_f:
                            sfsFeaturesSelected.remove(f)
                        df_pcas.drop(trainX,axis = 1, inplace = True)
                    # break

            if op > overall:
                if best not in best_attr:
                    for f in best_f:
                        opsubset.append(f)
                        # sfsFeaturesSelected.append(f)
                        df_headersName.remove(f)
                    print(opsubset)
                    best_attr.append(best)
                    print(best_attr)
                    df_attrName.remove(best)
                    df_pcas = pd.concat([original, df_b], axis=1)
                    original = df_pcas
                    print(original.shape)

                    overall = op
            else:
                break


            # else:
            #     for d in range(0, len(df_attrName) - 1):
            #
            #         performance = []
            #         # new_f = df_headersName[d]
            #         new_f = df_headersName[(d * 768) + 1: (d * 768 + 768) + 1]
            #         print(new_f)
            #
            #         if df_attrName[d] not in best_attr:
            #             for f in new_f:
            #                 sfsFeaturesSelected.append(f)
            #
            #             trainX = dfm[sfsFeaturesSelected]
            #             print(sfsFeaturesSelected)
            #             df_or = pd.concat([df_original, trainX], axis=1)
            #
            #             pca = PCA()
            #             pc = pca.fit_transform(df_or)
            #             cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            #             cm = list(cumulative_variance)
            #             pcindex = 0
            #             n = 0
            #             for c in cm:
            #                 c = round(c, 2)
            #                 # print(c)
            #                 if c == 0.90 or c > 0.9:
            #                     # print(c)
            #                     pcindex = n
            #                     break
            #                 n = n + 1
            #             # print(pcindex)
            #             # print(pc[:, :pcindex])
            #             df2 = pd.DataFrame(pc[:, :pcindex])
            #             print(df2.shape)
            #             # df_pcas = pd.concat([df, df2], axis=1)
            #             # print(df_pcas.shape)
            #
            #             testX = dfm[targetName]
            #             # ***** Running 5 fold S-CV *****
            #             for iterations in range(0, totalFolds):
            #                 fiveFoldPerf(performance, df2, testX)
            #             performance = np.array(performance)
            #             newp = performance.mean()
            #             print(newp)
            #             if (newp > op):
            #                 # ***** Adding feature *****
            #                 best_f = new_f
            #                 # overall = newp
            #                 for f in new_f:
            #                     sfsFeaturesSelected.remove(f)
            #                 df_or.drop(trainX, inplace=True, axis =1)
            #                 print('drop')
            #                 print(df_or.shape)
            #                 op = newp
            #                 best = df_attrName[d]
            #                 df_best = pd.DataFrame(trainX)
            #
            #
            #
            #             else:
            #                 for f in new_f:
            #                     sfsFeaturesSelected.remove(f)
            #                 df_or.drop(trainX, inplace=True, axis =1)
            #
            #     if op > overall:
            #         if best_f not in opsubset:
            #             for f in best_f:
            #                 opsubset.append(f)
            #                 # sfsFeaturesSelected.append(f)
            #                 df_headersName.remove(f)
            #             print(opsubset)
            #             best_attr.append(best)
            #             print(best_attr)
            #             df_attrName.remove(best)
            #             df_or = pd.concat([df_original, df_best], axis=1)
            #             df_original = df_or
            #             print(best_attr)
            #
            #             overall = op
            #     else:
            #         break
        subset.append(df.values)

    break

np.savetxt("best_sbuset.csv",
           subset,
           delimiter =", ",
           fmt ='% s')

# ***** PRINTING IMPORTANT FEATURES ON CONSOLE *****
def displayResults(selectedAttrb, acc):
    print("Selected Features Using SFS:")
    for featureIndx in range(len(selectedAttrb)):
            print(str(featureIndx+1)+". "+selectedAttrb[featureIndx])
    print('Accuracy using this feature set: ', acc,'%')
displayResults(best_attr, overall)