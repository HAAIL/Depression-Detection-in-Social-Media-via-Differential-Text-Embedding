

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ***** Reading the csv file  *****
dfm = pd.read_csv('experiment8-final-set-dif-emb-allmore-filled.csv', delimiter=',')
dfm.dropna(inplace=True)
print(len(dfm))
dfm.index = range(len(dfm))
# ***** Reading headers *****
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

opsubset = []
best_attr = []
subset = []
print("Running ...")
# ***** loop to incrementally add features *****
oldPerf = 0
# df = pd.DataFrame()
number_attr = len(df_headersName)-2
print(number_attr/768)
for i in range(0,len(df_attrName)):
    print('first')
    print(i)
    if len(opsubset) == 0:
        best_x = []
        performance = []

        fs = df_headersName[(i * 768)+1: (i * 768 + 768)+1]
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
            print(c)
            if c == 0.90 or c > 0.9:
                print(c)
                pcindex = n
                break
            n = n + 1
        print(pcindex)
        # print(dfm[best_x])
        # print(pc[:,:pcindex])
        # print(dfm[best_x])
        df = pd.DataFrame(pc[:,:pcindex])
        testX = dfm[targetName]
        for iterations in range(0, totalFolds):
            fiveFoldPerf(performance, df, testX)
        performance = np.array(performance)
        newPerf = performance.mean()
        print(newPerf)
        # print(newPerf)
        if (newPerf> oldPerf):
            best_f = fs
            print(best_f)
            # print(best_f)
            oldPerf = newPerf


        if i == len(df_attrName)-1:
            op = oldPerf
            print(op)
            for f in best_f:
                sfsFeaturesSelected.append(f)
            print(sfsFeaturesSelected)

        if len(sfsFeaturesSelected) > 0:
            for f in best_f:
                opsubset.append(f)
            print(opsubset)
            best_attr.append(df_attrName[i])
            print(best_attr)


        # op = oldPerf
        # overall = 0
        # if len(opsubset) > 0:
        #     while (True):
        #
        #         print('second')
        #         for d in range(0,len(df_attrName)-1):
        #             if i != d:
        #                 performance = []
        #                 # new_f = df_headersName[d]
        #                 new_f = df_headersName[(i * 768)+1: (i * 768 + 768)+1]
        #                 print(len(new_f))
        #
        #
        #                 if new_f not in sfsFeaturesSelected and df_attrName[d] not in best_attr:
        #                     for f in new_f:
        #                         sfsFeaturesSelected.append(f)
        #
        #                     trainX = dfm[sfsFeaturesSelected]
        #                     print(dfm[sfsFeaturesSelected])
        #                     pca = PCA()
        #                     pc = pca.fit_transform(trainX)
        #                     cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        #                     cm = list(cumulative_variance)
        #                     pcindex = 0
        #                     n = 0
        #                     for c in cm:
        #                       c = round(c, 2)
        #                       print(c)
        #                       if c == 0.90:
        #                           print(c)
        #                           pcindex = n
        #                           break
        #                       n = n + 1
        #                         print(pcindex)
        #                         print(pc[:, :pcindex])
        #                     df2 = pd.DataFrame(pc[:, :pcindex])
        #                     df_pcas = pd.concat([df, df2], axis=1)
        #                     testX = dfm[targetName]
        #                     # ***** Running 5 fold S-CV *****
        #                     for iterations in range(0, totalFolds):
        #                         fiveFoldPerf(performance, trainX, testX)
        #                     performance = np.array(performance)
        #                     newp = performance.mean()
        #                     print(newp)
        #                     if (newp > op):
        #                         # ***** Adding feature *****
        #                         best_f = new_f
        #                         overall = newp
        #                         for f in new_f:
        #                             sfsFeaturesSelected.remove(f)
        #                         df_pcas.remove(df2, inplace = True)
        #
        #                     else:
        #                         for f in new_f:
        #                             sfsFeaturesSelected.remove(f)
        #                         df_pcas.remove(df2, inplace = True)

        #         if overall > op:
        #             if best_f not in opsubset:
        #                 for f in best_f:
        #                     opsubset.append(f)
        #                     sfsFeaturesSelected.append(f)
        #                         # df_headersName.remove(best_f)
        #                 print(opsubset)
        #                 best_attr.append(df_attrName[d])
        #                 df_pcas = pd.concat([df, df2], axis=1)
        #                 df = df_pcas

        #                 op = overall
        #         else:
        #             break
#                 subset.append(df.values)


# ***** PRINTING IMPORTANT FEATURES ON CONSOLE *****
def displayResults(selectedAttrb, acc):
    print("Selected Features Using SFS:")
    for featureIndx in range(len(selectedAttrb)):
            print(str(featureIndx+1)+". "+selectedAttrb[featureIndx])
    print('Accuracy using this feature set: ', acc,'%')
displayResults(best_attr, op)