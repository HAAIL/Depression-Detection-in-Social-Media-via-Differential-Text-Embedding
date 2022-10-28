import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Loading pre-defined Boston Dataset
dfm = pd.read_csv('final3-embed-co-filled.csv', delimiter=',')
y = dfm['Depressed'].astype('int')

print(len(dfm))
print(len(y))
df_headersName=pd.read_csv('final3-embed-co-filled.csv', nrows=1).columns.tolist()
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
#
df_total = pd.DataFrame()

for d in range(0, 14):
    # new_f = df_headersName[d]
    # if d == 0 or d == 13:
    new_f = df_headersName[(d * 768) + 2: (d * 768 + 768) + 2]
    print(new_f)

    trainX = dfm[new_f]
    df_pcas = pd.concat([df_total, trainX], axis=1)
    df_total = df_pcas

print(df_total.shape)
# df_total = df_total.to_numpy()
# df_total = pd.read_csv('final3.csv')
# df_total = df_total.iloc[:,1:-2]

x_train, x_test, y_train, y_test = train_test_split(
    df_total, y,
    test_size=0.25)

print("Train data shape of X = % s and Y = % s : " % (
    x_train.shape, y_train.shape))

print("Test data shape of X = % s and Y = % s : " % (
    x_test.shape, y_test.shape))


# import model
from sklearn.linear_model import ElasticNet

# Train the model
e_net = ElasticNet(alpha=0.0009)
e_net.fit(x_train, y_train)
print(e_net.coef_)
# calculate the prediction and mean square error
y_pred_elastic = e_net.predict(x_test)
mean_squared_error = np.mean((y_pred_elastic - y_test) ** 2)
print("Mean Squared Error on test set", mean_squared_error)

e_net_coeff = pd.DataFrame()
e_net_coeff["Columns"] = x_train.columns
e_net_coeff['Coefficient Estimate'] = pd.Series(e_net.coef_)
for i in range(1,15):
    print(sum(e_net_coeff.iloc[(i-1)*768:i*768,1]))
