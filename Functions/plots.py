#---------------------------------------------
# Distribution histograms for features
#--------------------------------------------

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pal = sns.color_palette("RdPu")
plt. figure()
bins  = 10
data                 = pd.read_csv('final_set.csv')


def generate_his(feature_name: str, data: pd.DataFrame):
    data[feature_name +'_binned'] = pd.qcut(data[feature_name], bins , duplicates='drop')
    chart = pd.crosstab(data[feature_name +'_binned'],data['Depressed'])
    chart.div(chart.sum(1).astype(float), axis=0).plot(kind='bar', color = ['#D8A7B1','#A49393'],stacked=True, alpha=0.7)
    plt.xlabel('Bins', fontsize=14)
    plt.ylabel('Quantity', fontsize=14)
    plt.suptitle(feature_name + ' category distribution',fontsize=14)
    plt.savefig(feature_name + '.png', bbox_inches='tight')
#end def 

#ex. get the histogram of "Mind reading" cateogry
generate_his('Mind reading', data)


#---------------------------------------------
# Distribution for all features
#--------------------------------------------

selected_columns = ["Mind reading", "Labelling", "Fortune telling", "Overgeneralising", "Emotional Reasoning" , "Personalising", "Shoulds and Musts"]

np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({'font.size': 22})

fig = plt.figure(figsize = (20, 23))
j = 0
for i in selected_columns:
    plt.subplot(6, 4, j+1)
    j += 1
    sns.distplot(data[i][data['Depressed']==0], color='#D8A7B1', label = 'no symptoms' )
    sns.distplot(data[i][data['Depressed']==1], color='#A49393', label = 'depression symptoms' )
    plt.xlabel("Cosine Distance")
    plt.ylabel("Quantity")
    plt.title(i)
    plt.legend(loc='best', fontsize=12)
    plt.savefig(i)


fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.savefig('Dist.png', bbox_inches='tight')
plt.show()
