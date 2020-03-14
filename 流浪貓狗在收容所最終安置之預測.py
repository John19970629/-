


import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"



train_df=pd.read_csv('train.csv')


# # Classify Name



train_df.iloc[:]['Name'] = train_df.iloc[:]['Name'].fillna(0)
for i in range(26729):
    if (train_df.iloc[i][1]!=0): 
        train_df.iloc[i][1]=1    #有名字是1


# # process SexuponOutcome


train_df.iloc[:]['SexuponOutcome'] = train_df.iloc[:]['SexuponOutcome'].fillna(0)
gender=['']*26729
fertility=['']*26729

for i in range(26729):  #新增Gender欄位
    if (train_df.iloc[i]['SexuponOutcome']==0):
        gender[i]='Gender_Unknown'
    elif (train_df.iloc[i]['SexuponOutcome'][-4]=='M'):
        gender[i]='Male'
    elif (train_df.iloc[i]['SexuponOutcome'][-4]=='m'):
        gender[i]='Female'
    else :
        gender[i]='Gender_Unknown'
train_df['Gender']=gender

for i in range(26729):  #新增Fertility欄位
    if (train_df.iloc[i]['SexuponOutcome']==0):
        fertility[i]='Fertility_Unknown'
    elif (train_df.iloc[i]['SexuponOutcome'][0]=='I'):
        fertility[i]='Intact'
    elif (train_df.iloc[i]['SexuponOutcome'][0]=='S' or train_df.iloc[i]['SexuponOutcome'][0]=='N'):
        fertility[i]='Spayed'
    else :
        fertility[i]='Fertility_Unknown'
train_df['Fertility']=fertility

train_df=train_df.drop(columns='SexuponOutcome')  #去掉SexuponOutcome欄

#Onehot
gender = pd.get_dummies(train_df['Gender'])
train_df = pd.concat([train_df,gender],axis=1)
train_df=train_df.drop(columns='Gender')

fertility = pd.get_dummies(train_df['Fertility'])
train_df = pd.concat([train_df,fertility],axis=1)
train_df=train_df.drop(columns='Fertility')


# # Dummy year and month

def dummyData(data_df):
    data_df['DateTime']=pd.to_datetime(train_df.DateTime)
    data_df['Year']=train_df.DateTime.dt.year
    data_df['Month']=train_df.DateTime.dt.month
    data_df=pd.get_dummies(data_df,columns=['Year','Month'],drop_first=True)
    return data_df



train_df=dummyData(train_df)


# # Transfer Year ,Week, Month to Date

train_df.AgeuponOutcome.fillna('0 week',inplace=True)
splitAgeuponOutcomeTrain=train_df.AgeuponOutcome.str.split(' ')

# splitAgeuponOutcome_cols=['num','YWDtype']
# splitAgeuponOutcome.columns=splitAgeuponOutcome_cols
# splitAgeuponOutcome['YWDtype'].unique()

def transferToDate(arr):
    result=[]
    for item in arr:
        if(item==None):
            result.apeend(np.nan)
        elif item[1]=='years' or item[1]=='year':
            result.append(int(item[0])*365)
        elif item[1]=='month' or item[1]=='months':
            result.append(int(item[0])*30)
        elif item[1]=='weeks' or item[1]=='week':
            result.append(int(item[0])*7)
        else:
            result.append(int(item[0])*1)          
    return result



date_df=pd.DataFrame(transferToDate(splitAgeuponOutcomeTrain),columns=['AgeuponOutcomeDate']) 
train_df=pd.concat([train_df,date_df],axis=1)


# # Classify Color

a = train_df.groupby('Color').size()
color1 = []
for i in range(len(a)):
    if(a[i]<100):
        color1.append(i)

for i in range(len(train_df['Color'])):
    for j in range(len(a.index)):
        if(train_df['Color'][i] == a.index[j]):
            for k in range(len(color1)):                
                if(color1[k] == j):
                    train_df['Color'][i] = 'other'

color = pd.get_dummies(train_df['Color'])
train_df = pd.concat([train_df,color],axis=1)


#labelencoder = LabelEncoder()
#train_df['Color_encode']=labelencoder.fit_transform(train_df['Color'])


# # Classify Breed

b = train_df.groupby('Breed').size()

for i in range(len(train_df['Breed'])):
    if(train_df['Breed'][i][-3:] == 'Mix'):
        train_df['Breed'][i] = train_df['Breed'][i][:-4]
    else:
        for j in range(len(train_df['Breed'][i])):
            if(train_df['Breed'][i][j] == '/'):
                train_df['Breed'][i] = train_df['Breed'][i][:j]
                break

breed = []         
c = train_df.groupby('Breed').size()
            
for i in range(len(c)):
    
    if(c[i] < 200):
        breed.append(i)
        
for i in range(len(train_df['Breed'])):
    for j in range(len(c.index)):
        if(train_df['Breed'][i] == c.index[j]):
            for k in range(len(breed)):
                if(breed[k] == j):
                    if(train_df['AnimalType'][i] == 'Dog'):
                        train_df['Breed'][i] = 'dog_others'
                    else:
                        train_df['Breed'][i] = 'cat_others'

for i in range(len(train_df['Breed'])):
    if(train_df['Breed'][i] == 'Domestic Shorthair'):
        train_df['Breed'][i] = train_df['Color'][i] + ' Domestic Shorthair'


breed = pd.get_dummies(train_df['Breed'])
train_df = pd.concat([train_df,breed],axis=1)
train_df=train_df.drop(columns='Breed')
train_df=train_df.drop(columns='Color')
#labelencoder = LabelEncoder()
#train_df['Breed_encode']=labelencoder.fit_transform(train_df['Breed'])

train_df=train_df.drop(columns=['AnimalID','DateTime','OutcomeSubtype','AgeuponOutcome'])

Animal=['']*26729
for i in range(26729):
    if (train_df.iloc[i]['AnimalType']=='Dog'):
        Animal[i]='1'
    else:
        Animal[i]='0'
train_df['AnimalType']=Animal

from sklearn import preprocessing
train_df['AgeuponOutcomeDate']=preprocessing.scale(train_df['AgeuponOutcomeDate'])


# # Split Data
from sklearn.model_selection import train_test_split

label = train_df['OutcomeType']
train_df = train_df.drop(columns='OutcomeType')
X_train, X_test, y_train, y_test = train_test_split(train_df, label, test_size=0.2, random_state=0)


#train=train_df.sample(frac=0.75,random_state=99)
#test=train_df.loc[~train_df.index.isin(train.index),:]


# # Build Model



#train_df.sample(5)
#
#
#
#train_df.columns
#
#
#cols=['Name','Male','Female','Gender_Unknown','Fertility_Unknown','Intact','Spayed',
#     'Year_2014', 'Year_2015', 'Year_2016', 'Month_2', 'Month_3', 'Month_4',
#       'Month_5', 'Month_6', 'Month_7', 'Month_8', 'Month_9', 'Month_10',
#       'Month_11', 'Month_12','AgeuponOutcomeDate','Breed_encode', 'Color_encode']
#label=['OutcomeType']


from sklearn.ensemble import RandomForestClassifier
# X_train, X_test,y_train, y_test = train_test_split(feature欄位, label欄位,
#                                                 random_state=0)
model = RandomForestClassifier(n_estimators=2000,oob_score = True,n_jobs = -1,max_features=0.2,min_samples_leaf=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)




from sklearn.metrics import classification_report
result=classification_report(y_test,y_pred)
print(result)


#決策樹
from sklearn import tree
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=3)
clf = clf.fit(X_train,y_train)

 #視覺化決策樹
dot_data=tree.export_graphviz(
        clf,
        out_file=None,
        feature_names=train_df.columns[:],
        class_names=label,
        filled=True,
        impurity=True,
        rounded=True
    )
import pydotplus#該包為專門繪製DOT資料的視覺化包
graph=pydotplus.graph_from_dot_data(dot_data)#以DOT資料進行graph繪製
#graph.get_nodes()[7].set_fillcolor("#FFF2DD")#設定顯示顏色
from IPython.display import Image
img=Image(graph.create_png())#將graph影象顯示出來
display(img)
test_y_predicted = clf.predict(X_test)
print(classification_report(y_test,test_y_predicted))