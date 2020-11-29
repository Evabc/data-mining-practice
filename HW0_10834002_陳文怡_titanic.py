
# coding: utf-8

# #### 該競賽提供的train dataset其各欄位名稱及定義如下，經由這些欄位，我們可考慮是否作為feature來納入分析。
# #### 例如，PassengerId（乘客編號）欄位對於預測結果應無影響可忽略不看
# #### survival欄位是我們要針對Test dataset預測的是否倖存答案，因此它應該是Lable。
# #### 其餘欄位則需經由分析來判斷是否跟survival有直接或間接關係，再決定是否列為特徵。
# 
# * PassengerId 乘客ID編號
# * survival 是否倖存 (0 = No, 1 = Yes)
# * pclass 船票等級 (1 = 1st, 2 = 2nd, 3 = 3rd)
# * sex 性別
# * Age 年齡
# * sibsp 在船上同為兄弟姐妹或配偶的數目
# * parch 在船上同為家族的父母及小孩的數目
# * ticket 船票編號
# * fare 船票價格
# * cabin 船艙號碼
# * embarked 登船的口岸 (C = Cherbourg, Q = Queenstown, S = Southampton)

# #### 針對Kaggle的Titanic倖存預測競賽，將分為下列三個階段來進行，先開始第一階段。
# 
# - [x] 資料分析Data analysis
#    * 資料形態、架構的掌握。
#    * 資料發現Data exploration。
#    * 資料的相關及變異。
# 

# In[169]:


#import pakeage
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
import seaborn as sns


# In[170]:


#read data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[171]:


#check info
print(train.shape)
print(test.shape)
train.info()


# In[172]:


#combine train&test
x_train = train.append(test)
print(x_train.shape)


# In[173]:


#check all columns
print(x_train.columns.values)
x_train.head(5)


# In[174]:


# have some null in train data and test data
print('Train columns with null values:\n', x_train.isnull().sum())


# In[175]:


#count object type 
x_train.describe(include=['O'])
# name_count have 1309, but unique 1307. It mean we have two same name


# `1. 性別與倖存的關係` 
# ##### 所以我們可以先從這些object中去察看是否跟生存率有關連

# In[176]:


#first, caiculate the ratio of male and female by age group
figure = plt.figure(figsize=(15,8))
plt.hist([x_train[x_train['Sex']=='male']['Age'], 
          x_train[x_train['Sex']=='female']['Age']], 
          stacked=False, 
          color = ['g','r'], 
          bins = 30,
          label = ['Male','Female'])

plt.xlabel('Age')
plt.ylabel('Number of Sex')

plt.legend()


# In[177]:


x_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#female Survival rate more than male


# In[178]:


survived_sex = x_train[x_train['Survived']==1]['Sex'].value_counts()
dead_sex = x_train[x_train['Survived']==0]['Sex'].value_counts()
df = pd.DataFrame([survived_sex,dead_sex])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(15,8))
#Use 'Sex' and 'Survived' Graphy to display


# `2. 名稱與倖存的關係` 
# ##### 我們檢視一下Name欄位，會發現該欄位除了乘客姓名之外，還帶有稱呼及職稱

# In[179]:


x_train['Title1'] = x_train['Name'].str.split(", ", expand=True)[1]
x_train['Name'].str.split(", ", expand=True).head(5)


# In[180]:


x_train['Title1'].head(5)


# In[181]:


x_train['Title1'] = x_train['Title1'].str.split(". ", expand=True)[0]
# split "."


# In[182]:


x_train['Title1'].head(5)
x_train['Title1'].unique()
#find all title


# In[183]:


# Actually, we don't need all title (some title only few people) so we cleanup rare title names
stat_min = 10 
#while small is arbitrary, we'll use the common minimum in statistics
title_names = (x_train['Title1'].value_counts() < stat_min) 
#this will create a true false series with title name as index

x_train['Title2'] = x_train['Title1'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)
print(x_train['Title2'].value_counts())


# In[184]:


x_train.groupby(['Title2','Pclass'])['Age'].mean()


# In[187]:


survived_num = x_train[x_train['Survived']==1]['Title2'].value_counts()
print('-'*5 + 'survived_num' + '-'*5  )
print(survived_num)

died_num = x_train[x_train['Survived']==0]['Title2'].value_counts()
print('-'*5 + 'died_num' + '-'*5  )
print(died_num)

survived_ratio = (100*survived_num/(survived_num+died_num))
print('-'*5 + 'survived_ratio' + '-'*5 )
print(survived_ratio)


# `3. 船票等級與存活的關係` 
# ##### Pclass欄位指的是船票等級。
# ##### 以下看出Pclass的等級愈高（1>2>3）則存活機率愈大，很多死亡的人都落在class3，
# ##### 可以大膽的假設，有錢人比較有機會先搭上救生艇。

# In[188]:


survived_pclass = x_train[x_train['Survived']==1]['Pclass'].value_counts()
dead_pclass = x_train[x_train['Survived']==0]['Pclass'].value_counts()
df = pd.DataFrame([survived_pclass,dead_pclass])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=False, figsize=(15,8))

x_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# the number of different class which survived


# ###### 我們知道船上女性的倖存率很高，男性非常低，但這現象在不同等級船票之間會有差嗎？
# ###### 從下方的結果，持有P1, P2等級船票的女性有高達九成的倖存率，但持有等級最低的P3船票女性只有一半的存活率。
# ###### 至於男性方面，持有最高等級船票P1的存活率是另外二個等級的2倍。

# In[189]:


x_train[['Pclass', 'Sex', 'Survived']].groupby(['Pclass', 'Sex'], 
                                               as_index=False).mean().sort_values(by='Survived', ascending=False)

#the survived ratio of different class, sex


# `4. 不同年齡層與倖存的關係` 
# ###### 從下圖可看出年齡愈偏向兩極（較年長或較年幼）則存活率愈高，其中尤以年齡愈小愈明顯。

# In[190]:


figure = plt.figure(figsize=(15,8))
plt.hist([x_train[x_train['Survived']==1]['Age'], 
          x_train[x_train['Survived']==0]['Age']], 
         stacked=True, 
         color = ['g','r'],
         bins = 30,
         label = ['Survived','Dead'])

plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()


# `5. 不同票價與存活的關係` 
# ###### 經由下方統計結果發現，與船票等級類似，票價愈高則存活率愈大。

# In[191]:


figure = plt.figure(figsize=(15,8))
plt.hist([x_train[x_train['Survived']==1]['Fare'],
          x_train[x_train['Survived']==0]['Fare']], 
          stacked=True, 
          color = ['g','r'],
          bins = 30,
          label = ['Survived','Dead'])

plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend()


# `5. 親屬人數對於存活率的影響` 
# ###### SibSp與Parch這兩個欄位指的是旁系與直系親屬人數，親屬或家人一起可以互相幫助，對於存活率有一定的影響。
# ###### 理論上直系的Parch對於倖存率影響力會比旁系SibSp更大一些。
# ###### 不過先不考慮兩者的差異，而是將SibSp與Parch這兩個欄位相加作為親屬人數來與存活率比較。
# ###### 經由下方的統計，可發現親屬人數為3人時有最高的存活率，其次為2人、１人以及6人：

# In[193]:


x_train['Family'] = x_train['SibSp'] + x_train['Parch']

x_train[['Family', 'Survived']].groupby(['Family'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# `6. 上岸港口與存活率的關係` 
# ###### 如果從Cherbourg上船，那存活率有5成5
# ###### 如果從Queenstown上船那存活率降到3成8
# ###### 從Southampton存活率只有3成3了
# ###### 我們可以猜想，可能跟地區的組成人口,人口所得, 上船目的有關連

# In[194]:


#complete  missing values
x_train['Embarked'] = x_train['Embarked'].fillna('S')

x_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)


total_Embarked_S = x_train[x_train['Embarked']=='S']['PassengerId'].count()
total_Embarked_C = x_train[x_train['Embarked']=='C']['PassengerId'].count()
total_Embarked_Q = x_train[x_train['Embarked']=='Q']['PassengerId'].count()

Embarked_S = x_train[x_train['Embarked']=='S']['Survived'].value_counts() / total_Embarked_S
Embarked_C = x_train[x_train['Embarked']=='C']['Survived'].value_counts() / total_Embarked_C
Embarked_Q = x_train[x_train['Embarked']=='Q']['Survived'].value_counts() / total_Embarked_Q  

             

df = pd.DataFrame([Embarked_S,Embarked_C,Embarked_Q])
df.index = ['Southampton','Cherbourg','Queenstown']
df.plot(kind='bar',stacked=False, figsize=(15,8))


# ###### 我們可從下方統計看出不同港口乘客有不同的經濟狀況。
# ###### 從法國Cherbourg上船的目的可能為出訪旅遊，大多購買等級最高的P1船票（故Cherbourg上船的有較高的存活率）
# ###### 來自Queenstown的乘客則有極大的比例是購買最低等級的P3船票，目的可能為工作或移民。

# In[195]:


total_Pclass_S = x_train[x_train['Embarked']=='S']['Pclass'].count()
total_Pclass_C = x_train[x_train['Embarked']=='C']['Pclass'].count()
total_Pclass_Q = x_train[x_train['Embarked']=='Q']['Pclass'].count()

                       
Embarked_S = x_train[x_train['Embarked']=='S']['Pclass'].value_counts() / total_Pclass_S
Embarked_C = x_train[x_train['Embarked']=='C']['Pclass'].value_counts() / total_Pclass_C
Embarked_Q = x_train[x_train['Embarked']=='Q']['Pclass'].value_counts() / total_Pclass_Q  

             

df = pd.DataFrame([Embarked_S,Embarked_C,Embarked_Q])
df.index = ['Southampton','Cherbourg','Queenstown']
df.plot(kind='bar',stacked=False, figsize=(15,8))


# `7. 艙房號碼與存活的關係` 
# ###### Cabin欄位存放的是艙房號碼，此欄位的有極大比例是空值，僅僅只有295筆記錄有值
# ###### 發現遺失Cabin艙房編號的，有極大比例是屬於P3最低等級船票的乘客，其次為P2，最後是P1。

# In[196]:


#complete  missing values
x_train['Cabin'] = x_train['Cabin'].apply(lambda x : str(x)[0] if not pd.isnull(x) else 'NoCabin')


# In[197]:


x_train["Cabin"].unique()


# In[198]:


sns.countplot(x_train['Cabin'], hue=x_train['Survived'])


# In[140]:


total_Cabin_p1 = x_train[x_train['Pclass']==1]['Cabin'].value_counts()
total_Cabin_p2 = x_train[x_train['Pclass']==2]['Cabin'].value_counts()
total_Cabin_p3 = x_train[x_train['Pclass']==3]['Cabin'].value_counts()

print('-'*5 + 'p1 class' +'-'*5)
print(total_Cabin_p1)
print('-'*5 + 'p2 class' +'-'*5)
print(total_Cabin_p2)
print('-'*5 + 'p3 class' +'-'*5)
print(total_Cabin_p3)


# 
# - [x] 特徵工程Feature engineering
#    * 包含Feature cleaning、imputation、selection、encoding、normalization…等。
# 

# In[199]:


#combine SibSp&Parch to one 
x_train['Family'] = x_train['SibSp'] + x_train['Parch']
x_train.drop('SibSp' , 1, inplace=True)
x_train.drop('Parch', 1, inplace=True)


# In[200]:


#check null
print('Train columns with null values:\n', x_train.isnull().sum())

#Survived have 418 , these data belong test data


# In[201]:


x_train['Age'].fillna(x_train['Age'].median(), inplace = True)
#complete missing age with median
x_train['Fare'].fillna(x_train['Fare'].median(), inplace = True)
#complete missing fare with median


# In[202]:


#check null again
print('Train columns with null values:\n', x_train.isnull().sum())


# In[203]:


x_train.describe(include=['O'])


# In[204]:


x_train['Ticket_info'] = x_train['Ticket']
                         .apply(lambda x : x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')


# In[205]:


x_train['Ticket_info'].unique()


# In[206]:


#One-Hot Encoding

x_train['Sex'] = x_train['Sex'].map({'male':1,'female':0})    
x_train['Embarked'] = x_train['Embarked'].astype('category').cat.codes
x_train['Pclass'] = x_train['Pclass'].astype('category').cat.codes
x_train['Title1'] = x_train['Title1'].astype('category').cat.codes
x_train['Title2'] = x_train['Title2'].astype('category').cat.codes
x_train['Cabin'] = x_train['Cabin'].astype('category').cat.codes
x_train['Ticket_info'] = x_train['Ticket_info'].astype('category').cat.codes


# In[208]:


dataTrain = x_train[pd.notnull(x_train['Survived'])].sort_values(by=["PassengerId"])
dataTest = x_train[~pd.notnull(x_train['Survived'])].sort_values(by=["PassengerId"])


# In[209]:


dataTrain.columns


# In[211]:


dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family', 'Title2','Ticket_info','Cabin']]


# In[212]:


dataTrain


# - [x] 模型建立與訓練
#    * 模型選擇
#    * 訓練
#    * 評估
#    * 參數（Hyperparameter）調整
#    * 預測

# In[222]:


#import package
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate


# #### 特徵重要性(Feature Importance)
# ###### 我們使用最基本的樹狀模型 - 決策樹(Decision Tree)，來評估特徵欄位對於標籤欄位(Survived)的重要性
# ###### 目的是幫助我們篩選較具關鍵性的特徵欄位，提升模型的預測能力。評估特徵重要性的方法並非僅侷限於決策樹，另外其他常見的手法有：相關係數(Correlation)、Lasso等。

# In[225]:


# creat Random Forest Classifier
DTC = RandomForestClassifier( )
DTC.fit( X_train, Y_train )

# importance rank 
col_names = X_train.columns
importances = DTC.feature_importances_
Feature_Rank = pd.DataFrame( { 'Feature_Name':col_names, 'Importance':importances } )
Feature_Rank.sort_values( by='Importance', ascending=False, inplace=True ) 
Feature_Rank


# ### 在完成評估特徵重要性後，我們選擇使用隨機森林(Random Forest)來預測資料。
# * n_estimators: 樹的數量(default=10)。
# * min_samples_leaf: 最終葉節點最少樣本數(default=1)；當樣本不大時，可不設定使用預設，若樣本數量非常大時，則推薦增加此參數值。
# * min_samples_split:節點再劃分時所需的最小樣本數(default=2)；當樣本不大時，可不設定使用預設，若樣本數量非常大時，則推薦增加此參數值。
# * oob_score: 是否採用袋外樣本(out-of-bag samples)來評估模型的準確度(default=False)。

# In[214]:


#Random Forest model (ues all Feature)
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)


# In[226]:


X_Train = dataTrain[['Age', 'Fare',  'Pclass', 'Sex', 'Title2']]
Y_Train = dataTrain[['Survived']]
X_test = dataTest


# In[230]:


#Random Forest model (just ues top 5 Feature)
RFC_2 = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

RFC_2.fit( X_Train, Y_Train )
print("%.4f" % RFC_2.oob_score_)

