
# coding: utf-8

# In[2]:


#import pakeage
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')
import seaborn as sns


# In[3]:


data = pd.read_csv('character-deaths.csv')


# In[4]:


#check info
print(data.shape)
data.info()


# In[5]:


#check all columns
print(data.columns.values)
data.head(5)


# * Allegiances: 所屬國家
# * Death Year: 死亡年 (選擇)
# * Book of Death: 在第幾集死亡
# * Death Chapter: 在第幾章死亡
# * Book Intro Chapter: 書籍介紹章節
# * Gender: 1為男 0為女
# * Nobility: 1是貴族 0不是貴族
# * GoT: 1有出現在書本第一集 0沒有出現在書本第一集
# * CoK: 1有出現在書本第二集 0沒有出現在書本第二集
# * SoS: 1有出現在書本第三集 0沒有出現在書本第三集
# * FfC: 1有出現在書本第四集 0沒有出現在書本第四集
# * DwD: 1有出現在書本第五集 0沒有出現在書本第五集

# In[6]:


#drop Book of Death & Death Chapter
data.drop('Book of Death' , 1, inplace=True)
data.drop('Death Chapter', 1, inplace=True)


# In[7]:


#check all columns again
print(data.columns.values)
data.head(5)


# In[8]:


# have some null in data
print('Train columns with null values:\n', data.isnull().sum())
#Death Year & Book Intro Chapter


# In[9]:


#complete  missing values
data['Death Year'] = data['Death Year'].fillna(0)
data['Book Intro Chapter'] = data['Book Intro Chapter'].fillna(0)


# In[10]:


data.head(5)


# In[11]:


#change into 1 or 0, 0 is survived, 1 is dead
data['Death']=data['Death Year']
data.loc[data['Death Year'] !=0,'Death']=1


# In[12]:


data.head(5)


# In[13]:


# we only keep one death column
data.drop('Death Year' , 1, inplace=True)
data.head(5)


# In[14]:


#One-Hot Encoding
one_hot_encoding= pd.get_dummies(data['Allegiances'], prefix='A')


# In[15]:


one_hot_encoding.head(5)


# In[16]:


#combine these feature into data
data2 = pd.concat([one_hot_encoding, data], axis=1)


# In[17]:


#and drop Allegiances
data2.drop('Allegiances' , 1, inplace=True)
data2.head(5)


# In[18]:


#Death is predict
data_y = data['Death']


# In[19]:


data2.columns


# In[20]:


data_x = data2[['A_Arryn', 'A_Baratheon', 'A_Greyjoy', 'A_House Arryn',
       'A_House Baratheon', 'A_House Greyjoy', 'A_House Lannister',
       'A_House Martell', 'A_House Stark', 'A_House Targaryen',
       'A_House Tully', 'A_House Tyrell', 'A_Lannister', 'A_Martell',
       'A_None', 'A_Stark', 'A_Targaryen', 'A_Tully', 'A_Tyrell', 
       'A_Wildling', 'Book Intro Chapter', 'Gender',
       'Nobility', 'GoT', 'CoK', 'SoS', 'FfC', 'DwD']]


# In[26]:


#split data into train and test
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import tree

train_X, test_X, train_y, test_y = train_test_split(data_x, data_y, random_state=9487,test_size = 0.25)


# In[27]:


# build Classifier
clf = tree.DecisionTreeClassifier()
got_clf = clf.fit(train_X, train_y)


# In[28]:


# predict
test_y_predicted = got_clf.predict(test_X)
print(test_y_predicted)


# In[29]:


# ground truth
print(test_y.values)


# In[35]:


#show tree
fn = ['A_Arryn', 'A_Baratheon', 'A_Greyjoy', 'A_House Arryn',
       'A_House Baratheon', 'A_House Greyjoy', 'A_House Lannister',
       'A_House Martell', 'A_House Stark', 'A_House Targaryen',
       'A_House Tully', 'A_House Tyrell', 'A_Lannister', 'A_Martell',
       'A_None', 'A_Stark', 'A_Targaryen', 'A_Tully', 'A_Tyrell', 
       'A_Wildling', 'Book Intro Chapter', 'Gender',
       'Nobility', 'GoT', 'CoK', 'SoS', 'FfC', 'DwD']
cn = ['0', '1']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(got_clf, max_depth=3
                      , feature_names = fn
                      , class_names=cn
                      , filled = True)
fig.savefig('got.jpg')


# In[59]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[60]:


y_true = test_y.values
y_pred = test_y_predicted


# In[61]:


print('accuracy：{}'.format(accuracy_score(y_true, y_pred)))
print('precision：{}'.format(precision_score(y_true, y_pred, average=None)))
print('recall：{}'.format(recall_score(y_true, y_pred, average=None)))


# In[45]:


#show Confusion Matrix
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_true, y_pred=y_pred)
fig, ax = plt.subplots(figsize=(2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i,j], va='center', ha='center')
plt.xlabel('predicted label')        
plt.ylabel('true label')
plt.show()

