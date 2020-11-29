#!/usr/bin/env python
# coding: utf-8

# In[130]:


import pandas as pd
import numpy as np

data = pd.read_excel('107年新竹站_20190315.xls')


# In[131]:


data.head(18)
#total 18 classifications 


# In[132]:


#get Oct, Nov, Dec
data = data[data["日期"].between('2018/10/01','2018/12/31')]  
data.head(18)


# In[133]:


# NR replace 0
data = data.replace("NR",0) 


# In[134]:


#only take PM2.5
data = data[data['測項']=='PM2.5']
data.head(5)


# In[135]:


#separate train data and test data
train_pm25= data[data["日期"].between('2018/10/01','2018/11/30')] 
test_pm25= data[data["日期"].between('2018/11/01','2018/12/31')] 


# In[136]:


#drop 日期 測站 測項
train_pm25= train_pm25.iloc[:, 3:]
test_pm25= test_pm25.iloc[:, 3:]


# In[137]:


train_pm25


# In[138]:


test_pm25


# In[143]:


for i in range(61):
    for j in range(24):
        a=j-1
        b=j+1
        if str(train_pm25.iloc[i,j]).find("#") != -1 or str(train_pm25.iloc[i,j]).find("*") != -1 or str(train_pm25.iloc[i,j]).find("A") != -1 or str(train_pm25.iloc[i,j]).find("nan") != -1:
            while str(train_pm25.iloc[i,a]).find("#") != -1 or str(train_pm25.iloc[i,a]).find("*") != -1 or str(train_pm25.iloc[i,a]).find("A") != -1 or str(train_pm25.iloc[i,a]).find("nan") != -1:
                a=a-1
            while str(train_pm25.iloc[i,b]).find("#") != -1 or str(train_pm25.iloc[i,b]).find("*") != -1 or str(train_pm25.iloc[i,b]).find("A") != -1 or str(train_pm25.iloc[i,b]).find("nan") != -1:
                b=b+1                          
            #print(train_pm25.iloc[i,j])
            train_pm25.iloc[i,j] = (int(train_pm25.iloc[i,a]) + int(train_pm25.iloc[i,b])) / 2 


# In[120]:


for i in range(61):
    for j in range(24):
        a=j-1
        b=j+1
        if str(test_pm25.iloc[i,j]).find("#") != -1 or str(test_pm25.iloc[i,j]).find("*") != -1 or str(test_pm25.iloc[i,j]).find("A") != -1 or str(test_pm25.iloc[i,j]).find("nan") != -1:
            while str(test_pm25.iloc[i,a]).find("#") != -1 or str(test_pm25.iloc[i,a]).find("*") != -1 or str(test_pm25.iloc[i,a]).find("A") != -1 or str(test_pm25.iloc[i,a]).find("nan") != -1:
                a=a-1
            while str(test_pm25.iloc[i,b]).find("#") != -1 or str(test_pm25.iloc[i,b]).find("*") != -1 or str(test_pm25.iloc[i,b]).find("A") != -1 or str(test_pm25.iloc[i,b]).find("nan") != -1:
                b=b+1                          
            #print(train_pm25.iloc[i,j])
            test_pm25.iloc[i,j] = (int(test_pm25.iloc[i,a]) + int(test_pm25.iloc[i,b])) / 2 


# In[121]:


train_xlist=[]
train_ylist=[]
for i in range(18):                   #在一天24小時當中，如果使用6小時資料去預測第七小時，可以有18筆資料參考
    tempx=train_pm25.iloc[:,i:i+6]    #使用前6小時當train_x
    tempx.columns=np.array(range(6))
    tempy=train_pm25.iloc[:,i+6]      #使用第7小時當train_y
    tempy.columns=['1']
    train_xlist.append(tempx)
    train_ylist.append(tempy)


# In[122]:


test_xlist=[]
test_ylist=[]
for i in range(18):
    tempx2=test_pm25.iloc[:,i:i+6]        #使用前6小時當test_x
    tempx2.columns=np.array(range(6))
    tempy2=test_pm25.iloc[:,i+6]         #使用第7小時當test_y
    tempy2.columns=['1']
    test_xlist.append(tempx2)
    test_ylist.append(tempy2)


# In[123]:


train_X=pd.concat(train_xlist)     #train_X
train_X=np.array(train_X,int)     
train_X   


# In[124]:


test_X=pd.concat(test_xlist)     #test_X
test_X=np.array(test_X,int)     
test_X


# In[125]:


train_Y=pd.concat(train_ylist)      #train_Y
train_Y=(np.array(train_Y,int))
train_Y


# In[126]:


test_Y=pd.concat(test_ylist)      #test_Y
test_Y=(np.array(test_Y,int))
test_Y


# In[127]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
from sklearn import metrics


# In[128]:


#bulid model
rf = RandomForestRegressor(criterion='mae', 
                            n_estimators=100,
                            oob_score=True,
                            random_state=1,
                            n_jobs=-1) 

rf.fit(train_X ,train_Y)
#predicted
test_y_predicted = rf.predict(test_X)

print("Traing Score:%f"%rf.score(train_X,train_Y))
print("Testing Score:%f"%rf.score(test_X,test_Y))
print("Random Forest Regression MAE:%f"%metrics.mean_absolute_error(test_Y,test_y_predicted))


# In[129]:


#bulid model
lr = LinearRegression()
lr.fit(train_X,train_Y)
#predicted
lr_test_y_predicted = lr.predict(test_X)

print("Traing Score:%f"%lr.score(train_X,train_Y))
print("Testing Score:%f"%lr.score(test_X,test_Y))
print("LinearRegression MAE:%f"%metrics.mean_absolute_error(test_Y,lr_test_y_predicted))

