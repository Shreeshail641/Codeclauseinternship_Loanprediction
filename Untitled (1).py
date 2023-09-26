#!/usr/bin/env python
# coding: utf-8

# In[76]:


#importing the python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[77]:


#importing the Train dataset
dataset=pd.read_csv("train.csv")


# In[78]:


#Exploring the Dataset
dataset.head()


# In[79]:


#Finding the shape of Dataset
dataset.shape


# In[80]:


dataset.info()


# In[81]:


dataset.describe()


# In[82]:


pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'],margins=True)


# In[83]:


#visualizing the Dataset through Boxplot
dataset.boxplot(column='ApplicantIncome')


# In[84]:


#Finding a Histogram For Variable
dataset['ApplicantIncome'].hist(bins=20)


# In[85]:


#Finding a Histogram For Variable
dataset['CoapplicantIncome'].hist(bins=20)


# In[86]:


#Understanding the relationship between Applicationincome,education
dataset.boxplot(column='ApplicantIncome',by='Education')


# In[87]:


#Boxplot For loanamount
dataset.boxplot(column='LoanAmount')


# In[88]:


#Histogram For Loanamount
dataset['LoanAmount'].hist(bins=20)


# In[89]:


#Normalizing the Loanamount
dataset['LoanAmount_log']=np.log(dataset['LoanAmount'])
dataset['LoanAmount_log'].hist(bins=20)


# In[90]:


#finding the number of missing values
dataset.isnull().sum()


# In[91]:


dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True)


# In[92]:


dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)


# In[93]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[94]:


dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)


# In[95]:


dataset.LoanAmount=dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log=dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())


# In[96]:


dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)


# In[97]:


dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)


# In[98]:


dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True)


# In[99]:


dataset.isnull().sum()


# In[100]:


dataset['TotalIncome']=dataset['ApplicantIncome']+dataset['CoapplicantIncome']
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])


# In[101]:


#normalizing the Totalincome
dataset['TotalIncome_log'].hist(bins=20)


# In[102]:


#updated dataset
dataset.head()


# In[103]:


#dividing the dataset into dependent variable and independent variable
#X represents the independnt variable And Y represents the dependent variable
X=dataset.iloc[:,np.r_[1:5,9:11,13:15]].values
Y=dataset.iloc[:,12].values


# In[104]:


X


# In[105]:


Y


# In[106]:


#spliting the dataset into traindataset And testdataset In the ratio of 80And 20
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[107]:


print(X_train)


# In[108]:


from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()


# In[109]:


for i in range(0,5):
    X_train[:,i]=labelencoder_X.fit_transform(X_train[:,i])


# In[110]:


X_train[:,7]=labelencoder_X.fit_transform(X_train[:,7])


# In[111]:


X_train


# In[112]:


#converting categorical into numerical Format
labelencoder_Y=LabelEncoder()
Y_train=labelencoder_Y.fit_transform(Y_train)


# In[113]:


Y_train


# In[114]:


for i in range(0,5):
    X_test[:,i]=labelencoder_X.fit_transform(X_test[:,i])


# In[115]:


X_test[:,7]=labelencoder_X.fit_transform(X_test[:,7])


# In[116]:


labelencoder_Y=LabelEncoder()
Y_test=labelencoder_Y.fit_transform(Y_test)


# In[117]:


X_test


# In[118]:


Y_test


# In[119]:


#scaling the dataset
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)


# In[120]:


#applying the decisiontree  algorithms
from sklearn.tree import DecisionTreeClassifier
DTClassifier=DecisionTreeClassifier(criterion='entropy',random_state=0)
DTClassifier.fit(X_train,Y_train)


# In[121]:


Y_pred=DTClassifier.predict(X_test)
Y_pred


# In[122]:


#finding the accuracy
from sklearn import metrics
print('The accuracy of decision tree is:',metrics.accuracy_score(Y_pred,Y_test))


# In[123]:


#applying Gaussian algorithm
from sklearn.naive_bayes import GaussianNB
NBClassifier=GaussianNB()
NBClassifier.fit(X_train,Y_train)


# In[124]:


Y_pred=NBClassifier.predict(X_test)
Y_pred


# In[125]:


#finding the accuracy
print('The accuracy of Naive Bayes is:',metrics.accuracy_score(Y_pred,Y_test))


# In[126]:


#reading the test dataset
testdata=pd.read_csv("test.csv")


# In[127]:


testdata.head()


# In[128]:


testdata.info()


# In[129]:


#finding the missing values
testdata.isnull().sum()


# In[130]:


#handling missing values
testdata['Gender'].fillna(testdata['Gender'].mode()[0],inplace=True)


# In[131]:


testdata['Dependents'].fillna(testdata['Dependents'].mode()[0],inplace=True)


# In[132]:


testdata['Self_Employed'].fillna(testdata['Self_Employed'].mode()[0],inplace=True)


# In[133]:


testdata['Loan_Amount_Term'].fillna(testdata['Loan_Amount_Term'].mode()[0],inplace=True)


# In[134]:


testdata['Credit_History'].fillna(testdata['Credit_History'].mode()[0],inplace=True)


# In[135]:


testdata.isnull().sum()


# In[136]:


#finding the Boxplot for loan amount
testdata.boxplot(column='LoanAmount')


# In[137]:


#finding the boxplot for applicantincome
testdata.boxplot(column='ApplicantIncome')


# In[138]:


testdata.LoanAmount=testdata.LoanAmount.fillna(testdata.LoanAmount.mean())


# In[139]:


testdata['LoanAmount_log']=np.log(testdata['LoanAmount'])


# In[140]:


testdata.isnull().sum()


# In[141]:


testdata['Married'].fillna(testdata['Married'].mode()[0],inplace=True)


# In[142]:


testdata.isnull().sum()


# In[143]:


#suming the applicantincome and coapplicant income to get totalincome
testdata['TotalIncome']=testdata['ApplicantIncome']+testdata['CoapplicantIncome']
testdata['TotalIncome_log']=np.log(testdata['TotalIncome'])


# In[144]:


testdata.head()


# In[145]:


test=testdata.iloc[:,np.r_[1:5,9:11,13:15]].values


# In[146]:


for i in range(0,5):
    test[:,i]=labelencoder_X.fit_transform(test[:,7])


# In[147]:


test


# In[148]:


#scaling the dataset
test=ss.fit_transform(test)


# In[149]:


#predicting the values by using NB algorithm
pred=NBClassifier.predict(test)


# In[150]:


# 1 for eligible and 0 for not eligible
pred


# In[ ]:





# In[ ]:





# In[ ]:




