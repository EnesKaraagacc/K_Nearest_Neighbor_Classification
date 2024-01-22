#!/usr/bin/env python
# coding: utf-8

# # ENES KARAAĞAÇ

# STEP-1

# In[2]:


#importing libraries

import numpy as np
import pandas as pd


# In[3]:


#reading datasets

animes=pd.read_csv(r"C:\Users\90544\Desktop\Dataset_A1\animes.csv")

train_data=pd.read_csv(r"C:\Users\90544\Desktop\Dataset_A1\user_rates_train.csv")

test_data=pd.read_csv(r"C:\Users\90544\Desktop\Dataset_A1\user_rates_test.csv")

train_data


# In[4]:


big_data=pd.concat([train_data,test_data])
big_data


# In[5]:


animes.head()


# In[6]:


#Combining anime data with train data

merge_train=pd.merge(train_data,animes,on="anime_id",how="inner")
merge_test=pd.merge(test_data,animes,on="anime_id",how="inner")


# In[7]:


#Deleting unnecessary columns

merge_train=merge_train.drop(['Name','Image URL','Username'],axis=1)
merge_test=merge_test.drop(['Name','Image URL','Username'],axis=1)

merge_train.head()


# In[8]:


#Split genres parts and add to the list each part.

matrix=[]
for i in merge_train['Genres']:
    a=i.split(', ')
    for s in a:
        j=0
        for m in matrix:
            if(s==m):
                j=1
        if j==0:
            matrix.append(s)
print(matrix)


# In[9]:


#we add genres columns to merge_train dataframe and put the value 0
for genre in matrix:
    merge_train[genre]=0
merge_train.head()


# In[10]:


#if the anime has this genre we change the value 0 to 1

index=0
for row in merge_train['Genres']:
    a=row.split(', ')
    for genre in matrix:
        if genre in a:
             merge_train.at[index, genre] = 1
    index+=1
merge_train.head()


# In[11]:


#we drop the genres columns.
merge_train=merge_train.drop("Genres",axis=1)


# we did same step for test data.

# In[12]:


k=0
matrix=[]
for i in merge_test['Genres']:
    a=i.split(', ')
    for s in a:
        j=0
        for m in matrix:
            if(s==m):
                j=1
        if j==0:
            matrix.append(s)


# In[13]:


for genre in matrix:
    merge_test[genre]=0


# In[14]:


index=0
for row in merge_test['Genres']:
    a=row.split(', ')
    for genre in matrix:
        if genre in a:
             merge_test.at[index, genre] = 1
    index+=1


# In[15]:


merge_test=merge_test.drop("Genres",axis=1)


# In[ ]:





# In[16]:


#Converting hour to minutes for train data .

time_matrix=[]
for i in merge_train['Duration'] :
    time=0
    a=i.split(' ')
    k=0
    for j in a:
        if j==('hr'):
            time+=int(a[k-1])*60
        if j==('min'):
            time+=int(a[k-1])
        k+=1
    time_matrix.append(time)
    
merge_train["Duration"]=time_matrix
merge_train.head()


# In[17]:


#Converting hour to minutes for test data.

time_matrix=[]
for i in merge_test['Duration'] :
    time=0
    a=i.split(' ')
    k=0
    for j in a:
        if j==('hr'):
            time+=int(a[k-1])*60
        if j==('min'):
            time+=int(a[k-1])
        k+=1
    time_matrix.append(time)
    
merge_test["Duration"]=time_matrix
merge_test.head()


# In[ ]:





# In[18]:


#we drop unnecessary columns
merge_train=merge_train.drop("Anime Title",axis=1)
merge_train=merge_train.drop("Studios",axis=1)

merge_test=merge_test.drop("Anime Title",axis=1)
merge_test=merge_test.drop("Studios",axis=1)

merge_train.head()


# In[19]:


#we seperate type and source part
merge_train=pd.get_dummies(merge_train, drop_first=True, columns=["Type", "Source"])
merge_test=pd.get_dummies(merge_test, drop_first=True, columns=["Type", "Source"])

merge_train.head()


# In[20]:


#sorting the data for better perspective
merge_train.sort_values("user_id",inplace=True)
merge_test.sort_values("user_id",inplace=True)

merge_train


# In[21]:


merge_test.columns


# In[22]:


merge_train.columns


# In[23]:


#adding missing columns to test data and put the value 0. We want train and test data will have same columns
merge_test["Hentai"]=0
merge_test["UNKNOWN"]=0
merge_test["Type_Music"]=0
merge_test["Source_Book"]=0
merge_test["Source_Card game"]=0
merge_test["Source_Mixed media"]=0
merge_test["Source Music"]=0
merge_test["Source_Picture book"]=0
merge_test["Source_Radio"]=0
merge_test["Source_Web novel"]=0
merge_test


# In[24]:


# We sort the test data , because we want to test and train data will has same row order. Then we can calculate similarities.

sorted_columns = sorted(merge_test.columns)
merge_test = merge_test[sorted_columns]

merge_test


# In[25]:


sorted_columns = sorted(merge_train.columns)
merge_train = merge_train[sorted_columns]

merge_train


# STEP-2

# K-Fold=5

# In[ ]:





# In[26]:


X=merge_train.drop(['rating'],axis=1)
y=merge_train['rating']


# In[27]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

cross_val_score(LogisticRegression(),X,y,cv=5)


# In[ ]:





# STEP-3

# In[28]:


def euclidean_distance(row1, row2):
    distance = np.linalg.norm(row1 - row2)
    return distance


# In[109]:


#user1=train row , user2=test row
#this function calculate cosine similarity between test and train data.
from sklearn.metrics.pairwise import cosine_similarity
def user_similarity(user1,user2):
    ax=0
    for idx_test in range(user2.shape[0]):
        ay=0
        for idx_train in range(user1.shape[0]):
            ay+=cosine_similarity(np.array([user1[idx_train]]), np.array([user2[idx_test]]))
            
        ax+=(ay/user1.shape[0])
    return ax


# In[30]:


merge_train['user_id'].nunique()


# In[31]:


merge_train_user=merge_train['user_id'].unique()
merge_train_user


# In[32]:


merge_test['user_id'].nunique()


# In[33]:


merge_test_user=merge_test['user_id'].unique()
merge_test_user


# In[37]:


#we have 3000 train data users , and 50 test data users
matrix_similarity = np.zeros((3000, 50))


# In[110]:


#We calculate similarities between test and train data users.
#Then we assign the similarity value of each test user to the train user into the matrix.
for i in range(50):
    test_user=merge_test.loc[merge_test["user_id"]==merge_test_user[i]]
    for j in range(3000):
        train_user=merge_train.loc[merge_train["user_id"]==merge_train_user[j]]
        matrix_similarity[j][i]=user_similarity(np.array(test_user), np.array(train_user))


# In[111]:


matrix_similarity


# In[112]:


matrix_rating = np.zeros((877,3 ))
matrix_rating


# In[122]:


#we want to calculate most 7 similar users for each test user.
#we have 3 lists. one of them "largest" for while calculating , we assign list's biggest value.
#one of them "largest_similarity" for keep the value of 7 closest train users for each test user
#one of them index keep the index of train users in the largest_similarity list.

largest=[]
largest_similarity=np.zeros((50,7))
index=np.zeros((50,7))
for i in range(50):
    for k in range(7):
        x1=0
        x=0
        for j in range(3000):
            if (matrix_similarity[j][i]>x1 and matrix_similarity[j][i] not in largest):
                x1=matrix_similarity[j][i]
                x=j
        largest.append(x1)
        index[i][k]=x
        largest_similarity[i][k]=x1
print(index[0][0])


# In[ ]:





# In[143]:


#There is complicated part so I want to explain Turkish.
#Burada 6 tane liste oluşturdum.
#İlk 3 listeye en yakın 3,5,7 komşuya göre rating tahminlerini , son üç listeye ise 3,5,7 komşularının ağırlıklarına göre atadım. 
#Bunu yaparken önceden oluşturduğum listelerin içindeki değerleri kullandım.


key=0
rating_predict_values=[]
rating_predict_values2=[]
rating_predict_values3=[]

rating_predict_values_weighted=[]
rating_predict_values2_weighted=[]
rating_predict_values3_weighted=[]


for x in merge_test_user:
    mtx_test=merge_test.loc[merge_test["user_id"]==x]
    mtx_train1=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][0])])]
    mtx_train2=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][1])])]
    mtx_train3=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][2])])]
    mtx_train4=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][3])])]
    mtx_train5=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][4])])]
    mtx_train6=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][5])])]
    mtx_train7=merge_train.loc[merge_train["user_id"]==(merge_train_user[int(index[key][6])])]
    for i in mtx_test.to_numpy():
        predict=0
        predict_weighted=0
        summ = sum(largest_similarity[key])
        
        for j in mtx_train1.to_numpy():
        
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][0]/summ)
        for j in mtx_train2.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45] 
                predict_weighted+=j[45]*(largest_similarity[key][1]/summ)

        for j in mtx_train3.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][2]/summ)

        if (predict!=0):
            rating_predict_values.append(predict/3)
            rating_predict_values_weighted.append(predict_weighted)
        if (predict==0):
            rating_predict_values.append(np.mean(merge_train["rating"].to_numpy()))
            rating_predict_values_weighted.append(np.mean(merge_train["rating"].to_numpy()))
            
        for j in mtx_train4.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][3]/summ)
        for j in mtx_train5.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][4]/summ)
        if (predict!=0):
            rating_predict_values2.append(predict/5)
            rating_predict_values2_weighted.append(predict_weighted)
        if (predict==0):
            rating_predict_values2.append(np.mean(merge_train["rating"].to_numpy()))
            rating_predict_values2_weighted.append(np.mean(merge_train["rating"].to_numpy()))
        
        for j in mtx_train6.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][5]/summ)
        for j in mtx_train7.to_numpy():
            if(i[44]==j[44]):
                predict+=j[45]
                predict_weighted+=j[45]*(largest_similarity[key][6]/summ)
        if (predict!=0):
            rating_predict_values3.append(predict/7)
            rating_predict_values3_weighted.append(predict_weighted)
        if (predict==0):
            rating_predict_values3.append(np.mean(merge_train["rating"].to_numpy()))
            rating_predict_values3_weighted.append(np.mean(merge_train["rating"].to_numpy()))
    key+=1


# In[ ]:





# In[ ]:





# In[144]:


#for k=3 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values)]) / n
mae


# In[145]:


#for k=5 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values2)]) / n
mae


# In[146]:


#for k=7 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values3)]) / n
mae


# In[147]:


#for weighted and k=3 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values_weighted)]) / n
mae


# In[148]:


#for weighted and k=5 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values2_weighted)]) / n
mae


# In[149]:


#for weighted and k=7 mae
n = len(merge_test["rating"].to_numpy())
mae = sum([abs(gerçek - tahmin) for gerçek, tahmin in zip(merge_test["rating"].to_numpy(), rating_predict_values3_weighted)]) / n
mae


# # Report

# First I draw the matrix after merging animes and user data.Then I calculate similarities for each users between train and test data.Then I add the value into the matrix which is x coordinate is test value and y coordinate is train value. Then I check for 3,5,7 closest neighbors's rating data.I think there is problem with calculating similarities.Beceause there are a lot of closest neighbor which is no common anime. I put them mean of rating and it makes mae bigger.
# There is min error for k=3.So it is optimal value for these oppurtunities.
