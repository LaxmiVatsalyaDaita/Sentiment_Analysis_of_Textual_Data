#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd

#read dataset
data=pd.read_csv("Reviews.csv")
data.head()


# In[12]:


#imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px


# In[13]:


#plotting the product scores
fig = px.histogram(data, x="Score")
fig.update_traces(marker_color="yellow",marker_line_color='rgb(0,0,0)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Scores')
fig.show()


# In[14]:


import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])
textt = " ".join(review for review in data.Text)
wordcloud = WordCloud(stopwords=stopwords).generate(textt)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('wordcloud11.png')
plt.show()


# In[15]:


#Classifying the reviewes as positive and negative
data = data[data['Score'] != 3]
data['sentiment'] = data['Score'].apply(lambda rating : +1 if rating > 3 else -1)


# In[16]:


data.head()


# In[17]:


#splitting the dataset into positive and negative reviews

positive = data[data['sentiment'] == 1]
negative = data[data['sentiment'] == -1]


# In[18]:


data['sentimentt'] = data['sentiment'].replace({-1 : 'negative'})
data['sentimentt'] = data['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(data, x="sentimentt")
fig.update_traces(marker_color="yellow",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show()


# In[19]:


#datacleaning
def remove_punctuation(text):
    final = "".join(u for u in text if u not in ("?", ".", ";", ":",  "!",'"'))
    return final

data['Text'] = data['Text'].apply(remove_punctuation)
data = data.dropna(subset=['Summary'])
data['Summary'] = data['Summary'].apply(remove_punctuation)


# In[20]:


newData = data[['Summary','sentiment']]
newData.head()


# In[21]:


#split with 70% for training and 30% for testing
index = newData.index
newData['random_number'] = np.random.randn(len(index))
train = newData[newData['random_number'] <= 0.70]
test = newData[newData['random_number'] > 0.30]


# In[22]:


# count vectorizer:
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = vectorizer.fit_transform(train['Summary'])
test_matrix = vectorizer.transform(test['Summary'])


# In[23]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

model = LogisticRegression()

X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']

#fitting the model
model.fit(X_train,y_train)

#predicting
y_predict = model.predict(X_test)

print(data.shape)

print("accuracy is: ", accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[24]:


feature_importance = model.coef_[0][:10]
for i,v in enumerate(feature_importance):
    print('Feature: ', list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(i)], 'Score: ', v)
feature_importance = model.coef_[0]
sorted_idx = np.argsort(feature_importance)

top_10_pos_w = [list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(w)] for w in sorted_idx[range(-1,-11, -1)]]
print(top_10_pos_w)


# In[25]:


fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(top_10_pos_w, feature_importance[sorted_idx[range(-1,-11, -1)]])
plt.title("Most Important Words Used for Positive Sentiment",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 40)
plt.ylabel('Feature Importance', fontsize = 12)
plt.xlabel('Word', fontsize = 12);


# In[26]:


top_10_neg_w = [list(vectorizer.vocabulary_.keys())[list(vectorizer.vocabulary_.values()).index(w)] for w in sorted_idx[:10]]
print(top_10_neg_w)


# In[27]:


fig = plt.figure(figsize=(10, 6))
ax = sns.barplot(top_10_neg_w, feature_importance[sorted_idx[:10]])
plt.title("Most Important Words Used for Negative Sentiment",fontsize = 20)
x_locs,x_labels = plt.xticks()
plt.setp(x_labels, rotation = 40)
plt.ylabel('Feature Importance', fontsize = 12)
plt.xlabel('Word', fontsize = 12);


# In[35]:


import speech_recognition as sr


# In[36]:


recording = sr.Recognizer()


# In[37]:


with sr.Microphone() as source: 
    recording.adjust_for_ambient_noise(source)
    print("Your review please..:")
    audio = recording.listen(source)


# In[38]:


try:
    userin=recording.recognize_google(audio)
    print("You said: " + userin)
except Exception as e:
    print(e)


# In[39]:


test_review = vectorizer.transform([userin])
output = model.predict(test_review)
if(output == 1):
    print("Review: Positive :)")
else:
    print("Review: Negative :(")


# In[26]:


recording = sr.Recognizer()


# In[27]:


with sr.Microphone() as source: 
    recording.adjust_for_ambient_noise(source)
    print("Please Say the Product_ID:")
    audio = recording.listen(source)


# In[28]:


try:
    prod_id=recording.recognize_google(audio)
    upproduct=prod_id.upper()
    finproduct=upproduct.replace(" ", "")
    print("You said: " + finproduct)
except Exception as e:
    print(e)


# In[29]:


key=int(input('Does your product ID and displayed Message match?? if yes enter 1 or else 0...'))


# In[30]:


if key==1:
    prod_id=finproduct
else:
    prod_id=input('Enter the product ID here')


# In[40]:


#product = data[data['ProductId'] == "B0000CDAY2"]
#product.head()
product = data[data['ProductId'] == prod_id]
product.head()


# In[41]:


string = " ".join(product['Summary'].tolist())
string = [string]
#print(string)

testing = vectorizer.transform(string)
Output = model.predict(testing)
if(Output == 1):
    print("Review: Positive :)")
else:
    print("Review: Negative :(")


# In[ ]:




