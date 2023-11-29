#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Task(ML HuggingFace Inference models) 
Explore the data (question2.xlsx) and answer the following questions:
i. Perform sentiment analysis on the data and display the top 10 most negative, and the top 10
most positive strings of text.
# ## Binary Classification(0,1) 

# ### Import some needed packages

# In[32]:


# Import some needed libraries
# python --version ==3.8

# for data processing
#!pip install nltk
#!pip install openpyxl
# for model inference
#!pip install emot (package for emoje processing)
#pip install torch # in ordeer to user the package "transformers", the torch version must be >2.0(2.1 in this case)
#!pip install transformers (Install HuggingFace's sentiment analysis Transformer model)

# for ploting
#!pip install plotly
#!pip install matplotlib
#!pip install seaborn



import pandas as pd
import data_preprocessing

# For model analysis
from transformers import pipeline #model parameters = 268M
import plotly.express as px
import matplotlib.pyplot as plt


# In[33]:


# Load the data from the Excel file
df = pd.read_excel('question2.xlsx')


# In[34]:


print(df.head(20)) #this looks like Twitter data(Problem is Twitter sentiment analysis)


# In[35]:


#Data Preprocessing


# In[36]:


#apply all the functions above
df['hashtag'] = df.TextData.apply(func = data_preprocessing.hashtags)
df['TextData'] = df.TextData.apply(func = data_preprocessing.emoji)
df['TextData'] = df.TextData.apply(func = data_preprocessing.non_ascii)
df['TextData'] = df.TextData.apply(func = data_preprocessing.lower)
df['TextData'] = df.TextData.apply(func = data_preprocessing.removeStopWords)
df['TextData'] = df.TextData.apply(func = data_preprocessing.punct)
df['TextData'] = df.TextData.apply(func = data_preprocessing.remove_)


# In[37]:


print(df.head(20))


# ### Model Inference

# In[38]:


#Model Inference 
sentiment_model = pipeline(model="federicopascual/finetuning-sentiment-model-3000-samples")


# In[39]:


# Create a list to store the sentiments and scores
sentiments = []

# Iterate through the rows of the DataFrame and analyze the sentiment
for index, row in df.iterrows():
    text = row['TextData']
    result = sentiment_model([text])
    sentiment_label = result[0]['label']
    sentiment_score = result[0]['score']
    sentiments.append((index, text, sentiment_label, sentiment_score))


# In[40]:


sentiments


# In[41]:


# Create a DataFrame from the sentiments list
sentiments_df = pd.DataFrame(sentiments, columns=['ID', 'TextData', 'Sentiment', 'SentimentScore'])


# In[42]:


sentiments_df


# ### Model Post-Analysis

# #### Displaying the top 10 most positive and negative sentiments

# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have sentiments_df from the previous code

# Sort sentiments_df by SentimentScore to get top negative and top positive
top_negative = sentiments_df.nlargest(10, 'SentimentScore')
top_positive = sentiments_df.nsmallest(10, 'SentimentScore')


# In[44]:


# Plot bar chart for Top 10 Most Negative Sentiments
plt.figure(figsize=(10, 6))
sns.barplot(x='SentimentScore', y='TextData', data=top_negative, palette='Reds')
plt.title('Top 10 Most Negative Sentiments')
plt.xlabel('Sentiment Score')
plt.ylabel('Text Data')
plt.show()


# In[ ]:





# In[45]:


# Plot bar chart for Top 10 Most Positive Sentiments
plt.figure(figsize=(10, 6))
sns.barplot(x='SentimentScore', y='TextData', data=top_positive, palette='Greens')
plt.title('Top 10 Most Positive Sentiments')
plt.xlabel('Sentiment Score')
plt.ylabel('Text Data')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




