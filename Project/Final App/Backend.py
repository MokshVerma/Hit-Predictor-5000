
# coding: utf-8

# In[15]:


import pandas as pd


# In[16]:


import numpy as np


# In[17]:


testCSV = pd.read_csv('jugaad(1).csv')


# In[29]:


#Baby by Justin Bieber (shitty popularity)
userLyrics = "Ooh whoa, ooh whoa, ooh whoa You know you love me, I know you care Just shout whenever and I'll be there You are my love, you are my heart And we will never, ever, ever be apart Are we an item? Girl quit playin' We're just friends, what are you sayin' Said there's another, look right in my eyes My first love, broke my heart for the first time"


# In[30]:


def basicLyricsProcessing(userLyrics):
    #Everything in lowercase
    lowerCase = lambda x: " ".join(x.lower() for x in str(x).split())
    userLyrics = lowerCase(userLyrics)

    #Removing punctuation that does not add meaning to the song
    userLyrics = userLyrics.replace('[^\w\s]','')
    
    #Removing of stop words
    from nltk.corpus import stopwords

    stop = stopwords.words('english')
    removeStopWords = lambda x: " ".join(x for x in str(x).split() if x not in stop)
    userLyrics = removeStopWords(userLyrics)
    
    #Correction of Spelling mistakes
    from textblob import TextBlob
    spellingMistake = lambda x: str(TextBlob(x).correct())
    userLyrics = spellingMistake(userLyrics)
    
    #Lemmatization is basically converting a word into its root word. It is preferred over Stemming.
    from textblob import Word
    lemmatize = lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
    userLyrics = lemmatize(userLyrics)
    
    #CountVectorization of user lyrics
    from sklearn.feature_extraction.text import CountVectorizer
    user_bow = CountVectorizer(max_features=1000,
                          lowercase=True,
                          ngram_range=(1,1),
                          analyzer = "word").fit([userLyrics])
    
    #Bag of Words conversion of user lyrics
    user_lyrics_bow = bow.transform([userLyrics])
    
    #Tf-idf transforming of user lyrics
    from sklearn.feature_extraction.text import TfidfTransformer
    user_tfidf_transformer = TfidfTransformer().fit(user_lyrics_bow)
    user_lyrics_tfidf = user_tfidf_transformer.transform(user_lyrics_bow)
    return user_lyrics_tfidf


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000,
                      lowercase=True,
                      ngram_range=(1,1),
                      analyzer = "word").fit(testCSV['Lyrics'].values.astype(str))
len(bow.vocabulary_)


# In[32]:


lyrics_bow = bow.transform(testCSV['Lyrics'].values.astype(str))
print('Shape of Sparse Matrix: ', lyrics_bow.shape)
lyrics_bow.nnz


# In[33]:


df= pd.DataFrame()
df['lyrics']=list(lyrics_bow.toarray())
df['hit'] =  testCSV['Hit']


# In[34]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()


# In[35]:


model.fit(df['lyrics'].tolist(), df['hit'].tolist())


# # Testing model

# In[36]:


#user_bow = basicLyricsProcessing(userLyrics)


# In[37]:


df_user = pd.DataFrame()
df_user['lyrics']=list(basicLyricsProcessing(userLyrics).toarray())


# In[38]:


user_prediction = model.predict(df_user['lyrics'].tolist())


# In[39]:


user_prediction

