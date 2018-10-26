import pandas as pd

testCSV = pd.read_csv('Dataset(Analysis)(processed lyrics).csv') 

from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000,
                      lowercase=True,
                      ngram_range=(1,1),
                      analyzer = "word").fit(testCSV['Lyrics'].values.astype(str))
len(bow.vocabulary_)


lyrics_bow = bow.transform(testCSV['Lyrics'].values.astype(str))
print('Shape of Sparse Matrix: ', lyrics_bow.shape)
lyrics_bow.nnz

df= pd.DataFrame()
df['lyrics']=list(lyrics_bow.toarray())
df['hit'] =  testCSV['Hit']

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(df['lyrics'].tolist(), df['hit'].tolist())




def process(userLyrics):
    # Everything in lowercase
    lowerCase = lambda x: " ".join(x.lower() for x in str(x).split())
    userLyrics = lowerCase(userLyrics)

    # Removing punctuation that does not add meaning to the song
    userLyrics = userLyrics.replace('[^\w\s]', '')

    # Removing of stop words
    from nltk.corpus import stopwords

    stop = stopwords.words('english')
    removeStopWords = lambda x: " ".join(x for x in str(x).split() if x not in stop)
    userLyrics = removeStopWords(userLyrics)

    # Correction of Spelling mistakes
    from textblob import TextBlob
    spellingMistake = lambda x: str(TextBlob(x).correct())
    userLyrics = spellingMistake(userLyrics)

    # Lemmatization is basically converting a word into its root word. It is preferred over Stemming.
    from textblob import Word
    lemmatize = lambda x: " ".join([Word(word).lemmatize() for word in x.split()])
    userLyrics = lemmatize(userLyrics)

    # CountVectorization of user lyrics
    from sklearn.feature_extraction.text import CountVectorizer
    user_bow = CountVectorizer(max_features=1000,
                               lowercase=True,
                               ngram_range=(1, 1),
                               analyzer="word").fit([userLyrics])

    # Bag of Words conversion of user lyrics
    user_lyrics_bow = bow.transform([userLyrics])

    # Tf-idf transforming of user lyrics
    from sklearn.feature_extraction.text import TfidfTransformer
    user_tfidf_transformer = TfidfTransformer().fit(user_lyrics_bow)
    user_lyrics_tfidf = user_tfidf_transformer.transform(user_lyrics_bow)
    df_user = pd.DataFrame()
    df_user['lyrics'] = list(user_lyrics_tfidf.toarray())
    return df_user['lyrics']