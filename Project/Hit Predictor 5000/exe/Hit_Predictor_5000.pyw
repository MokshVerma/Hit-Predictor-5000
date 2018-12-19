
# coding: utf-8

# In[16]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn_pandas import DataFrameMapper
from sklearn.naive_bayes import MultinomialNB

testCSV = pd.read_csv(r'Dataset(Analysis)(processed lyrics).csv') 


bow = CountVectorizer(max_features=1000,
                      lowercase=True,
                      ngram_range=(1,1),
                      analyzer = "word").fit(testCSV['Lyrics'].values.astype(str))
len(bow.vocabulary_)


lyrics_bow = bow.transform(testCSV['Lyrics'].values.astype(str))
print('Shape of Sparse Matrix: ', lyrics_bow.shape)
lyrics_bow.nnz

tfidf_transformer = TfidfTransformer().fit(lyrics_bow)
lyrics_tfidf = tfidf_transformer.transform(lyrics_bow)

df= pd.DataFrame()
df['lyrics']=list(lyrics_tfidf.toarray())
df['hit'] =  testCSV['Hit']


model = MultinomialNB()
model.fit(df['lyrics'].tolist(), df['hit'].tolist())



# In[17]:


def predict():
    userLyrics = lyricsTextBox.get(1.0,"end-1c")
    print(userLyrics)
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
    
    print(userLyrics)
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
    user_prediction = model.predict(user_lyrics_tfidf)
    
    
    if(user_prediction[0] == 1):
        resultLabel.config(text = 'Song is Hit')
    else:
        resultLabel.config(text = 'Song is not Hit')


# In[18]:


def process(userLyrics):
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
    return userLyrics


# In[21]:


def submit():
    user_Lyrics = lyricsTextBox2.get(1.0,"end-1c")
    user_Lyrics = process(user_Lyrics)
    print(type(user_Lyrics))
    print(user_Lyrics)
    df = pd.read_csv(r'Dataset(Advanced)(processed lyrics).csv') 
    df1 = pd.DataFrame([[float(EnergyEntry.get()), float(TempoEntry.get()), 
                         str(user_Lyrics),  int(ArtistPopularityEntry.get())]], 
                       columns = ['Energy', 'Tempo', 'Lyrics', 'Artist Hit'])
    print(len(df))
    df = df.append(df1, sort=True, ignore_index=True)
    print(len(df))
    print(df['Lyrics'].tail())
    
    df['Lyrics'] = df['Lyrics'].astype(str)
    mapper = DataFrameMapper([
      ('Lyrics', TfidfVectorizer()),
      ('Tempo', None),
      ('Energy', None),
     ('Artist Hit', None)
     ])
    
    features = mapper.fit_transform(df[['Lyrics', 'Tempo', 'Energy', 'Artist Hit']])
    
    #features1 = mapper.fit_transform()
    
    y = df['Hit'][:-1]
    
    print(len(features))
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()

    model.fit(features[:-1], y)
    predictions = model.predict(features[-1:])
    print(predictions)
    
    if(predictions[0] == 1):
        resultLabel2.config(text = 'Song is Hit')
    else:
        resultLabel2.config(text = 'Song is not Hit')


# In[24]:


import tkinter as tk
import tkinter.ttk as ttk

# creating root
root = tk.Tk()
root.geometry("800x500")
rows = 0

while rows < 2:
    root.rowconfigure(rows, weight=1)
    root.columnconfigure(rows, weight=1)
    rows += 1

# creating a frame for storing all widgets
AppFrame = tk.Frame(root)
AppFrame.grid(row=0, column=0)  # , columnspan=300)

# Creating label for title
AppTitle = tk.Label(AppFrame, text="Song HIT Prediction", font=("Arial", 30))
AppTitle.grid(row=0, column=0)
AppTitle2 = tk.Label(AppFrame, text="", font=("Arial", 5))
AppTitle2.grid(row=0, column=1)

''' Creating the first tab of the application... this tab is for the songwriter.'''
# Creating tabs for 2 use cases
notebook = ttk.Notebook(AppFrame)
notebook.grid(row=2, column=0, sticky="W", rowspan=100, columnspan=230)

# Define our first Tab. This tab contains textbox for
# entering lyrics and button for applying algorithm.
page1 = ttk.Frame(AppFrame)
notebook.add(page1, text='Lyrics Analysis')

# Label above textbox to tell user to "Enter lyrics Here".
EnterLyricsHereLabel = ttk.Label(page1, text="Enter Lyrics here:", padding=5)
EnterLyricsHereLabel.grid(row=1, column=0)

# Now create textbox in this "Lyrics Analysis" tab. Lyrics are input in this textbox
lyricsTextBox = tk.Text(page1, height=20, width=50)
lyricsTextBox.grid(row=2, column=0)  # , rowspan=50, columnspan=50)

# Button for Applying processing and prediction to Lyrics.
processButton = ttk.Button(page1, text="Apply Processing", width=40, command=predict)
processButton.grid(row=3, column=0)

# Label for showcasing the result.
resultLabel = ttk.Label(page1, text="\"Click Apply to see result\"",
                        font=("Arial", 20))
resultLabel.grid(row=5, column=0)

'''  This part of code now declares the page2, use case of music production team.'''

''' Creating page2 for Music Production company use cases features'''
page2 = ttk.Frame(AppFrame)
notebook.add(page2, text='Music Extras')

''' Adding Widgets on page2 Music Extras'''
EnterLyricsHereLabel2 = ttk.Label(page2, text="Enter Lyrics here:", padding=5)
EnterLyricsHereLabel2.grid(row=1, column=0)

# Now create textbox in this "Extras" tab. Lyrics are input in this textbox
lyricsTextBox2 = tk.Text(page2, height=10, width=40)
lyricsTextBox2.grid(row=2, column=0, columnspan=3)  # , rowspan=50, columnspan=50)

# Result Label
resultLabel2 = ttk.Label(page2, text="", font=("Arial", 20), padding=7)
resultLabel2.grid(row=2, column=3)

# Artist popularity entry box
ArtistPopularityLabel = ttk.Label(page2, text="Artist popularity: ", padding=8)
ArtistPopularityLabel.grid(row=3, column=0, sticky='E')
ArtistPopularityEntry = ttk.Entry(page2)
ArtistPopularityEntry.grid(row=3, column=1, sticky="W")

# Song Tempo label and entry
TempoLabel = ttk.Label(page2, text="Song Tempo: ")
TempoLabel.grid(row=3, column=3, sticky="E")
TempoEntry = ttk.Entry(page2)
TempoEntry.grid(row=3, column=4, sticky="W")

# Energy label and entry
EnergyLabel = ttk.Label(page2, text="Song Energy: ")
EnergyLabel.grid(row=4, column=0, sticky="E")
EnergyEntry = ttk.Entry(page2)
EnergyEntry.grid(row=4, column=1, sticky="W")

submitButton = ttk.Button(page2, text="Submit", width=40, padding=10,  command=submit)
submitButton.grid(row=7, column=0, columnspan=8)
# main loop the root window
root.mainloop()

