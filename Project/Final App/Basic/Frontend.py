from Backend import *

def predict():
    userLyrics = lyricsTextBox.get(1.0,"end-1c")
    print(userLyrics)

    user_prediction = model.predict(process(userLyrics))
    if(user_prediction[0] == 1):
        resultLabel.config(text = 'Song is Hit')
    else:
        resultLabel.config(text = 'Song is not Hit')








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
processButton = ttk.Button(page1, text="Apply Processing", width=40, command = predict)  # , command=predict)
processButton.grid(row=3, column=0)

# Label for showcasing the result.
ResultLabel = ttk.Label(page1, text="\"Click Apply to see result\"",
                        font=("Arial", 20))
ResultLabel.grid(row=5, column=0)

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
ResultLabel2 = ttk.Label(page2, text="Enter details and click \"Apply\" \nto see result", font=("Arial", 10), padding=7)
ResultLabel2.grid(row=2, column=3)

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

# Danceability label and entry
DanceabilityLabel = ttk.Label(page2, text="Song Danceability")
DanceabilityLabel.grid(row=4, column=3, sticky="E")
DanceabilityEntry = ttk.Entry(page2)
DanceabilityEntry.grid(row=4, column=4, sticky="W")

# Loudness label and entry Spotify measures loudness by ReplayGain standard.
LoudnessLabel = ttk.Label(page2, text="Song Loudness", padding=8)
LoudnessLabel.grid(row=5, column=0, sticky="E")
LoudnessEntry = ttk.Entry(page2)
LoudnessEntry.grid(row=5, column=1, sticky="W")

# Speechiness label and entry
SpeechinessLabel = ttk.Label(page2, text="Song Speechiness: ")
SpeechinessLabel.grid(row=5, column=3, sticky="E")
SpeechinessEntry = ttk.Entry(page2)
SpeechinessEntry.grid(row=5, column=4, sticky="W")

# Acousticness label and entry
AcousticnessLabel = ttk.Label(page2, text="Song Acousticness", padding=8)
AcousticnessLabel.grid(row=6, column=0, sticky="E")
AcousticnessEntry = ttk.Entry(page2)
AcousticnessEntry.grid(row=6, column=1, sticky="W")

# Button for Applying processing and prediction to Lyrics.
processButton2 = ttk.Button(page2, text="Apply Processing", width=40, padding=10)  # , command=predict)
processButton2.grid(row=7, column=0, columnspan=8)

# main loop the root window
root.mainloop()

