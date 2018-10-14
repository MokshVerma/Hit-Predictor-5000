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
root.geometry("600x500")
rows = 0

while rows < 2:
    root.rowconfigure(rows, weight=1)
    root.columnconfigure(rows, weight=1)
    rows += 1

# creating a frame for storing all widgets
AppFrame = tk.Frame(root)
AppFrame.grid(row=0, column=0)

# Creating label for title
AppTitle = tk.Label(AppFrame, text="Hit Predictor 5000", font=("Arial", 30))
AppTitle.grid(row=0, column=0)
AppTitle2 = tk.Label(AppFrame, text="Patent Pending", font=("Arial", 5))
AppTitle2.grid(row=0, column=1)

''' Creating the first tab of the application... this tab is for the songwriter.'''
# Creating tabs for 2 use cases
notebook = ttk.Notebook(AppFrame)
notebook.grid(row=2, column=0, sticky="W", rowspan=100, columnspan=230)

# Define our first Tab. This tab contains textbox for
# entering lyrics and button for applying algorithm.
page1 = ttk.Frame(root)
notebook.add(page1, text='Lyrics Analysis')

# Label above textbox to tell user to "Enter lyrics Here".
EnterLyricsHereLabel = ttk.Label(page1, text="Enter Lyrics here:", padding=5)
EnterLyricsHereLabel.grid(row=1, column=0)

# Now create textbox in this "Lyrics Analysis" tab. Lyrics are input in this textbox
lyricsTextBox = tk.Text(page1, height=20, width=50)
lyricsTextBox.grid(row=2, column=0)# , rowspan=50, columnspan=50)


# Button for Applying processing and prediciton to Lyrics.
processButton = ttk.Button(page1, text="Apply Processing", width=40, command=predict)
processButton.grid(row=3, column=0)

# Label for showcasing the result.
resultLabel = ttk.Label(page1, text="\"Result will be shown here inplace of this text\"",
                             font=("Arial", 20))
resultLabel.grid(row=5, column=0)

'''  This part of code now declares the page2, use case of music production team.
'''

''' Creating page2 for Music Production company use cases features'''
page2 = ttk.Frame(root)
notebook.add(page2, text='Music Extras')

# main loop the root window
root.mainloop()

