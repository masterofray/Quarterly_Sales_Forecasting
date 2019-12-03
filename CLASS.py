import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import chardet as ch

class Openfiles :
	def __init__(self, name):
        print('Testing initial class')
        self.name = name

    def browse_file(self):
        Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
		filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
		self.filename = filename
		return filename

    def load_data(self):
        dfmy = pd.read_csv(self.filename, delimiter = ';')
		