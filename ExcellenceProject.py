from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tkinter.filedialog import askopenfilename
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE
from flask import render_template
from tkinter import messagebox
from sklearn import svm
import numpy as np
import pandas
import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

#------------------------------GLOBAL VARIABLES------------------------------
app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#------------------------------FUNCTIONS------------------------------
def loadFile():
   filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
   df = pandas.read_excel(filename, sheet_name='Sheet1')
   print(filename)
   return(filename)

def readDataFromExcelFile(filename):
   #Read from the excel file
   df = pandas.read_excel(filename, sheet_name='Sheet1')
   #Take 80% of the values to teach the machine the behavior of the groups
   values=np.matrix(df.values)
   match=np.array(df.values)
   y_values=match=match[:,-1]
   match = match[: int(len(match) * .70)]
   x_values =prediction=values=np.delete(values,-1,1)
   values = values[: int(len(values) * .70)]
   prediction = prediction[int(len(prediction) * .70) :]
   return(values,match,prediction,x_values,y_values)

def printValuesOfArrays(values,match,prediction):
   #Printing the values
   print("Those are the values")
   print(values.tolist())
   #Printing the matches
   print("Those are the matches")
   print(match.tolist())
   print("Those are the predictions")
   print(prediction.tolist())

def featureSelection(values,match,numberOfFeatures):
   # feature extraction
   model = LogisticRegression()
   rfe = RFE(model, numberOfFeatures)
   fit = rfe.fit(values, match)
   print(("Num Features: %d") % fit.n_features_)
   print(("Selected Features: %s") % fit.support_)
   print(("Feature Ranking: %s") % fit.ranking_)

def teachingTheMachine(values,match):
   #Making a linear SVM
   clf = svm.SVC(kernel='linear')
   clf.fit(values,match)
   return(clf)

def predictingValues(clf,prediction):
   myList=[]
   print("The prediction for",len(prediction)," is:")
   for i in range(len(prediction)):
      myList.append(list(clf.predict(prediction[i])))
   print(myList)

def openExelFile():
   filename = loadFile()
   if (filename == ""):
      messagebox.showerror("Error", "Something went wrong")
   else:
      print("file name: ",os.path.basename(filename),"\n"+"file path: ",filename)

def confusionMatrix(x_values,y_values):
   x_train, x_test, y_train, y_test = train_test_split(x_values,y_values, test_size=0.20, random_state=0)
   print("this is the x_treain:")
   print(x_train)
   print("this is the x_test:")
   print(x_test)
   print("this is the y_train:")
   print(y_train)
   print("this is the y_test:")
   print(y_test)
   sc_x = StandardScaler()
   x_train = sc_x.fit_transform(x_train)
   x_test = sc_x.transform(x_test)
   logit = LogisticRegression(random_state=0)
   logit.fit(x_train, y_train)
   y_predicted = logit.predict(x_test)
   cm = confusion_matrix(y_test, y_predicted)
   print(cm)

def stapsOfTheMachineLearning():
   file=loadFile()
   values,match,prediction,x_values,y_values=readDataFromExcelFile(file)
   printValuesOfArrays(values,match,prediction)
   clf = teachingTheMachine(values, match)
   predictingValues(clf, prediction)
   featureSelection(values,match,4)
   confusionMatrix(x_values,y_values)

def allowed_file(filename):
    return '.' in filename and \
         filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
#------------------------------FLASK FUNCTIONS------------------------------
@app.route('/', methods=('GET', 'POST'))
def main():
    return render_template('home.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

#------------------------------MAIN------------------------------
if __name__ == "__main__":
    app.run(host='localhost', port=8000, debug=True)


