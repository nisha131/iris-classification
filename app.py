from flask import * 
import pandas as pd
import numpy as np
from  sklearn.linear_model import LogisticRegression

app = Flask  (__name__) 
url ='https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/iris.csv'

namelist = ['sepal_length',	'sepal_width',	'petal_length',	'petal_width',	'species']
iris = pd.read_csv(url , header=None,names = namelist )

iris1 = np.array(iris)
X = iris1[:,0:4] 
Y = iris1[:,4]

model = LogisticRegression()
model.fit(X,Y)
res= model.predict([[6.5,3.0,5.2,2.0]])

@app.route('/') 
def hello_world(): 
  return render_template('index.html')
  # return 'Hello, MGCU champs !   ' + str(res[0])
  
@app.route('/rec', methods=['POST']) 
def processdata(): 
  spl= float( request.form['spl'] )
  spw= float( request.form['spw'] )
  ptl= float( request.form['ptl'] )
  ptw= float( request.form['ptw'] )
  
  arr1 = np.array([ spl,spw,ptl,ptw]) 
  res= model.predict([arr1])
  
  return render_template("index.html",result=str(res[0]))


if __name__ == '__main__': 
  app.run()
