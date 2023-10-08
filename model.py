import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("Crop_recommendation.csv")

pd.pandas.set_option('display.max_columns', None)

x=dataset.drop('label',axis=1)  # for extracting attributes

#la=LabelEncoder()
#dataset['label']=la.fit_transform(dataset['label'])

print(dataset)

model=[]
accuracy=[]

y=dataset['label']

x_train,x_test,y_train,y_test=train_test_split(x,y, test_size = 0.2, random_state=40)

rf=RandomForestClassifier()
rf.fit(x_train, y_train)

# values = np.array([[99,15,27,27.41,56.6,6.08,127.92]])
# predict=rf.predict(values)
# print(predict)

y_pred = rf.predict(x_test)


# Calculate the accuracy of the model
rf_accuracy = rf.score(x_test, y_test)

# print(rf_accuracy)

pickle.dump(rf,open('crop.pkl','wb'))
# model = pickle.load(open('crop.pkl','rb'))




