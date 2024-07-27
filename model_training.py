#model training

#train classifier

import pickle

from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
import numpy as np


data_dict = pickle.load(open('./data.pickle', 'rb'))
data = []
labels = []
for i in range(len(data_dict['data'])):
    if len(data_dict['data'][i]) == 42: #part of selective training - pre-processing
        data.append(data_dict['data'][i])
        labels.append(data_dict['labels'][i])

data = np.array(data)
labels = np.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
model = RandomForestClassifier()
model.fit(x_train, y_train)
# model = KNeighborsClassifier()
# model.fit(x_train,y_train)
# model = LogisticRegression()
# model.fit(x_train,y_train)
# model = SVC()
# model.fit(x_train,y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print('{}% of samples were classified correctly !'.format(score * 100))

f = open('model_svm2.p', 'wb')
pickle.dump({'model': model}, f)
f.close()