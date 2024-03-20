import  pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dict = pickle.load(open('./Air Draw/data.pickle','rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier(max_features=42)

model.fit(x_train, y_train)

predict = model.predict(x_test)

score = accuracy_score(y_test, predict)

print(f'Score for the model is {score}')

f = open('./Air Draw/model.p', 'wb')
pickle.dump({'model':model,},f)
f.close()