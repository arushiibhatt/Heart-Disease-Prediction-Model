from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
import csv

train = pd.read_csv("heart_train.csv")
test = pd.read_csv("heart_test_cleaned.csv")
test = test.drop(['id'], axis=1)

feature_cols = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope']
X = train[feature_cols]
y = train.HeartDisease

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state = 16)
hgbc = HistGradientBoostingClassifier(max_bins=255, max_iter=1000)
hgbc.fit(X_train, y_train)
print(hgbc.score(X_test, y_test))

values = hgbc.predict(test)

"""Write ouput to csv file"""
with open('submission.csv', mode='w', newline='') as file: 
    writer = csv.writer(file) 
    writer.writerow(['output']) 
    for i in range(len(values)): 
        new_value = values[i]  
        writer.writerow([values[i]])  
