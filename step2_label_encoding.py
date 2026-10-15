# Step 2: Label Encoding

import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data.csv')

le = LabelEncoder()

df['class_encoded'] = le.fit_transform(df['class'])

print("Class labels mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[['class', 'class_encoded']].head())
