# Step 4: Ordinal Encoding

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv('data.csv')

ordinal_cols = ['safety']
categories_order = [['low', 'med', 'high']]

oe = OrdinalEncoder(categories=categories_order)
df['safety_ord'] = oe.fit_transform(df[['safety']])
print("\n")
print(df[['safety', 'safety_ord']].head())
