# Step 3: One-Hot Encoding

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('data.csv')

categorical_cols = ['buying', 'maint',
                    'doors', 'persons', 'lug_boot', 'safety']
ohe = OneHotEncoder(sparse_output=False)

ohe_array = ohe.fit_transform(df[categorical_cols])
print("OHE feature names:", ohe.get_feature_names_out(categorical_cols))

ohe_df = pd.DataFrame(
    ohe_array, columns=ohe.get_feature_names_out(categorical_cols))
df_ohe = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
print("\n" )
print(df_ohe.head())