# Step 5: Putting Data Together with ColumnTransformer

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# Load data from CSV file
df = pd.read_csv('data.csv')

ordinal_features = ['safety']
ordinal_categories = [['low', 'med', 'high']]

nominal_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot']

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ('nom', OneHotEncoder(sparse_output=False), nominal_features)
    ]
)

features = ordinal_features + nominal_features
X = df[features]
X_prepared = preprocessor.fit_transform(X)
print("\n")


print("Transformed shape:", X_prepared.shape)
print("\n")