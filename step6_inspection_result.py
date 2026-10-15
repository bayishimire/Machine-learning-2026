

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load data 
df = pd.read_csv('data.csv')

# Encode target (from Step 2) 
le = LabelEncoder()
df['class_encoded'] = le.fit_transform(df['class'])

# Build preprocessor (from Step 5) 
ordinal_features    = ['safety']
ordinal_categories  = [['low', 'med', 'high']]

nominal_features = ['buying', 'maint', 'doors', 'persons', 'lug_boot']

preprocessor = ColumnTransformer(
    transformers=[
        ('ord', OrdinalEncoder(categories=ordinal_categories), ordinal_features),
        ('nom', OneHotEncoder(sparse_output=False), nominal_features)
    ]
)

features   = ordinal_features + nominal_features
X          = df[features]
X_prepared = preprocessor.fit_transform(X)

# Build final DataFrame (Step 6) 
final_df = pd.DataFrame(
    np.hstack([X_prepared, df[['class_encoded']].values]),
    columns=list(preprocessor.get_feature_names_out()) + ['class_encoded']
)
print("\n")
print(final_df.head())
