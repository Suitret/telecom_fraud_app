import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load Data
df = pd.read_csv('data/realtime_cdr_log.csv')

# Simple Preprocessing (Mapping the original logic)
le = LabelEncoder()
df['call_type'] = le.fit_transform(df['call_type'])
df['location_origin'] = le.fit_transform(df['location_origin'])

# Features based on the dataset snippet
features = ['duration_sec', 'call_type', 'is_night_call', 'location_origin']
X = df[features]
y = df['transaction_status'].apply(lambda x: 1 if x == 'Fraudulent' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("Model trained and saved!")
