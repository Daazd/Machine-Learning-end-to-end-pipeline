import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv('data.csv')

# Explore and clean data (if needed)


# Split the data into features(X) and target(y)

X = data.drop(columns=['target', axis=1])
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_predict= model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_predict)
print(f'Accuracy: {accuracy}')
```


