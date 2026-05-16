import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Optional for Google Colab download
from google.colab import files

# Load dataset
df = pd.read_csv('Placement1.csv')

# Convert target values
df['Placed'] = df['Placed'].map({'Yes': 1, 'No': 0})

# Features and target
X = df[['CGPA', 'Internships']]
y = df['Placed']

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X, y)

# Save model
pickle.dump(model, open('placement_model.pkl', 'wb'))

print("Model trained and saved successfully!")

# Download model automatically
files.download('placement_model.pkl')