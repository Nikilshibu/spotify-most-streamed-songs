# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset, handle potential delimiter issues, and skip bad lines
data = pd.read_csv('Spotify Most Streamed Songs.csv', delimiter=',', engine='python', on_bad_lines='skip')

# Clean column names (remove tabs, extra spaces)
data.columns = data.columns.str.replace('\t', '').str.strip()

# Convert 'streams' column to numeric and drop rows with NaN values
data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
data.dropna(subset=['streams'], inplace=True)

# Print the cleaned column names (Optional)
print(data.columns)

# Select features and the target variable for linear regression
features = ['released_year', 'artist_count', 'in_spotify_playlists']
X = data[features]
y = data['streams']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Streams')
plt.ylabel('Predicted Streams')
plt.title('Actual vs Predicted Streams')
plt.show()

# Optional: Visualize the residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.title('Distribution of Residuals')
plt.show()
