# Assuming you have cleaned the data or used a method to handle errors
data = pd.read_csv('Spotify Most Streamed Songs.csv', delimiter=',', engine='python', on_bad_lines='skip')

# Continue with logistic regression code after loading the dataset successfully
data.columns = data.columns.str.replace('\t', '').str.strip()
print(data.columns)

# Assume we have a 'Hit' column
features = ['released_year', 'artist_count', 'in_spotify_playlists']
X = data[features]
y = data['streams']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Visualize the Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
