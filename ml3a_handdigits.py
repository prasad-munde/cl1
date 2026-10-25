import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
 
# Step 2: Load the handwritten digits dataset 
digits = datasets.load_digits() 
 
# Features and target 
X = digits.data 
y = digits.target 
 
print("Dataset shape:", X.shape) 
print("Target shape:", y.shape) 
 
# Step 3: Display some sample images 
plt.figure(figsize=(8, 4)) 
 
for i in range(10): 
    plt.subplot(2, 5, i + 1) 
    plt.imshow(digits.images[i], cmap="gray") 
    plt.title("Digit: " + str(digits.target[i])) 
    plt.axis("off") 
 
plt.tight_layout() 
plt.show() 
 
# Step 4: Split the dataset into training and testing data 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42 
) 
 
print("Training data:", X_train.shape) 
print("Testing data:", X_test.shape) 
 
# Step 5: Create the SVM model 
model = SVC(kernel="rbf") 
 
# Step 6: Train the model 
model.fit(X_train, y_train) 
 
# Step 7: Predict the test data 
y_pred = model.predict(X_test) 
 
# Step 8: Calculate accuracy 
accuracy = accuracy_score(y_test, y_pred) 
 
print("\nAccuracy:", accuracy) 
 
# Step 9: Classification report 
print("\nClassification Report:") 
print(classification_report(y_test, y_pred)) 
 
# Step 10: Confusion matrix 
cm = confusion_matrix(y_test, y_pred) 
 
print("\nConfusion Matrix:") 
print(cm) 
 
# Step 11: Display some predictions 
plt.figure(figsize=(10, 5)) 
 
for i in range(10): 
    plt.subplot(2, 5, i + 1) 
    plt.imshow(X_test[i].reshape(8, 8), cmap="gray") 
    plt.title("Actual: " + str(y_test[i]) + 
              "\nPredicted: " + str(y_pred[i])) 
    plt.axis("off") 
 
plt.tight_layout() 
plt.show()