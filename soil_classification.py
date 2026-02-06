import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


# Load and prepare the data
def load_and_prepare_data(url):
    print("Downloading data from UCI Repository...")

    # Define column names based on the dataset description
    columns = [
        "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
        "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points"
    ]

    # Add wilderness area columns
    for i in range(1, 5):
        columns.append(f"Wilderness_Area_{i}")

    # Add soil type columns
    for i in range(1, 41):
        columns.append(f"Soil_Type_{i}")

    # Add target column
    columns.append("Cover_Type")

    # Load the data
    data = pd.read_csv(url, header=None, names=columns)
    print(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns.")

    return data

# Process soil type columns to create a single target variable
def preprocess_target(data):
    print("\nProcessing soil types...")

    # Identify soil type columns
    soil_columns = [c for c in data.columns if "Soil_Type" in c]

    # Create a new target column based on the soil type columns
    data['Soil_Target'] = data[soil_columns].idxmax(axis=1).apply(lambda x: int(x.split('_')[-1]))

    # Drop the original soil type columns and the original target column
    X = data.drop(columns=soil_columns + ['Soil_Target'])
    Y = data['Soil_Target']

    return X, Y

# Train a Random Forest Classifier
def train_model(X, Y):
    # Use stratified sampling to maintain class distribution in train/test split
    print("\nSplitting data into training and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Print the size of training and test sets
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Train the Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, Y_train)

    return clf, X_test, Y_test

# Evaluate the model on the test set
def evaluate_model(clf, X_test, Y_test):
    # Predict on the test set
    print("\nPredicting on test set...")
    Y_pred = clf.predict(X_test)

    # Calculate and print accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print(f"Model accuracy: {100*accuracy:.2f}%")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(Y_test, Y_pred, zero_division=0))

    return Y_test, Y_pred

# Main execution
if __name__ == "__main__":
    # URL for the Covertype dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"

    # Wrap the main execution in a try-except block to handle potential errors gracefully
    try:
        # Load and prepare the data, preprocess the target variable, train the model, and evaluate it
        data = load_and_prepare_data(data_url)
        X, Y = preprocess_target(data)
        model, X_test, Y_test = train_model(X, Y)
        evaluate_model(model, X_test, Y_test)

        # Make a sample prediction to demonstrate how to use the trained model
        print("\nMaking a sample prediction...")
        sample = X_test.iloc[0].to_frame().T
        prediction = model.predict(sample)
        true_value = Y_test.iloc[0]
        print(f"Predicted Soil Type: {prediction[0]}, True Soil Type: {true_value}")
        
    except Exception as e:
        # Catch any exceptions that occur during the execution and print an error message
        print(f"An error occurred: {e}")