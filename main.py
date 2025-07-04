Creating a Python program to optimize waste management processes using predictive analytics involves data collection, data preprocessing, model training, and prediction phases. Below is a simplified example of such a program, assuming that a dataset related to waste management is available. The program includes comments and basic error handling to guide you through its functionality.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Function to load dataset
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return data
    except FileNotFoundError as e:
        logging.error("File not found. Please check the file path.")
        raise
    except pd.errors.EmptyDataError as e:
        logging.error("File is empty. Please provide a valid dataset.")
        raise
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


# Function to preprocess data
def preprocess_data(df):
    try:
        # Example preprocessing steps
        df.fillna(method='ffill', inplace=True)
        
        # Convert categorical data into numeric data if needed
        df = pd.get_dummies(df)
        
        logging.info("Data preprocessing completed.")
        
        # Split features and target
        X = df.drop('target_column', axis=1)  # Replace 'target_column' with actual target column name
        y = df['target_column']
        
        return X, y
    except KeyError as e:
        logging.error(f"Column not found: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during preprocessing: {e}")
        raise


# Function to train predictive model
def train_model(X, y):
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        
        logging.info(f"Model trained successfully with MSE: {mse:.2f}")
        return model
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}")
        raise


# Function to make predictions
def make_predictions(model, X_new):
    try:
        predictions = model.predict(X_new)
        logging.info(f"Predictions made: {predictions}")
        return predictions
    except ValueError as e:
        logging.error(f"An error occurred with the input data for prediction: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred while making predictions: {e}")
        raise


def main():
    # Replace with the actual path to your dataset
    filepath = 'waste_management_data.csv'  

    try:
        # Load and preprocess dataset
        data = load_data(filepath)
        X, y = preprocess_data(data)

        # Train model
        model = train_model(X, y)

        # Example: Make a prediction with new data
        # Replace this with actual new data
        new_data = np.array([[1, 0, 0, 1, 5]])  # Example data; replace with actual feature values
        make_predictions(model, new_data)

    except Exception as e:
        logging.error(f"A critical error occurred: {e}")

if __name__ == '__main__':
    main()
```

### Key Points:

1. **Data Handling:** 
   - The program attempts to load data from a CSV file using `pandas`.
   - It handles common errors such as file not found or the file being empty.

2. **Preprocessing:** 
   - The program performs basic preprocessing like forward filling missing values and converting categorical data to numerical using one-hot encoding.

3. **Model Training:** 
   - A Random Forest model is trained on the preprocessed data.
   - Errors during the model training are captured and logged.

4. **Prediction:** 
   - Make predictions with new data examples after training the model.
   - Includes error handling for input data issues.

5. **Logging and Comments:** 
   - Logging is used extensively throughout the program to provide information about the program’s execution and potential issues.

Please note that this program provides a foundational structure. Each section, especially data preprocessing and model particulars, should be tailored to the specifics of the dataset and problem in real scenarios.