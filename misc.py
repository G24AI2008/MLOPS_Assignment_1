import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None) 
    
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
		'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
      
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Mean Squared Error (MSE): {mse:.4f}")
    return mse
def run_training_pipeline(model_class):
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Training model: {model_class.__name__}...")
    model = model_class()
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    mse = evaluate_model(model, X_test, y_test)
    
    return model, (X_train, X_test, y_train, y_test), mse

