import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

class DogPricePredictor:
    def __init__(self):
        """
        Initialize the Dog Price Predictor with logging and default configurations
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Model and preprocessing components
        self.model = None
        self.preprocessor = None
        self.label_encoders = {}
        
    def create_comprehensive_dataset(self):
        """
        Create a more comprehensive synthetic dataset for dog price prediction
        """
        # Expanded dataset with more features and variability
        data = {
            "Breed": ["Labrador", "Poodle", "Bulldog", "Beagle", "Husky", 
                      "German Shepherd", "Golden Retriever", "Chihuahua", "Border Collie"],
            "Location": ["Urban", "Rural", "Urban", "Suburban", "Urban", 
                         "Suburban", "Rural", "Urban", "Rural"],
            "Age (Months)": [12, 8, 36, 24, 18, 15, 10, 6, 20],
            "Gender": ["Male", "Female", "Male", "Female", "Male", 
                       "Female", "Male", "Female", "Male"],
            "Vaccination Status": ["Complete", "Incomplete", "Complete", "Complete", "Incomplete", 
                                   "Complete", "Incomplete", "Complete", "Complete"],
            "Pedigree Certification": ["Yes", "No", "Yes", "No", "Yes", 
                                       "Yes", "No", "No", "Yes"],
            "Size": ["Large", "Small", "Medium", "Medium", "Large", 
                     "Large", "Large", "Small", "Medium"],
            "Training Level": ["Basic", "None", "Advanced", "Basic", "Advanced", 
                               "Advanced", "Basic", "None", "Advanced"],
            "Health Screening": ["Comprehensive", "Basic", "Basic", "None", "Comprehensive", 
                                 "Comprehensive", "Basic", "None", "Comprehensive"],
            "Parent Champion Status": ["Yes", "No", "No", "No", "Yes", 
                                       "Yes", "No", "No", "Yes"],
            "Demand": [80, 60, 90, 70, 85, 95, 75, 50, 88],  # Demand score out of 100
            "Price (KES)": [10000, 50000, 80000, 90000, 110000, 
                            130000, 95000, 150000, 115000],  # Target variable
        }
        return pd.DataFrame(data)
    
    def prepare_preprocessing_pipeline(self, X):
        """
        Create a preprocessing pipeline with handling for categorical and numerical features
        """
        # Identify column types
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Preprocessing for numerical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Preprocessing for categorical data
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor
    
    def train_model(self, X, y):
        """
        Train the Random Forest Regressor with cross-validation
        """
        # Prepare preprocessing pipeline
        self.preprocessor = self.prepare_preprocessing_pipeline(X)
        
        # Create the full pipeline
        model_pipeline = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=100, 
                random_state=42, 
                max_depth=10
            ))
        ])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Fit the model
        model_pipeline.fit(X_train, y_train)
        
        # Predictions and evaluation
        y_pred = model_pipeline.predict(X_test)
        
        # Detailed model evaluation
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
        
        self.logger.info(f"Model Performance Metrics:")
        self.logger.info(f"Mean Absolute Error: {mae}")
        self.logger.info(f"Mean Squared Error: {mse}")
        self.logger.info(f"R-squared Score: {r2}")
        self.logger.info(f"Cross-Validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Save the entire pipeline
        self.model = model_pipeline
        return model_pipeline
    
    def predict_prices(self, new_data):
        """
        Predict dog prices for new data
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Predict prices
            predictions = self.model.predict(new_data)
            return predictions
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def save_model(self, filepath='dog_price_predictor.pkl'):
        """
        Save the entire model pipeline
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            self.logger.info(f"Model saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath='dog_price_predictor.pkl'):
        """
        Load the entire model pipeline
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.logger.info(f"Model loaded from {filepath}")
            return self.model
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None

def main():
    # Create predictor instance
    predictor = DogPricePredictor()
    
    # Create dataset
    dataset = predictor.create_comprehensive_dataset()
    print("Dataset Preview:\n", dataset)
    
    # Separate features and target
    X = dataset.drop('Price (KES)', axis=1)
    y = dataset['Price (KES)']
    
    # Train the model
    trained_model = predictor.train_model(X, y)
    
    # Save the model
    predictor.save_model()
    
    # Example new data for prediction
    new_dog_data = pd.DataFrame({
        "Breed": ["German Shepherd", "Chihuahua"],
        "Location": ["Urban", "Rural"],
        "Age (Months)": [12, 6],
        "Gender": ["Male", "Female"],
        "Vaccination Status": ["Complete", "Incomplete"],
        "Pedigree Certification": ["Yes", "No"],
        "Size": ["Large", "Small"],
        "Training Level": ["Advanced", "None"],
        "Health Screening": ["Comprehensive", "Basic"],
        "Parent Champion Status": ["Yes", "No"],
        "Demand": [90, 55]
    })
    
    # Make predictions
    predicted_prices = predictor.predict_prices(new_dog_data)
    
    if predicted_prices is not None:
        new_dog_data['Predicted Price (KES)'] = predicted_prices
        print("\nNew Data with Predictions:\n", new_dog_data)

if __name__ == "__main__":
    main()