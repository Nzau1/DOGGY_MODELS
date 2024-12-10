import pandas as pd
import numpy as np
import random

class DogDatasetGenerator:
    def __init__(self, num_samples=1000):
        """
        Initialize the dataset generator with predefined categories
        """
        self.num_samples = num_samples
        
        # Comprehensive breed categories
        self.breeds = [
            "German Shepherd", "Chihuahua", "Labrador Retriever", 
            "Golden Retriever", "Poodle", "Bulldog", "Beagle", 
            "Husky", "Border Collie", "Rottweiler", "Doberman", 
            "Shih Tzu", "Australian Shepherd", "Great Dane", 
            "Boxer", "Dachshund", "Corgi", "Maltese", 
            "Bernese Mountain Dog", "Jack Russell Terrier"
        ]
        
        # Locations with probabilities
        self.locations = [
            "Urban", "Rural", "Suburban", "Urban", "Rural", 
            "Urban", "Suburban", "Rural"
        ]
        
        # Categorical feature mappings
        self.categorical_features = {
            "Gender": ["Male", "Female"],
            "Vaccination Status": ["Complete", "Incomplete"],
            "Pedigree Certification": ["Yes", "No"],
            "Size": ["Large", "Medium", "Small"],
            "Training Level": ["Advanced", "Intermediate", "Basic", "None"],
            "Health Screening": ["Comprehensive", "Basic", "None"],
            "Parent Champion Status": ["Yes", "No"]
        }
    
    def generate_price_logic(self, breed, size, training_level, health_screening, 
                              pedigree_cert, parent_champion, demand):
        """
        Generate a price based on various dog characteristics
        """
        # Base price by breed
        breed_base_prices = {
            "German Shepherd": 100000, "Chihuahua": 50000, 
            "Labrador Retriever": 80000, "Golden Retriever": 90000, 
            "Poodle": 110000, "Bulldog": 70000, "Beagle": 60000,
            "Husky": 85000, "Border Collie": 75000, "Rottweiler": 95000,
            "Doberman": 100000, "Shih Tzu": 55000, "Australian Shepherd": 85000,
            "Great Dane": 120000, "Boxer": 90000, "Dachshund": 65000,
            "Corgi": 70000, "Maltese": 50000, "Bernese Mountain Dog": 130000,
            "Jack Russell Terrier": 60000
        }
        
        # Price modifiers
        size_modifier = {"Large": 1.2, "Medium": 1.0, "Small": 0.8}
        training_modifier = {
            "Advanced": 1.3, "Intermediate": 1.1, 
            "Basic": 1.0, "None": 0.9
        }
        health_modifier = {
            "Comprehensive": 1.2, "Basic": 1.1, "None": 0.9
        }
        
        # Calculate base price
        base_price = breed_base_prices.get(breed, 75000)
        
        # Apply modifiers
        price = base_price * size_modifier.get(size, 1.0) * \
                training_modifier.get(training_level, 1.0) * \
                health_modifier.get(health_screening, 1.0)
        
        # Add premium for pedigree and champion status
        if pedigree_cert == "Yes":
            price *= 1.15
        if parent_champion == "Yes":
            price *= 1.2
        
        # Adjust based on demand (1-10 scale)
        price *= (1 + (demand - 5) * 0.05)
        
        return int(price + np.random.normal(0, price * 0.1))  # Add some randomness
    
    def generate_dataset(self):
        """
        Generate a comprehensive dog dataset
        """
        # Create empty lists to store data
        data = {
            "Breed": [], "Location": [], "Age (Months)": [], 
            "Gender": [], "Vaccination Status": [], 
            "Pedigree Certification": [], "Size": [], 
            "Training Level": [], "Health Screening": [], 
            "Parent Champion Status": [], "Demand": [], 
            "Price (KES)": []
        }
        
        # Generate random samples
        for _ in range(self.num_samples):
            # Randomly select features
            breed = random.choice(self.breeds)
            location = random.choice(self.locations)
            age = random.randint(2, 120)  # 2 to 120 months
            gender = random.choice(self.categorical_features["Gender"])
            vax_status = random.choice(self.categorical_features["Vaccination Status"])
            pedigree_cert = random.choice(self.categorical_features["Pedigree Certification"])
            size = random.choice(self.categorical_features["Size"])
            training_level = random.choice(self.categorical_features["Training Level"])
            health_screening = random.choice(self.categorical_features["Health Screening"])
            parent_champion = random.choice(self.categorical_features["Parent Champion Status"])
            demand = random.randint(1, 10)
            
            # Generate price
            price = self.generate_price_logic(
                breed, size, training_level, health_screening, 
                pedigree_cert, parent_champion, demand
            )
            
            # Store data
            data["Breed"].append(breed)
            data["Location"].append(location)
            data["Age (Months)"].append(age)
            data["Gender"].append(gender)
            data["Vaccination Status"].append(vax_status)
            data["Pedigree Certification"].append(pedigree_cert)
            data["Size"].append(size)
            data["Training Level"].append(training_level)
            data["Health Screening"].append(health_screening)
            data["Parent Champion Status"].append(parent_champion)
            data["Demand"].append(demand)
            data["Price (KES)"].append(price)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    def save_dataset(self, df, filename='dog_price_dataset.csv'):
        """
        Save the generated dataset to a CSV file
        """
        df.to_csv(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Dataset shape: {df.shape}")
        print("\nDataset Preview:")
        print(df.head())
        
        # Additional dataset insights
        print("\nDataset Insights:")
        print(df.describe())
        print("\nBreed Distribution:")
        print(df['Breed'].value_counts())

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Create dataset generator
    generator = DogDatasetGenerator(num_samples=1000)
    
    # Generate dataset
    dog_dataset = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset(dog_dataset)

if __name__ == "__main__":
    main()