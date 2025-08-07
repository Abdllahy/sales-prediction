import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

def detailed_test():
    print("=== DETAILED MODEL TESTING ===")
    print(f"Model type: {type(model)}")
    
    # Test with the exact same features as your Streamlit app
    print("\n--- Testing with Streamlit default values ---")
    
    # Default Streamlit values: [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    default_features = [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    default_pred = model.predict([default_features])
    print(f"Default features: {default_features}")
    print(f"Default prediction: {default_pred[0]}")
    
    # Test with different customer counts
    print("\n--- Testing different customer counts ---")
    for customers in [0, 100, 500, 1000, 2000]:
        features = [1, customers, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"Customers: {customers} -> Prediction: {pred[0]:.2f}")
    
    # Test with different promo values
    print("\n--- Testing different promo values ---")
    for promo in [0.0, 0.25, 0.5, 0.75, 1.0]:
        features = [1, 0, promo, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"Promo: {promo} -> Prediction: {pred[0]:.2f}")
    
    # Test with different days of week
    print("\n--- Testing different days of week ---")
    for day in range(1, 8):
        features = [day, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"Day {day} -> Prediction: {pred[0]:.2f}")
    
    # Test with different store types
    print("\n--- Testing different store types ---")
    store_configs = [
        ("a", [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]),
        ("b", [1, 0, 0.0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]),
        ("c", [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0]),
        ("d", [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0])
    ]
    
    for store_type, features in store_configs:
        pred = model.predict([features])
        print(f"Store type {store_type} -> Prediction: {pred[0]:.2f}")
    
    # Test with school holiday
    print("\n--- Testing school holiday ---")
    for school_holiday in [0, 1]:
        features = [1, 0, 0.0, school_holiday, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"School holiday: {school_holiday} -> Prediction: {pred[0]:.2f}")
    
    # Test with promo2
    print("\n--- Testing promo2 ---")
    for promo2 in [0, 1]:
        features = [1, 0, 0.0, 0, 0, promo2, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"Promo2: {promo2} -> Prediction: {pred[0]:.2f}")
    
    # Test with competition distance
    print("\n--- Testing competition distance ---")
    for distance in [0, 100, 500, 1000, 5000]:
        features = [1, 0, 0.0, 0, distance, 0, 1, 1, 0, 0, 0, 0, 0]
        pred = model.predict([features])
        print(f"Competition distance: {distance} -> Prediction: {pred[0]:.2f}")

if __name__ == "__main__":
    detailed_test() 