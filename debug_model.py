import pickle
import numpy as np

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

def test_model():
    print("Testing model with different inputs...")
    
    # Test case 1: Default values
    features1 = [1, 0, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    prediction1 = model.predict([features1])
    print(f"Test 1 - Default values: {prediction1[0]}")
    
    # Test case 2: High customer count
    features2 = [1, 1000, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    prediction2 = model.predict([features2])
    print(f"Test 2 - High customers (1000): {prediction2[0]}")
    
    # Test case 3: With promo
    features3 = [1, 0, 0.5, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    prediction3 = model.predict([features3])
    print(f"Test 3 - With promo (50%): {prediction3[0]}")
    
    # Test case 4: Different store type
    features4 = [1, 0, 0.0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
    prediction4 = model.predict([features4])
    print(f"Test 4 - Store type b: {prediction4[0]}")
    
    # Test case 5: Weekend with high customers
    features5 = [7, 500, 0.0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
    prediction5 = model.predict([features5])
    print(f"Test 5 - Weekend with 500 customers: {prediction5[0]}")
    
    # Check if all predictions are the same
    predictions = [prediction1[0], prediction2[0], prediction3[0], prediction4[0], prediction5[0]]
    if len(set(predictions)) == 1:
        print("\n⚠️  WARNING: All predictions are identical! The model may not be working correctly.")
        print("Possible issues:")
        print("1. Model was trained on a very small dataset")
        print("2. Model is overfitting to a single value")
        print("3. Feature scaling issues")
        print("4. Model type issue (e.g., using a classifier instead of regressor)")
    else:
        print(f"\n✅ Model is working correctly. Predictions vary from {min(predictions):.2f} to {max(predictions):.2f}")
    
    # Print model info
    print(f"\nModel type: {type(model)}")
    print(f"Model parameters: {model.get_params() if hasattr(model, 'get_params') else 'No parameters method'}")

if __name__ == "__main__":
    test_model() 