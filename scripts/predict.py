import os
import joblib
import pandas as pd
import numpy as np

# Global variables for model and label encoder
model = None
label_encoder = None
feature_columns = [
    'packet_count',
    'avg_packet_size',
    'max_packet_size',
    'min_packet_size',
    'tcp_count',
    'udp_count',
    'total_bytes',
    'tcp_udp_ratio',
    'avg_inter_arrival_time'
]

def load_assets():
    """
    Loads the saved RandomForest model and LabelEncoder.
    Raises FileNotFoundError if assets are missing.
    """
    global model, label_encoder
    model_path = os.path.join('model', 'device_classifier.pkl')
    encoder_path = os.path.join('model', 'label_encoder.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        raise FileNotFoundError(
            "Required model files are missing. Please run 'python scripts/train_model.py' "
            "first to train the model and generate the assets."
        )
        
    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

def predict_device(features: dict) -> dict:
    """
    Accepts a dictionary of features, prepares it for prediction,
    predicts the device label and confidence, and handles unknown devices.
    """
    if model is None or label_encoder is None:
        load_assets()
        
    # Convert dictionary to DataFrame with the correct column order
    df_features = pd.DataFrame([features])[feature_columns]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(df_features)[0]
    
    # Identify maximum probability and its corresponding class index
    max_idx = np.argmax(probabilities)
    confidence = float(probabilities[max_idx])
    
    # Decode class name
    decoded_label = label_encoder.inverse_transform([max_idx])[0]
    
    # Check if confidence is below threshold (70%)
    is_unknown = confidence < 0.70
    predicted_label = "Unknown Device Detected" if is_unknown else decoded_label
    
    # Create dictionary mapping all class names to their probabilities
    all_probabilities = {}
    for idx, prob in enumerate(probabilities):
        class_name = label_encoder.inverse_transform([idx])[0]
        all_probabilities[class_name] = round(float(prob), 4)
        
    return {
        "predicted_label": predicted_label,
        "confidence": round(confidence, 4),
        "all_probabilities": all_probabilities,
        "is_unknown": is_unknown
    }

if __name__ == "__main__":
    # Test file paths
    dataset_path = os.path.join('dataset', 'wifi_dataset.csv')
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run feature extraction first.")
        exit(1)
        
    # Load assets
    try:
        load_assets()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
        
    # Read the dataset to obtain mean feature dictionaries
    df_csv = pd.read_csv(dataset_path)
    
    # 1. Test Case: MyPhone mean values
    my_phone_df = df_csv[df_csv['label'] == 'MyPhone'][feature_columns]
    test_my_phone = my_phone_df.mean().to_dict()
    
    # 2. Test Case: FriendPhone mean values
    friend_phone_df = df_csv[df_csv['label'] == 'FriendPhone'][feature_columns]
    test_friend_phone = friend_phone_df.mean().to_dict()
    
    # 3. Test Case: Unknown values (simulating random values)
    test_unknown = {col: 1.0 for col in feature_columns}
    
    # Execute predictions
    results = {
        "Test 1 — MyPhone (Mean Values)": predict_device(test_my_phone),
        "Test 2 — FriendPhone (Mean Values)": predict_device(test_friend_phone),
        "Test 3 — Unknown Device (Random Values)": predict_device(test_unknown)
    }
    
    # Display results clearly
    print("="*60)
    print(" PREDICTION DEMO RESULTS ".center(60, "="))
    print("="*60)
    for title, res in results.items():
        print(f"\n[ {title} ]")
        print(f"  Predicted Label   : {res['predicted_label']}")
        print(f"  Confidence Score  : {res['confidence']:.4f}")
        print(f"  Is Unknown Device : {res['is_unknown']}")
        print(f"  All Probabilities : {res['all_probabilities']}")
    print("="*60)
