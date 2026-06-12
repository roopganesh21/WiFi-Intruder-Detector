import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    # Load dataset
    dataset_path = os.path.join('dataset', 'wifi_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run feature extraction first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Drop source_file (non-feature column)
    if 'source_file' in df.columns:
        df = df.drop(columns=['source_file'])
        
    # Ensure model and docs/images directories exist
    os.makedirs('model', exist_ok=True)
    os.makedirs(os.path.join('docs', 'images'), exist_ok=True)
    
    # Encode the label column
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])
    
    # Save the label encoder
    encoder_path = os.path.join('model', 'label_encoder.pkl')
    joblib.dump(le, encoder_path)
    print(f"Saved LabelEncoder to {encoder_path}")
    
    # Define features and target
    X = df.drop(columns=['label'])
    y = df['label']
    
    # Print the list of feature columns being used
    feature_cols = list(X.columns)
    print("\nFeature columns being used:")
    for col in feature_cols:
        print(f"- {col}")
        
    # Split the data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nData split summary:")
    print(f"- Training samples: {len(X_train)}")
    print(f"- Test samples: {len(X_test)}")
    
    # Train RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)
    
    print("\n" + "="*50)
    print(" MODEL EVALUATION ".center(50, "="))
    print("="*50)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix printed as a formatted table
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    print("\nConfusion Matrix:")
    print(cm_df.to_string())
    print("="*50)
    
    # Save confusion matrix heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 14})
    plt.title("Confusion Matrix — Test Set", fontsize=14, pad=15)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    confusion_matrix_path = os.path.join('docs', 'images', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix heatmap to {confusion_matrix_path}")
    
    # Print the top 5 most important features
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 5 Most Important Features:")
    for i in range(min(5, len(feature_cols))):
        idx = indices[i]
        print(f"{i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")
        
    # Save the trained model
    model_path = os.path.join('model', 'device_classifier.pkl')
    joblib.dump(clf, model_path)
    print(f"\nSaved trained model to: {model_path}")

if __name__ == "__main__":
    main()
