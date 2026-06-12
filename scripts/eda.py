import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Set styling
    sns.set_theme(style="whitegrid")
    
    # Load dataset
    dataset_path = os.path.join('dataset', 'wifi_dataset.csv')
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run feature extraction first.")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Ensure docs/images/ exists
    images_dir = os.path.join('docs', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # Plot 1 — Class distribution bar chart
    plt.figure(figsize=(6, 4))
    # Count occurrences
    label_counts = df['label'].value_counts()
    ax = sns.barplot(x=label_counts.index, y=label_counts.values, hue=label_counts.index, palette='viridis', legend=False)
    plt.title("Class Distribution", fontsize=14, pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    # Add value labels on top of each bar
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'class_distribution.png'), dpi=150)
    plt.close()
    print("Saved: class_distribution.png")

    # Plot 2 — Packet count distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='packet_count', hue='label', kde=True, multiple='dodge', palette='Set2')
    plt.title("Packet Count Distribution by Class", fontsize=14, pad=15)
    plt.xlabel("Packet Count", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'packet_count_distribution.png'), dpi=150)
    plt.close()
    print("Saved: packet_count_distribution.png")

    # Plot 3 — Average packet size distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='avg_packet_size', hue='label', kde=True, multiple='dodge', palette='Set2')
    plt.title("Average Packet Size Distribution by Class", fontsize=14, pad=15)
    plt.xlabel("Average Packet Size (bytes)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'avg_packet_size_distribution.png'), dpi=150)
    plt.close()
    print("Saved: avg_packet_size_distribution.png")

    # Plot 4 — TCP vs UDP comparison
    avg_counts = df.groupby('label')[['tcp_count', 'udp_count']].mean().reset_index()
    melted_df = pd.melt(avg_counts, id_vars=['label'], value_vars=['tcp_count', 'udp_count'],
                        var_name='protocol', value_name='average_count')
    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted_df, x='label', y='average_count', hue='protocol', palette='Set1')
    plt.title("Average TCP vs UDP Count by Class", fontsize=14, pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Average Packet Count", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'tcp_udp_comparison.png'), dpi=150)
    plt.close()
    print("Saved: tcp_udp_comparison.png")

    # Plot 5 — Total bytes boxplot
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x='label', y='total_bytes', hue='label', palette='Pastel1', legend=False)
    plt.title("Total Bytes Transferred by Class", fontsize=14, pad=15)
    plt.xlabel("Class Label", fontsize=12)
    plt.ylabel("Total Bytes", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'total_bytes_boxplot.png'), dpi=150)
    plt.close()
    print("Saved: total_bytes_boxplot.png")

    # Plot 6 — Feature correlation heatmap
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_cols.corr(method='pearson')
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
                cbar_kws={'shrink': .8}, annot_kws={'size': 10})
    plt.title("Feature Correlation Heatmap", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'feature_correlation_heatmap.png'), dpi=150)
    plt.close()
    print("Saved: feature_correlation_heatmap.png")
    
    # Console output summary statistics
    print("\n" + "="*50)
    print(" SUMMARY STATISTICS FOR NUMERIC COLUMNS ".center(50, "="))
    print("="*50)
    print(df.describe().to_string())
    print("="*50)

if __name__ == "__main__":
    main()
