import os
import glob
import pandas as pd
import numpy as np
from scapy.all import rdpcap, TCP, UDP

def extract_features_from_pcap(file_path):
    """
    Extracts features from a given .pcapng file using Scapy.
    """
    packets = rdpcap(file_path)
    packet_count = len(packets)
    
    if packet_count == 0:
        return {
            'packet_count': 0,
            'avg_packet_size': 0.0,
            'max_packet_size': 0,
            'min_packet_size': 0,
            'tcp_count': 0,
            'udp_count': 0,
            'total_bytes': 0,
            'tcp_udp_ratio': 0.0,
            'avg_inter_arrival_time': 0.0
        }
    
    sizes = [len(pkt) for pkt in packets]
    total_bytes = sum(sizes)
    avg_packet_size = float(np.mean(sizes))
    max_packet_size = int(np.max(sizes))
    min_packet_size = int(np.min(sizes))
    
    tcp_count = sum(1 for pkt in packets if pkt.haslayer(TCP))
    udp_count = sum(1 for pkt in packets if pkt.haslayer(UDP))
    
    tcp_udp_sum = tcp_count + udp_count
    tcp_udp_ratio = float(tcp_count) / tcp_udp_sum if tcp_udp_sum > 0 else 0.0
    
    # Calculate average inter-arrival time in milliseconds
    timestamps = [float(pkt.time) for pkt in packets]
    if len(timestamps) > 1:
        # Consecutive packet timestamp differences
        diffs = [(timestamps[i + 1] - timestamps[i]) * 1000.0 for i in range(len(timestamps) - 1)]
        avg_inter_arrival_time = float(np.mean(diffs))
    else:
        avg_inter_arrival_time = 0.0
        
    return {
        'packet_count': packet_count,
        'avg_packet_size': avg_packet_size,
        'max_packet_size': max_packet_size,
        'min_packet_size': min_packet_size,
        'tcp_count': tcp_count,
        'udp_count': udp_count,
        'total_bytes': total_bytes,
        'tcp_udp_ratio': tcp_udp_ratio,
        'avg_inter_arrival_time': avg_inter_arrival_time
    }

def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    
    # Ensure dataset directory exists
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Find all .pcapng files in the data directory
    pcap_files = glob.glob(os.path.join(data_dir, '*.pcapng'))
    
    if not pcap_files:
        print(f"No .pcapng files found in {data_dir}")
        return
        
    features_list = []
    processed_count = 0
    
    for file_path in pcap_files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        try:
            features = extract_features_from_pcap(file_path)
            
            # Determine label
            filename_lower = filename.lower()
            if 'my_phone' in filename_lower:
                label = 'MyPhone'
            elif 'friend_phone' in filename_lower:
                label = 'FriendPhone'
            else:
                label = 'Unknown'
                
            features['label'] = label
            features['source_file'] = filename
            features_list.append(features)
            processed_count += 1
            
        except Exception as e:
            print(f"Warning: Failed to read {filename}. Error: {e}")
            continue
            
    if not features_list:
        print("No features extracted. Dataset was not created.")
        return
        
    # Build DataFrame
    df = pd.DataFrame(features_list)
    
    # Export to CSV
    output_path = os.path.join(dataset_dir, 'wifi_dataset.csv')
    df.to_csv(output_path, index=False)
    
    # Print summary
    print("\n--- Summary ---")
    print(f"Files successfully processed: {processed_count}")
    print(f"Total rows in dataset: {len(df)}")
    print("\nClass distribution:")
    print(df['label'].value_value_counts() if hasattr(df['label'], 'value_value_counts') else df['label'].value_counts())

if __name__ == "__main__":
    main()
