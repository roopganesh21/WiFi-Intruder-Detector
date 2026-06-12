# WiFi Intruder Detector

A machine learning-based system to detect intruder devices on a WiFi network using packet captures (.pcapng).

## Project Structure

- `data/`: Raw network packet captures (`.pcapng` files).
- `dataset/`: Preprocessed dataset generated from packet captures (ignored by git).
- `scripts/`: Python scripts for data extraction, preprocessing, model training, and inference.
- `model/`: Trained machine learning model binaries and configuration files.
- `dashboard/`: Web application/dashboard for visualizing detection results and network activity.
- `docs/`: Documentation and assets.
  - `docs/images/`: Images and diagrams for documentation.
