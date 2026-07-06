import csv
import time
import numpy as np


def compute_anomaly_rate(labels_array):
    """Calculates the percentage of anomalous/malicious traffic lines in the dataset."""
    anomalies_mask = labels_array != "normal"
    return np.mean(anomalies_mask) * 100


def main():
    durations = []
    src_bytes = []
    dst_bytes = []
    labels = []

    # 1. Read the dataset safely using try/except
    try:
        with open("nsl_kdd_dataset.csv", mode="r", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            # Clean the fieldnames by removing any accidental spaces
            reader.fieldnames = [
                name.strip() for name in reader.fieldnames if name
            ]

            # Automatically detect if the target column is named 'class' or 'label'
            class_col = None
            for possible_name in ["class", "label"]:
                if possible_name in reader.fieldnames:
                    class_col = possible_name
                    break

            if not class_col:
                raise KeyError(
                    f"Could not find a 'class' or 'label' column. Available columns are: {reader.fieldnames}"
                )

            # 2. Extract crucial columns for anomaly calculations
            for row in reader:
                # Clean row keys to avoid whitespace issues
                clean_row = {k.strip(): v for k, v in row.items() if k}
                durations.append(float(clean_row["duration"]))
                src_bytes.append(float(clean_row["src_bytes"]))
                dst_bytes.append(float(clean_row["dst_bytes"]))
                labels.append(clean_row[class_col].strip())

    except FileNotFoundError:
        print(
            "Error: 'nsl_kdd_dataset.csv' not found. Please ensure it is in the same folder."
        )
        return
    except KeyError as e:
        print(f"Error: Missing column configuration: {e}")
        return
    except ValueError as e:
        print(
            f"Error: Found non-numeric data in numerical columns. details: {e}"
        )
        return

    # 3. Convert parsed lists into a 2D NumPy Array for numerical features
    features_matrix = np.column_stack((durations, src_bytes, dst_bytes))
    labels_array = np.array(labels)

    print("==================================================")
    print("      SECURE NET: ANOMALY DETECTION ENGINE       ")
    print("==================================================\n")

    # Basic Descriptive Statistics
    means = features_matrix.mean(axis=0)
    maxes = features_matrix.max(axis=0)

    print("--- Network Traffic Baselines ---")
    print(
        f"Average Connection Duration : {means[0]:.2f} sec | Max: {maxes[0]} sec"
    )
    print(
        f"Average Outbound (Src Bytes): {means[1]:.2f} B   | Max: {maxes[1]} B"
    )
    print(
        f"Average Inbound (Dst Bytes) : {means[2]:.2f} B   | Max: {maxes[2]} B\n"
    )

    # 4. Vectorized Normalization (Min-Max Scaling)
    mins = features_matrix.min(axis=0)
    ranges = maxes - mins
    ranges[ranges == 0] = 1.0
    normalized_features = (features_matrix - mins) / ranges

    # 5. Anomaly Rate tracking using Boolean Masking
    anomaly_pct = compute_anomaly_rate(labels_array)
    print(f"--- Threat Analysis Assessment ---")
    print(f"Overall Network Threat Anomaly Rate: {anomaly_pct:.2f}%\n")

    # 6. Extract Most Severe Outliers/Anomalies using argmax
    biggest_src_burst_idx = np.argmax(features_matrix[:, 1])
    print("--- Critical Payload Outlier Alert ---")
    print(
        f"Highest Packet Spike Event: Row {biggest_src_burst_idx + 1}"
    )
    print(
        f"Payload Transmitted       : {features_matrix[biggest_src_burst_idx, 1]} Bytes"
    )
    print(
        f"Traffic Status Assertion  : Security Status -> [{labels_array[biggest_src_burst_idx].upper()}]\n"
    )

    # 7. Rank connections using argsort (Highest to Lowest)
    ranked_indices = np.argsort(features_matrix[:, 0])[::-1]

    print("--- Top 5 Longest Lasting Network Flows ---")
    print(f"{'Rank':<5} | {'Duration':<10} | {'Src Bytes':<12} | {'Class'}")
    print("-" * 45)
    for i in range(min(5, len(ranked_indices))):
        idx = ranked_indices[i]
        print(
            f"{i+1:<5} | {features_matrix[idx, 0]:<10.1f} | {features_matrix[idx, 1]:<12.0f} | {labels_array[idx]}"
        )

    # --- BONUS CHALLENGE: BENCHMARK PERFORMANCE ---
    large_scale_list = src_bytes * 1000
    large_scale_array = np.array(large_scale_list)

    t0 = time.time()
    py_transform = [x * 1.05 for x in large_scale_list]
    py_time = time.time() - t0

    t0 = time.time()
    np_transform = large_scale_array * 1.05
    np_time = time.time() - t0

    speed_ratio = py_time / np_time if np_time > 0 else 0
    print("\n==================================================")
    print("       VECTORIZATION PERFORMANCE BENCHMARK       ")
    print("==================================================")
    print(f"NumPy handled threat metrics {speed_ratio:.1f}x faster.")


if __name__ == "__main__":
    main()