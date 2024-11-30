import os
import numpy as np
from datasets import Features, Value, Sequence
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

# Define the dataset path and Parquet save path
dataset_path = "/p/openvocabdustr/probing_midlevel_vision/code/probing-mid-level-vision/data/nyuv2_test_processed"
parquet_save_path = "/p/openvocabdustr/probing_midlevel_vision/code/probing-mid-level-vision/data/nyu_geonet_hf_datasets_parquet/test"

# Ensure the Parquet directory exists
os.makedirs(parquet_save_path, exist_ok=True)

# Define the schema for both train and test
features = Features(
    {
        "image": Sequence(Value(dtype="uint8")),  # Variable-sized RGB image
        "depth": Sequence(Value(dtype="float32")),  # Variable-sized depth image
        "normal": Sequence(Value(dtype="float32")),  # Variable-sized normal image
        "segmentation": Sequence(
            Value(dtype="uint8")
        ),  # Variable-sized segmentation map
        "id2label": Value(dtype="string"),  # Store id2label as JSON string
        "room": Value(dtype="string"),  # Room metadata for test set
        "nyu_index": Value(dtype="int32"),  # NYU index metadata for test set
    }
)


def load_image(image_path):
    """
    Load an image from the provided path and return as a numpy array.
    """
    with Image.open(image_path) as img:
        return np.array(img)


def load_npy_file(npy_path):
    """
    Load an .npy file from the provided path and return the array.
    """
    return np.load(npy_path)


def load_npz_segmentation(npz_path):
    """
    Load a .npz file and return the segmentation array and id2label dictionary.
    """
    with np.load(npz_path, allow_pickle=True) as data:
        panoptic_map = data["panoptic_map"]
        id2label = data["id2label"].item()  # Convert object array to dict
        return panoptic_map, id2label


def convert_dict_numpy_to_python(d):
    """
    Recursively convert NumPy types in a dictionary to standard Python types.
    """
    for key, value in d.items():
        if isinstance(value, np.integer):
            d[key] = int(value)
        elif isinstance(value, np.floating):
            d[key] = float(value)
        elif isinstance(value, np.ndarray):
            d[key] = value.tolist()  # Convert NumPy array to list
        elif isinstance(value, dict):
            d[key] = convert_dict_numpy_to_python(
                value
            )  # Recursively convert nested dicts
    return d


def process_file(file_name, dataset_path, metadata=None):
    """
    Process a single file, loading image, depth, normal, segmentation, and optionally, metadata.
    Returns a dictionary with the data.
    """
    base_name = os.path.splitext(file_name)[0]

    try:
        # Load the image
        image = load_image(os.path.join(dataset_path, "images", file_name))

        # Replace `_image` in base_name with `_depth`, `_normal`, and `_segmentation` for the other files
        depth = load_npy_file(
            os.path.join(
                dataset_path, "depths", base_name.replace("_image", "_depth") + ".npy"
            )
        )
        normal = load_npy_file(
            os.path.join(
                dataset_path, "normals", base_name.replace("_image", "_norm") + ".npy"
            )
        )
        segmentation_npz_path = os.path.join(
            dataset_path, "segmentations", base_name + ".npz"
        )
        panoptic_map, id2label = load_npz_segmentation(segmentation_npz_path)
        id2label = convert_dict_numpy_to_python(id2label)
        id2label_str = json.dumps(id2label)

        # Prepare the record
        record = {
            "image": image.flatten().tolist(),
            "depth": depth.flatten().tolist(),
            "normal": normal.flatten().tolist(),
            "segmentation": panoptic_map.flatten().tolist(),
            "id2label": id2label_str,
        }

        # If metadata exists (for test set), include additional fields
        if metadata:
            record.update(metadata)

        return record

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None


def validate_record(record):
    """
    Validate if a record has all the required keys and if their shapes match the schema.
    """
    required_keys = ["image", "depth", "normal", "segmentation", "id2label"]

    # Check for missing keys
    for key in required_keys:
        if key not in record:
            print(f"Missing key in record: {key}")
            return False

    return True


def save_to_parquet(records, chunk_id, parquet_save_path):
    """
    Save a chunk of records to Parquet format, only valid records.
    """
    valid_records = [record for record in records if validate_record(record)]

    if valid_records:
        print(f"Saving chunk {chunk_id} with {len(valid_records)} records")

        # Prepare data in a format suitable for Arrow Table
        columns = {
            key: [record[key] for record in valid_records]
            for key in valid_records[0].keys()
        }

        # Create Arrow table and save to Parquet
        table = pa.table(columns, schema=features.arrow_schema)
        pq.write_table(
            table, os.path.join(parquet_save_path, f"dataset_chunk_{chunk_id}.parquet")
        )
    else:
        print(f"No valid records in chunk {chunk_id} to save.")


def process_in_parallel(
    dataset_path, parquet_save_path, num_workers=4, chunk_size=1000, metadata=None
):
    """
    Process the dataset in parallel using ProcessPoolExecutor and save to Parquet.
    """
    print("Listing image files...")
    image_files = sorted(os.listdir(os.path.join(dataset_path, "images")))
    total_files = len(image_files)
    print(f"Total image files: {total_files}")

    records = []
    chunk_id = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for file_name in image_files:
                future = executor.submit(
                    process_file, file_name, dataset_path, metadata
                )
                futures.append(future)

                if len(futures) >= num_workers:
                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            records.append(result)

                        futures.remove(future)
                        pbar.update(1)

                        if len(records) >= chunk_size:
                            save_to_parquet(records, chunk_id, parquet_save_path)
                            records = []
                            chunk_id += 1

            for future in as_completed(futures):
                result = future.result()
                if result:
                    records.append(result)
                pbar.update(1)

            if records:
                save_to_parquet(records, chunk_id, parquet_save_path)


if __name__ == "__main__":
    # Process the dataset in parallel and save to Parquet
    # Example for test set with metadata
    test_metadata = {"room": "test_room", "nyu_index": 1}  # For each test file
    process_in_parallel(
        dataset_path, parquet_save_path, num_workers=1, metadata=test_metadata
    )

    print(f"Dataset saved in Parquet format at {parquet_save_path}")
