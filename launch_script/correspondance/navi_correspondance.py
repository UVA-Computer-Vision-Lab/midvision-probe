import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# List of backbones to evaluate
models = [
    "croco_vitb16",
    "barlowtwins_resnet50",
    "beit_v2_vitb16",
    "byol_resnet50",
    "clusterfit_resnet50",
    "deepcluster-v2-resnet50",
    "densecl_resnet50",
    "dino_b16",
    "eva_vitb16",
    "ibot_b16",
    "jigsaw_resnet50",
    "mae_b16",
    "maskfeat_vitb16",
    "milan_vitb16",
    "mocov2_resnet50",
    "mocov3_b14",
    "npid-plusplus_resnet50",
    "pirl_resnet50",
    "pixmlm_vitb16",
    "rotnet_resnet50",
    "simsiam_resnet50",
    "sela-v2_resnet50",
    "simclr_resnet50",
    "swav_resnet50",
    "npid_resnet50",
    "mocov3_resnet50",
    "dino_resnet50",
    "dinov2_b14",
    "dinov2_b14_reg",
    "dinov2_l14",
]

# Path to the project directory
project_directory = (
    "/p/openvocabdustr/probing_midlevel_vision/code/probing-mid-level-vision"
)

# Base command for running the evaluation script
base_command = "python evaluate_navi_correspondence.py +backbone={model}"


# Function to run an evaluation job for a specific model
def run_evaluation(model):
    try:
        # Prepare the command with the model name
        command = base_command.format(model=model)

        # Print the command to confirm it's correct
        print(f"Running command: {command}")

        # Execute the command in the specific directory
        subprocess.run(command, shell=True, check=True, cwd=project_directory)
        print(f"Completed evaluation: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to evaluate model {model}: {e}")


max_threads = 1

# Use ThreadPoolExecutor to run evaluations in parallel
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    # Submit evaluation tasks to the pool
    futures = [executor.submit(run_evaluation, model) for model in models]

    # Wait for the tasks to complete
    for future in as_completed(futures):
        try:
            future.result()  # This will raise an exception if the task failed
        except Exception as exc:
            print(f"An exception occurred: {exc}")
