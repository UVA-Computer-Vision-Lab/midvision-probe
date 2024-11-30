import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

models = [
    "mocov3_resnet50",
    "barlowtwins_resnet50",
    "beit-v2_vitb16",
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
    "dino_resnet50",
]

project_directory = (
    "/p/openvocabdustr/probing_midlevel_vision/code/probing-mid-level-vision"
)


def run_evaluation(model):
    try:
        batch_size = 32

        output_dir = "<YOUR OUTPUT DIT>"

        command = (
            f"python evaluate_model_percepture.py "
            f"backbone={model} "
            f"experiment_model={model} "
            f"system.random_seed=8 system.num_gpus=1 "
            f"batch_size={batch_size} "
            f"dataset=twoafcdataset "
            f"output_dir={output_dir} "
            f"backbone.return_cls=True"
        )

        print(f"Running command: {command}")

        subprocess.run(command, shell=True, check=True, cwd=project_directory)
        print(f"Completed evaluation: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to evaluate model {model}: {e}")


max_threads = 1

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(run_evaluation, model) for model in models]

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"An exception occurred: {exc}")
