import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

models_and_experiments = {
    # "croco_vitb16": "objectness_croco_vitb16",
    "barlowtwins_resnet50": "objectness_barlowtwins_resnet50",
    # "beit_v2_vitb16": "objectness_beit_v2_vitb16",
    "byol_resnet50": "objectness_byol_resnet50",
    "clusterfit_resnet50": "objectness_clusterfit_resnet50",
    "deepcluster-v2-resnet50": "objectness_deepcluster_v2_resnet50",
    "densecl_resnet50": "objectness_densecl_resnet50",
    "dino_b16": "objectness_dino_b16",
    "eva_vitb16": "objectness_eva_vitb16",
    "ibot_b16": "objectness_ibot_b16",
    "jigsaw_resnet50": "objectness_jigsaw_resnet50",
    "mae_b16": "objectness_mae_b16",
    "maskfeat_vitb16": "objectness_maskfeat_vitb16",
    # "milan_vitb16": "objectness_milan_vitb16",
    "mocov2_resnet50": "objectness_mocov2_resnet50",
    # "mocov3_b14": "objectness_mocov3_b14",
    "npid-plusplus_resnet50": "objectness_npid_plusplus_resnet50",
    "pirl_resnet50": "objectness_pirl_resnet50",
    # "pixmlm_vitb16": "objectness_pixmlm_vitb16",
    "rotnet_resnet50": "objectness_rotnet_resnet50",
    "simsiam_resnet50": "objectness_simsiam_resnet50",
    # "sela-v2_resnet50": "objectness_sela_v2_resnet50",
    "simclr_resnet50": "objectness_simclr_resnet50",
    "swav_resnet50": "objectness_swav_resnet50",
    "npid_resnet50": "objectness_npid_resnet50",
}

transformer_models = [
    "croco_vitb16",
    "beit_v2_vitb16",
    "dino_b16",
    "eva_vitb16",
    "ibot_b16",
    "mae_b16",
    "maskfeat_vitb16",
    "milan_vitb16",
    "pixmlm_vitb16",
]

# Base command for running the training script
base_command = (
    "python train_generic_objectness.py "
    "backbone={model} +backbone.return_multilayer=False +backbone.return_kqv=True "
    "experiment_model={experiment_model} system.random_seed=8 system.num_gpus=1 "
    "batch_size=1 dataset=voc +output_dir=/p/openvocabdustr/probing_midlevel_vision/code/probing-mid-level-vision/objectness_output"
)


def run_training(model, experiment_model):
    try:
        command = base_command.format(model=model, experiment_model=experiment_model)

        if model not in transformer_models:
            command += " +backbone.mode_selected=3"
        else:
            command += " +backbone.mode_selected=k"

        print(f"Running command: {command}")

        subprocess.run(command, shell=True, check=True)
        print(f"Completed: {model}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run model {model}: {e}")


max_threads = 3

with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [
        executor.submit(run_training, model, experiment_model)
        for model, experiment_model in models_and_experiments.items()
    ]

    for future in as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            print(f"Generated an exception: {exc}")
