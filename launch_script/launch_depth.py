import submitit
import os


def train_depth(model_name, config_path, port, seed):
    os.system(
        f"""
    cd <YOUR PATH>/probing-mid-level-vision
    export OMP_NUM_THREADS=1
    export TORCH_SHOW_CPP_STACKTRACES=1

    # Set API key for Weights & Biases
    export WANDB_API_KEY="<WANDB KEY>"

    # Run the training script
    python train_depth.py backbone={model_name} +backbone.return_multilayer=True experiment_model=depth_{model_name} system.port={port} system.random_seed={seed} system.num_gpus=2 batch_size=16 dataset=navi_reldepth
    """
    )


def main():
    models = [
        "mocov3_resnet50",
        "crocov2_b16",
        "croco_b16",
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

    log_folder = "logs/depth"
    base_port = 12366
    fixed_seed = 42

    # Create submitit executor
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(
        timeout_min=1440,
        slurm_partition="gpu",
        gpus_per_node=2,
        nodes=1,
        slurm_ntasks_per_node=1,
        slurm_qos="csresnolim",
        cpus_per_task=12,
        mem_gb=72,
        slurm_mail_type="ALL",
        slurm_mail_user="<YOUR EMAIL>",
        slurm_additional_parameters={
            "reservation": "<YOUR RESERVATION>"  # if you have a reservation
        },
    )

    jobs = []
    for i, model_name in enumerate(models):
        port = base_port + i

        executor.update_parameters(name=f"train_depth_{model_name}")
        try:
            job = executor.submit(
                train_depth, model_name, "train_depth.py", port, fixed_seed
            )
            jobs.append(job)
            print(f"Job submitted successfully: {model_name} - depth")
        except Exception as e:
            print(f"Failed to submit job for {model_name} - depth: {e}")


if __name__ == "__main__":
    main()
