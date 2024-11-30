import submitit
import os


def train_snorm(model_name, config_path, port, seed):
    os.system(
        f"""
    cd /<YOUR PATH>/probing-mid-level-vision
    export OMP_NUM_THREADS=1
    export TORCH_SHOW_CPP_STACKTRACES=1
    
    conda activate <YOUR ENV>

    # Set API key for Weights & Biases
    export WANDB_API_KEY="<YOUR WANDB KEY>"

    # Run the training script
    python train_snorm.py backbone={model_name} +backbone.return_multilayer=True experiment_model=snorm_{model_name} system.port={port} system.random_seed={seed}
    """
    )


def main():
    models = [
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
        # "milan_vitb16",
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
    ]

    log_folder = (
        "/<YOUR PATH>/probing-mid-level-vision/launch_script/snorm/submitit_output_nyu"
    )
    base_port = 12366
    fixed_seed = 58

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
            "reservation": "<YOUR RESERVATION>"
        },  # if you have a reservation
    )

    jobs = []
    for i, model_name in enumerate(models):

        port = base_port + i

        executor.update_parameters(name=f"train_snorm_{model_name}")

        try:
            job = executor.submit(
                train_snorm, model_name, "train_snorm.py", port, fixed_seed
            )
            print(f"Submitted job: {job.job_id}: {model_name}")
        except Exception as e:
            print(f"Failed to submit job: {model_name}: {e}")
        jobs.append(job)


if __name__ == "__main__":
    main()
