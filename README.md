Probing the Mid-level Vision Capabilities of Self-supervised Learning Methods
=============================================================================

This repository contains official implementation of the code for the paper [Probing the Mid-level Vision Capabilities of Self-Supervised Learning](https://arxiv.org/abs/2411.17474) which presents an analysis of the mid level perception of pretrained SSLs.


[Xuweiyi Chen](https://xuweiyichen.github.io/), [Markus Marks](https://damaggu.github.io/), [Zezhou Cheng](https://sites.google.com/site/zezhoucheng/)

If you find this code useful, please consider citing:  
```text
@article{chen2024probingmidlevelvisioncapabilities,
      title={Probing the Mid-level Vision Capabilities of Self-Supervised Learning}, 
      author={Xuweiyi Chen and Markus Marks and Zezhou Cheng},
      year={2024},
      eprint={2411.17474},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.17474}, 
}
```
**:warning: Note:** This is a cleanup version. Further edits and refinements are in progress. This note will be removed once the content has been finalized.


Model Checkpoints
-----------------

| **Model Name**       | **Backbone**         | **Dataset**    | **Source Link**                                                                                   |
|-----------------------|----------------------|----------------|---------------------------------------------------------------------------------------------------|
| Jigsaw               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| RotNet               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| NPID                 | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| SeLa-v2              | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| NPID++               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| PIRL                 | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| ClusterFit           | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| DeepCluster-v2       | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| SwAV                 | ResNet-50           | ImageNet-1K    | [SwAV repository](https://github.com/facebookresearch/swav)                                       |
| SimCLR               | ResNet-50           | ImageNet-1K    | [VISSL model zoo](https://github.com/facebookresearch/vissl/blob/main/MODEL_ZOO.md)              |
| MoCo v2              | ResNet-50           | ImageNet-1K    | [MoCo v2 repository](https://github.com/facebookresearch/moco)                                   |
| SimSiam              | ResNet-50           | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |
| BYOL                 | ResNet-50           | ImageNet-1K    | [Unofficial BYOL repo](https://github.com/yaox12/BYOL-PyTorch)                                   |
| Barlow Twins         | ResNet-50           | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |
| DenseCL              | ResNet-50           | ImageNet-1K    | [DenseCL repository](https://github.com/WXinlong/DenseCL)                                        |
| DINO                 | ResNet-50/ViT-B/16  | ImageNet-1K    | [DINO repository](https://github.com/facebookresearch/dino)                                      |
| MoCo v3              | ResNet-50/ViT-B/16  | ImageNet-1K    | [MoCo v3 repository](https://github.com/facebookresearch/moco-v3)                                |
| iBOT                 | ViT-B/16            | ImageNet-1K    | [iBOT repository](https://github.com/bytedance/ibot)                                             |
| MAE                  | ViT-B/16            | ImageNet-1K    | [MAE repository](https://github.com/facebookresearch/mae)                                       |
| MaskFeat             | ViT-B/16            | ImageNet-1K    | [MMSelfSup model zoo](https://mmselfsup.readthedocs.io/en/dev-1.x/model_zoo.html)                |


Environment Setup
-----------------

We recommend using Anaconda or Miniconda. To setup the environment, follow the instructions below.

```bash
conda create -n mid-probe python=3.9 --yes
conda activate mid-probe
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia 
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
conda install -c conda-forge nb_conda_kernels=2.3.1

pip install -r requirements.txt
python setup.py develop

pip install protobuf==3.20.3 
pre-commit install
```


Finally, please follow the dataset download and preprocessing instructions [here](./data_processing/README.md).


Evaluation Experiments
-----------

We provide code to train the depth probes and evaluate the correspondence. All experiments use
hydra configs which can be found [here](./configs). Below are example commands for running the
evaluations with the DINO ViT-B/16 backbone.

```python
python train_depth.py backbone=dino_b16 +backbone.return_multilayer=True dataset=nyu
python train_snorm.py backbone=dino_b16 +backbone.return_multilayer=True dataset=nyu
python train_generic_objectness.py backbone=dino_b16 dataset=voc12
python evaluate_model_percepture.py backbone=dino_b16 experiment_model=dino_b16 system.random_seed=8 system.num_gpus=1 batch_size=8 dataset=twoafcdataset output_dir=<OUTPUT_PATH> backbone.return_cls=True

python evaluate_navi_correspondence.py +backbone=dino_b16
python evaluate_scannet_correspondence.py +backbone=dino_b16
```

Obtabin Visualization
-----------
```python
python train_depth.py backbone=beit_v2_vitb16 +backbone.return_multilayer=True experiment_model=depth_beitv2_vitb16 system.port=12345 system.random_seed=10 system.num_gpus=1 batch_size=8 is_eval=true ckpt_path=<PATH_TO_CKPT>
```


Acknowledgments
-----------------

We would also like to acknowledge the following repositories and users for releasing very valuable
code and datasets: 

- [GeoNet](https://github.com/xjqi/GeoNet) for releasing the extracted surface normals for full NYU.  
- [Probe3D](https://github.com/mbanani/probe3d) for releasing probing algorithms for 3D foundation models.
- [Comparing evaluation protocols for self-supervised pre-training with image classification](https://github.com/XuweiyiChen/probing-mid-level-vision/tree/ssl-previous) for releasing a collection of Self-Supervised Learning methods and their usages.
