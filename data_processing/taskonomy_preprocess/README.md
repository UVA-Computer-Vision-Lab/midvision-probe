## Data Preprocess

### Download Taskonomy
```
omnitools.download mask_valid --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download depth_euclidean --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download depth_zbuffer --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download edge_occlusion --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download keypoints2d --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download keypoints3d --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download normal --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download principal_curvature --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download reshading --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree

omnitools.download rgb --components taskonomy --subset tiny \
  --dest /p/openvocabdustr/probing_midlevel_vision/data/ \
  --connections_total 40 --agree
```

### Construct Dataset

Please see the jupyter notebook.

```
from datasets import load_dataset

# Load the train split from the Hugging Face Hub
train_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy", split="train")

# Load the test split from the Hugging Face Hub
val_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy", split="validation")

# Load the test split from the Hugging Face Hub
test_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy", split="test")

# Access the train and test datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
```

```
from datasets import load_dataset

# Load the train split from the Hugging Face Hub
train_normal_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy_snorm", split="train")

# Load the test split from the Hugging Face Hub
val_normal_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy_snorm", split="validation")

# Load the test split from the Hugging Face Hub
test_normal_dataset = load_dataset("Xuweiyi/ssl_probing_taskonomy_snorm", split="test")

# Access the train and test datasets
print(f"Train dataset size: {len(train_dataset)}")
print(f"Val dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
```