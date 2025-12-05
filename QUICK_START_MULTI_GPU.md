# Quick Start: Multi-GPU Training with Garfield

Get started with multi-GPU distributed training in **< 5 minutes**.

---

## Prerequisites

```bash
# Install dependencies
pip install pytorch-lightning>=2.0.0

# Or update from requirements
pip install -r requirements.txt
```

---

## Single-GPU (Backward Compatible)

Your existing scripts work with **zero changes**:

```python
import Garfield as gf
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')

user_config = dict(
    adata_list=adata,
    profile='RNA',
    sample_col='batch',
    n_epochs=100,
    # ... other params
)

model = gf.Garfield(gf.settings.set_gf_params(user_config))
model.train()
```

---

## Multi-GPU (3 Lines Changed!)

```python
import Garfield as gf
import scanpy as sc

adata = sc.read_h5ad('data.h5ad')

user_config = dict(
    adata_list=adata,
    profile='RNA',
    sample_col='batch',

    # ✨ NEW: Multi-GPU configuration (3 lines!)
    accelerator='gpu',
    devices=4,                                   # Use 4 GPUs
    strategy='ddp_find_unused_parameters_true',  # Required for Garfield

    n_epochs=100,
    # ... other params
)

model = gf.Garfield(gf.settings.set_gf_params(user_config))
model.train()
```

**Launch:**
```bash
torchrun --nproc_per_node=4 train.py
```

**That's it!** 🎉

---

## Optional Performance Boosts

### Mixed Precision (2x faster)

```python
user_config = dict(
    # ...
    precision='16-mixed',  # FP16 mixed precision
)
```

### Async Data Loading

```python
user_config = dict(
    # ...
    num_workers=4,  # 4 workers per GPU
)
```

### Larger Batches

```python
user_config = dict(
    # ...
    edge_batch_size=8192,   # 4x larger for 4 GPUs
    node_batch_size=512,    # 4x larger for 4 GPUs
)
```

---

## Troubleshooting

**Out of memory?**
```python
# Reduce batch size
edge_batch_size=2048
node_batch_size=128

# Or use mixed precision
precision='16-mixed'
```

**Hanging at initialization?**
```bash
# Ensure devices matches nproc_per_node
torchrun --nproc_per_node=4 train.py  # devices=4
```

---

## Full Documentation

- **Comprehensive Guide**: [MULTI_GPU_TRAINING.md](./MULTI_GPU_TRAINING.md)
- **Examples**: [examples/](./examples/)
- **Implementation Details**: [IMPLEMENTATION_SUMMARY.md](./IMPLEMENTATION_SUMMARY.md)

---

## Need Help?

- GitHub Issues: https://github.com/zhou-1314/Garfield/issues
- Email: zhouwg1314@gmail.com
