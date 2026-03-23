# Inference Tools

## 1) Interactive checkpoint tester

Run:

```powershell
python -m model.test_model_viewer
```

Optional arguments:

```powershell
python -m model.test_model_viewer --weights model/annotations/best_model.pt --split test --threshold 0.5 --device auto
```

What it does:
- lets you pick a checkpoint (`.pt`) if not provided,
- loads selected split metadata and WFDB records,
- prompts for sample index,
- prints full CSV row and model prediction side by side.

## 2) External inference API

Use `model/inference_api.py` in other modules:

```python
import numpy as np
from model.inference_api import predict_from_checkpoint

x = np.random.randn(12, 5000).astype("float32")
out = predict_from_checkpoint(
    weights_path="model/annotations/best_model.pt",
    data=x,
    threshold=0.5,
    class_names=["class_0", "class_1", "class_2", "class_3", "class_4", "class_5", "class_6", "class_7"],
)

print(out["probabilities"][0])
print(out["predictions"][0])
print(out["positive_labels"][0])
```

