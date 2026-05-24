# CNN Layer Visualizer

A full interactive CNN visualization tool built with **PyTorch + Flask**.
Upload any image, configure your architecture or use pretrained VGG16,
and watch activations flow through every layer in real time.

---

## Project Structure

```
cnn_visualizer/
├── app.py                  ← Flask backend + PyTorch CNN logic
├── requirements.txt        ← Python dependencies
├── README.md               ← this file
└── templates/
    └── index.html          ← full frontend UI
```

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch CPU version is fine — no GPU needed for visualization.

### 3. Fix Windows OpenMP conflict (Windows only)

The first two lines of `app.py` handle this automatically:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

No manual action needed.

### 4. Run

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## Two Modes

### 🎲 Random Weights (Custom Architecture)

Build your own CNN from scratch in the UI.
Weights are randomly initialized — activations show raw filter responses,
not learned features. Good for understanding the pipeline and shapes.

**What you configure per layer:**

| Field | What it does |
|---|---|
| Hidden units | Number of filters = output depth |
| Kernel size | Spatial size of each filter (3 = 3×3) |
| Stride | How many pixels the filter jumps per step |
| Padding | Zero-border around image (padding=1 keeps size same with kernel=3) |
| BatchNorm | Normalize activations per channel across the batch |
| ReLU | Kill all negative activations |
| MaxPool 2×2 | Halve spatial size, keep strongest activations |

**Output depth rule:**
```
filter actual size  = kernel × kernel × in_channels   (auto by PyTorch)
output depth        = hidden_units                     (your choice)
```

---

### 🧠 VGG16 Pretrained

Uses VGG16 trained on ImageNet (1.2M images, 1000 classes).
Filters have real learned weights — activations show genuine detected features.

**VGG16 Architecture:**

```
Input: 224×224×3

Block 1:  Conv1_1(64)  → ReLU → Conv1_2(64)  → ReLU → MaxPool  →  112×112×64
Block 2:  Conv2_1(128) → ReLU → Conv2_2(128) → ReLU → MaxPool  →   56×56×128
Block 3:  Conv3_1(256) → ReLU → Conv3_2(256) → ReLU → Conv3_3(256) → ReLU → MaxPool → 28×28×256
Block 4:  Conv4_1(512) → ReLU → Conv4_2(512) → ReLU → Conv4_3(512) → ReLU → MaxPool → 14×14×512
Block 5:  Conv5_1(512) → ReLU → Conv5_2(512) → ReLU → Conv5_3(512) → ReLU → MaxPool →  7×7×512
                                                                                           ↓
                                                                                    Flatten: 25088
                                                                                           ↓
                                                                                     FC → 4096
                                                                                           ↓
                                                                                     FC → 4096
                                                                                           ↓
                                                                                   FC → 1000 classes
```

13 conv layers total. Depth grows (64→128→256→512), spatial size shrinks (224→7).

**Preprocessing (applied automatically):**

```python
Resize(256) → CenterCrop(224) → ToTensor → Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)
```

These exact values match how VGG16 was trained on ImageNet.

**First run downloads pretrained weights (~550 MB) once.**
After that they are cached at `~/.cache/torch/` and load instantly.

---

## What You See in the UI

### Per Layer

| Element | What it means |
|---|---|
| Shape badge | `Cch × H×W px` — channels × spatial size after this layer |
| mean | Average positive activation across the map |
| max | Strongest single activation value |
| % active | Percentage of neurons that survived ReLU (not zeroed) |
| Op tags | Which operations this block applied |
| Activation maps | One image per filter channel, colorized |

### Final Layer (highlighted in pink)

The last conv layer's maps are sorted by **highest mean activation**.
Channels with a pink glow + TOP badge fired strongest for your image —
these are the features that drove the classification.

### Predictions (VGG16 only)

Top 5 ImageNet class predictions with confidence bars.
The model identifies the image AFTER the final conv layer —
the FC layers convert the 7×7×512 volume into class probabilities.

---

## How Forward Hooks Work (VGG16)

PyTorch does not expose intermediate outputs by default.
We attach hooks to each ReLU layer:

```python
def make_h(i):
    def h(module, input, output):
        acts[i] = output.squeeze(0).detach().numpy()
    return h

for layer in model.features:
    if isinstance(layer, nn.ReLU):
        layer.register_forward_hook(make_h(count))
```

Every time the model passes through a ReLU during `model(tensor)`,
the hook fires and saves that layer's activation map automatically.
Hooks are removed after each request to avoid memory leaks.

---

## Colormaps

| Name | Good for |
|---|---|
| Viridis | General purpose (default) |
| Heat | Spotting high-activation regions |
| Cool | Low activation / smooth layers |
| Plasma | High contrast |
| Grayscale | Raw intensity view |

---

## Key Concepts Recap

```
One filter → collapses ALL input channels → 1 activation map
6 filters  → 6 activation maps stacked   → output depth 6

Early layers  → detect edges, corners, color gradients
Middle layers → detect textures, shapes, curves
Deep layers   → detect object parts (eyes, wheels, fur)
Final layer   → high-level feature volume → FC layer → class label
```

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CPU is fine)
- Flask 2.3+
- torchvision 0.15+
- matplotlib, Pillow, numpy