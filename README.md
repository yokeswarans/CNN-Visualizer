# CNN Layer Visualizer

Upload any image, configure your CNN architecture, and watch the activation maps at every layer in real time.

---

## Setup

### 1. Clone / download this folder

```
cnn_visualizer/
├── app.py
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> PyTorch CPU version is fine — no GPU needed for visualization.

### 4. Run the app

```bash
python app.py
```

Open your browser at **http://localhost:5000**

---

## How to use

1. **Upload any image** (jpg, png, etc.) using the left panel
2. **Set global settings** — image resize size, how many maps to display, colormap
3. **Configure CNN layers** — add/remove layers, set per-layer:
   - Hidden units (number of filters)
   - Kernel size
   - Stride
   - Padding
   - BatchNorm / ReLU / MaxPool toggles
4. Click **▶ Run Visualization**
5. See activation maps at every layer, plus stats (mean, max, % active pixels)

---

## What you will see

- **Original image** — resized input
- **Each Conv Block** — activation maps after ReLU, one image per filter channel
- **Stats per layer** — mean activation, max value, sparsity (% of ReLU-killed neurons)
- **Final layer banner** — explains what the last feature volume means for classification

---

## Colormaps

| Name     | Good for                        |
|----------|---------------------------------|
| Viridis  | General purpose (default)       |
| Heat     | Spotting high-activation regions|
| Cool     | Low-activation / smooth layers  |
| Plasma   | High contrast                   |
| Grayscale| Raw intensity view              |

---

## Requirements

- Python 3.8+
- PyTorch 2.0+ (CPU is fine)
- Flask 2.3+