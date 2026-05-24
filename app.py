import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import io
import base64

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

COLORMAPS = {
    'viridis': 'viridis',
    'heat':    'hot',
    'cool':    'cool',
    'gray':    'gray',
    'plasma':  'plasma',
}

# ── VGG16 global cache ──────────────────────────────────────────────────────
_vgg16      = None
_categories = None

def get_vgg16():
    global _vgg16, _categories
    if _vgg16 is None:
        print("⏳ Loading VGG16 pretrained weights (first time only)...")
        weights     = models.VGG16_Weights.DEFAULT
        _vgg16      = models.vgg16(weights=weights)
        _vgg16.eval()
        _categories = weights.meta["categories"]   # 1000 ImageNet class names
        print("✅ VGG16 ready!")
    return _vgg16, _categories

VGG16_LAYER_LABELS = [
    "Conv1_1",  "Conv1_2",
    "Conv2_1",  "Conv2_2",
    "Conv3_1",  "Conv3_2",  "Conv3_3",
    "Conv4_1",  "Conv4_2",  "Conv4_3",
    "Conv5_1",  "Conv5_2",  "Conv5_3",
]

# ── Helpers ─────────────────────────────────────────────────────────────────

def map_to_b64(act2d, cmap_name, highlight=False):
    mn, mx = act2d.min(), act2d.max()
    act2d  = (act2d - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(act2d)

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.imshow(act2d, cmap=cmap_name, vmin=0, vmax=1, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])

    if highlight:
        for spine in ax.spines.values():
            spine.set_edgecolor('#ff6ab0')
            spine.set_linewidth(4)
    else:
        ax.axis('off')

    plt.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight',
                pad_inches=0.03 if highlight else 0, dpi=72)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def pack_layer(act, max_maps, cmap, is_last=False):
    """Convert C×H×W numpy activation into the dict the frontend expects."""
    C, H, W = act.shape
    n_show  = min(C, max_maps)

    # rank channels by mean positive activation
    ch_means = np.maximum(act, 0).mean(axis=(1, 2))   # shape (C,)
    top_idx  = set(np.argsort(ch_means)[::-1][:n_show].tolist())

    # for last layer show TOP channels; otherwise show first n_show
    show_idx = sorted(top_idx) if is_last else list(range(n_show))

    maps = []
    for j in show_idx:
        highlight = is_last and (j in top_idx)
        maps.append({
            'b64':     map_to_b64(act[j], cmap, highlight=highlight),
            'channel': j + 1,
            'mean':    round(float(ch_means[j]), 4),
            'is_top':  highlight,
        })

    return {
        'maps':            maps,
        'shape':           [C, H, W],
        'mean_activation': round(float(ch_means.mean()), 4),
        'max_activation':  round(float(act.max()), 4),
        'sparsity':        round(float((act <= 0).mean() * 100), 1),
        'total_maps':      C,
        'shown_maps':      n_show,
        'is_last':         is_last,
    }

# ── Custom (random weights) mode ────────────────────────────────────────────

def build_blocks(layers_config):
    blocks, in_ch = [], 3
    for lc in layers_config:
        out_ch = int(lc['hidden_units'])
        seq = [nn.Conv2d(in_ch, out_ch,
                         kernel_size=int(lc['kernel_size']),
                         stride=int(lc['stride']),
                         padding=int(lc['padding']))]
        if lc.get('batchnorm'): seq.append(nn.BatchNorm2d(out_ch))
        if lc.get('relu'):      seq.append(nn.ReLU())
        if lc.get('pooling'):   seq.append(nn.MaxPool2d(2, 2))
        blocks.append(nn.Sequential(*seq))
        in_ch = out_ch
    return blocks


def run_custom(tensor, layers_config, max_maps, cmap):
    blocks, results, x = build_blocks(layers_config), [], tensor
    with torch.no_grad():
        for i, block in enumerate(blocks):
            block.eval(); x = block(x)
            act     = x.squeeze(0).numpy()
            lc      = layers_config[i]
            is_last = (i == len(blocks) - 1)
            ops     = ['Conv2d (random)']
            if lc.get('batchnorm'): ops.append('BatchNorm2d')
            if lc.get('relu'):      ops.append('ReLU')
            if lc.get('pooling'):   ops.append('MaxPool2d')

            layer = pack_layer(act, max_maps, cmap, is_last=is_last)
            layer.update({
                'layer_idx':    i + 1,
                'label':        f"Conv Block {i + 1}",
                'ops':          ops,
                'hidden_units': lc['hidden_units'],
                'kernel_size':  lc['kernel_size'],
            })
            results.append(layer)
    return results, None   # no predictions for random mode

# ── VGG16 (pretrained) mode ─────────────────────────────────────────────────

def run_vgg16(img_pil, max_layers, max_maps, cmap):
    model, categories = get_vgg16()

    # VGG16 official preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(img_pil).unsqueeze(0)

    results   = []
    conv_idx  = 0
    x         = tensor

    with torch.no_grad():
        # ── walk through VGG16 feature layers ──
        for layer in model.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                act     = x.squeeze(0).numpy()
                is_last = (conv_idx == min(max_layers, 13) - 1)
                info    = pack_layer(act, max_maps, cmap, is_last=is_last)
                info.update({
                    'layer_idx':    conv_idx + 1,
                    'label':        VGG16_LAYER_LABELS[conv_idx],
                    'ops':          ['Conv2d (pretrained)', 'ReLU'],
                    'hidden_units': act.shape[0],
                    'kernel_size':  3,
                })
                results.append(info)
                conv_idx += 1
                if conv_idx >= max_layers:
                    break

        # ── continue through the rest of features + classifier for prediction ──
        if conv_idx < 13:
            # finish remaining feature layers
            for layer in list(model.features)[list(model.features).index(layer) + 1:]:
                x = layer(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)
        logits = model.classifier(x)

    # top-5 predictions
    probs          = torch.softmax(logits[0], dim=0)
    top5_p, top5_i = torch.topk(probs, 5)
    predictions = [
        {'label': categories[i.item()], 'confidence': round(p.item() * 100, 2)}
        for p, i in zip(top5_p, top5_i)
    ]

    return results, predictions

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/visualize', methods=['POST'])
def visualize():
    try:
        img_file = request.files['image']
        config   = json.loads(request.form['config'])
        cmap     = COLORMAPS.get(config.get('colormap', 'viridis'), 'viridis')
        max_maps = int(config.get('max_maps', 8))
        mode     = config.get('mode', 'custom')

        img_pil = Image.open(img_file).convert('RGB')

        # display-size original (always show 224 square for consistency)
        disp = img_pil.resize((224, 224))
        buf  = io.BytesIO(); disp.save(buf, format='PNG')
        orig_b64 = base64.b64encode(buf.getvalue()).decode()

        if mode == 'vgg16':
            max_layers = int(config.get('vgg_layers', 5))
            layers, predictions = run_vgg16(img_pil, max_layers, max_maps, cmap)
        else:
            img_size = int(config.get('img_size', 128))
            img_r    = img_pil.resize((img_size, img_size))
            tensor   = transforms.ToTensor()(img_r).unsqueeze(0)
            layers, predictions = run_custom(tensor, config['layers'], max_maps, cmap)

        return jsonify({
            'success':     True,
            'original':    orig_b64,
            'layers':      layers,
            'predictions': predictions,
            'mode':        mode,
        })

    except Exception as e:
        import traceback
        return jsonify({'success': False,
                        'error': str(e),
                        'trace': traceback.format_exc()}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)