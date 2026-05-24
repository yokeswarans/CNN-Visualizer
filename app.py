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
from torchvision import transforms
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

def build_blocks(layers_config):
    blocks = []
    in_ch = 3
    for lc in layers_config:
        seq = []
        out_ch = int(lc['hidden_units'])
        seq.append(nn.Conv2d(
            in_ch, out_ch,
            kernel_size=int(lc['kernel_size']),
            stride=int(lc['stride']),
            padding=int(lc['padding'])
        ))
        if lc.get('batchnorm'):
            seq.append(nn.BatchNorm2d(out_ch))
        if lc.get('relu'):
            seq.append(nn.ReLU())
        if lc.get('pooling'):
            seq.append(nn.MaxPool2d(2, 2))
        blocks.append(nn.Sequential(*seq))
        in_ch = out_ch
    return blocks

def map_to_b64(act2d, cmap_name):
    mn, mx = act2d.min(), act2d.max()
    if mx - mn > 1e-8:
        act2d = (act2d - mn) / (mx - mn)
    else:
        act2d = np.zeros_like(act2d)
    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.imshow(act2d, cmap=cmap_name, vmin=0, vmax=1, aspect='auto')
    ax.axis('off')
    plt.subplots_adjust(0, 0, 1, 1)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

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
        img_size = int(config.get('img_size', 128))

        img = Image.open(img_file).convert('RGB').resize((img_size, img_size))

        # original as b64
        orig_buf = io.BytesIO()
        img.save(orig_buf, format='PNG')
        orig_b64 = base64.b64encode(orig_buf.getvalue()).decode()

        tensor = transforms.ToTensor()(img).unsqueeze(0)

        blocks  = build_blocks(config['layers'])
        results = []
        x = tensor

        with torch.no_grad():
            for i, block in enumerate(blocks):
                block.eval()
                x = block(x)
                act = x.squeeze(0).numpy()          # C x H x W
                C, H, W = act.shape
                n_show = min(C, max_maps)

                maps = [map_to_b64(act[j], cmap) for j in range(n_show)]
                lc   = config['layers'][i]

                results.append({
                    'layer_idx':       i + 1,
                    'shape':           [C, H, W],
                    'hidden_units':    lc['hidden_units'],
                    'kernel_size':     lc['kernel_size'],
                    'stride':          lc['stride'],
                    'padding':         lc['padding'],
                    'batchnorm':       lc.get('batchnorm', False),
                    'relu':            lc.get('relu', True),
                    'pooling':         lc.get('pooling', False),
                    'maps':            maps,
                    'mean_activation': round(float(np.maximum(act, 0).mean()), 4),
                    'max_activation':  round(float(act.max()), 4),
                    'sparsity':        round(float((act <= 0).mean() * 100), 1),
                    'total_maps':      C,
                    'shown_maps':      n_show,
                })

        return jsonify({'success': True, 'original': orig_b64, 'layers': results})

    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)