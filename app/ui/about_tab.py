"""
About / Guide tab — full in-app walkthrough explaining every step.
"""
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel, QScrollArea, QVBoxLayout, QWidget
)


GUIDE_HTML = """
<html>
<head>
<style>
  body  { font-family: Segoe UI, Arial, sans-serif; font-size: 13px;
          color: #E0E0E0; background: transparent; line-height: 1.7; }
  h1    { color: #4FC3F7; font-size: 20px; margin-top: 18px; }
  h2    { color: #80DEEA; font-size: 15px; margin-top: 14px; border-bottom: 1px solid #37474F; }
  h3    { color: #A5D6A7; font-size: 13px; margin-top: 10px; }
  code  { background: #1E2D3D; padding: 2px 6px; border-radius: 4px;
          font-family: Consolas, monospace; font-size: 12px; color: #FFD54F; }
  .note { background: #1B3A4B; border-left: 4px solid #4FC3F7;
          padding: 10px 14px; border-radius: 6px; margin: 10px 0; }
  .warn { background: #3B2F00; border-left: 4px solid #FFB300;
          padding: 10px 14px; border-radius: 6px; margin: 10px 0; }
  .ok   { background: #1B3B1E; border-left: 4px solid #66BB6A;
          padding: 10px 14px; border-radius: 6px; margin: 10px 0; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #37474F; padding: 6px 10px; text-align: left; }
  th    { background: #1E2D3D; color: #4FC3F7; }
  tr:nth-child(even) { background: #1A2730; }
  ul li  { margin-bottom: 4px; }
</style>
</head>
<body>

<h1>🌎 Illegal Mining Detection — Complete Guide</h1>
<p>
This application uses a <b>deep learning CNN</b> (EfficientNet-B0) trained on 40,000+
satellite image chips from the Amazon rainforest to automatically detect
<b>artisanal and conventional mining</b> activity in aerial imagery.
</p>
<div class="warn">Research prototype: mining labels do not establish legality.
Scores are uncalibrated and low scores cannot rule out mining. Grad-CAM is not
a verified mine boundary. Independent deployment testing is still required.</div>

<div class="note">
<b>Quick Start:</b>
Step 1 → Setup tab (download data) → Step 2 → Train tab (train model) → Step 3 → Predict tab (analyse images)
</div>

<h2>📡 The Dataset — Planet: Understanding the Amazon from Space</h2>
<p>
The dataset is from a <b>Kaggle competition</b> hosted by Planet Labs. It contains
satellite image chips at 3–5 metre resolution covering the entire Amazon basin.
</p>
<table>
  <tr><th>Property</th><th>Details</th></tr>
  <tr><td>Source</td><td>Planet Labs / Kaggle competition</td></tr>
  <tr><td>Training images</td><td>40,479 chips (256×256 px, JPEG)</td></tr>
  <tr><td>Test images</td><td>61,191 chips</td></tr>
  <tr><td>Labels</td><td>Multi-label: 17 tags per image</td></tr>
  <tr><td>Mining-related tags</td><td><code>artisanal_mine</code>, <code>conventional_mine</code></td></tr>
  <tr><td>Image format</td><td>RGB JPEG (3-band visible) or 4-band TIFF</td></tr>
</table>

<h3>All 17 Classification Tags</h3>
<ul>
  <li><b>🏗 Mining:</b> artisanal_mine, conventional_mine</li>
  <li><b>🌿 Vegetation:</b> primary, agriculture, cultivation, blooming, selective_logging</li>
  <li><b>🔥 Disturbance:</b> slash_burn, clear_cut, bare_ground, blow_down</li>
  <li><b>🏠 Human:</b> habitation, road</li>
  <li><b>💧 Hydro:</b> water</li>
  <li><b>☁ Weather:</b> clear, partly_cloudy, haze, cloudy</li>
</ul>

<h2>🧠 How the AI Model Works</h2>

<h3>1. Architecture — EfficientNet-B0</h3>
<p>
EfficientNet-B0 is a state-of-the-art compact CNN that balances accuracy and speed.
It was originally trained on <b>1.2 million ImageNet images</b> and we <b>fine-tune</b>
it on the Amazon data. This technique is called <b>transfer learning</b>.
</p>
<ul>
  <li><b>Input:</b> 224×224 RGB image</li>
  <li><b>Feature extractor:</b> 16 MBConv blocks (~5M parameters)</li>
  <li><b>Head:</b> Dropout(0.3) → Linear(1280 → 17) → Sigmoid</li>
  <li><b>Output:</b> 17 independent probabilities (multi-label)</li>
</ul>

<h3>2. Loss Function — BCEWithLogitsLoss</h3>
<p>
Binary Cross-Entropy with logits is applied <b>independently per tag</b>.
This means the model learns that an image can have multiple tags simultaneously
(e.g., both <code>artisanal_mine</code> and <code>road</code>).
</p>

<h3>3. Metrics — Fβ Score (β=2)</h3>
<p>
The Kaggle competition uses the <b>F2-score</b> which weights recall twice as heavily
as precision. This is ideal for environmental monitoring because <b>missing a real
mining site is worse than a false alarm</b>.
</p>

<h3>4. Mining Score</h3>
<p>
<code>mining_score = max(P(artisanal_mine), P(conventional_mine))</code><br/>
A score above <b>0.4</b> triggers a WARNING; above <b>0.65</b> triggers CRITICAL.
</p>

<h3>5. Explainability — Grad-CAM</h3>
<p>
<b>Gradient-weighted Class Activation Mapping</b> visualises <i>where</i> in the image
the model looked when making its decision. The heatmap overlay helps experts
verify whether the model is detecting genuine mining features (disturbed earth,
tailings ponds, access roads) or noise.
</p>

<h2>⚙ Step-by-Step Usage</h2>

<h3>Step 1 — Setup & Download</h3>
<ol>
  <li>Create a Kaggle account at <b>kaggle.com</b> if you don't have one.</li>
  <li>Go to <b>Account → API → Create New Token</b> — this downloads <code>kaggle.json</code>.</li>
  <li>Open the <b>Setup</b> tab, enter your Kaggle username and API key.</li>
  <li>Accept the competition rules at kaggle.com (required once).</li>
  <li>Click <b>Download Dataset</b>. The app will download ~6GB of imagery.</li>
</ol>
<div class="warn">
<b>⚠ Important:</b> You must accept the competition rules on Kaggle.com before downloading.
Go to <a>kaggle.com/c/planet-understanding-the-amazon-from-space</a> → Join Competition.
</div>

<h3>Step 2 — Train the Model</h3>
<ol>
  <li>Open the <b>Train</b> tab.</li>
  <li>Adjust hyperparameters (defaults are good for first run).</li>
  <li>Click <b>Start Training</b>. A GPU will train ~5× faster than CPU.</li>
  <li>Monitor loss and F2-score curves in real time.</li>
  <li>Training auto-stops when validation mining average precision stops improving.</li>
  <li>Best checkpoint is saved to <code>models/best_model.pth</code>.</li>
</ol>
<div class="note">
<b>Tip:</b> On a modern GPU (RTX 3070+), 15 epochs takes ~30 minutes.
On CPU, reduce epochs to 3–5 for a quick test.
</div>

<h3>Step 3 — Predict</h3>
<ol>
  <li>Open the <b>Predict</b> tab.</li>
  <li>Drag &amp; drop an image or click <b>Browse</b>.</li>
  <li>The model returns a <b>Mining Score</b> (0–100%), all 17 tag probabilities,
      and a <b>Grad-CAM heatmap</b> highlighting suspicious regions.</li>
  <li>Use <b>Batch Scan</b> to analyse a whole folder and export results to CSV.</li>
</ol>

<h3>Step 4 — Dashboard</h3>
<p>View training history, confusion matrix, and per-class F2 scores in the <b>Dashboard</b> tab.</p>

<h2>🖥 Hardware Requirements</h2>
<table>
  <tr><th></th><th>Minimum</th><th>Recommended</th></tr>
  <tr><td>RAM</td><td>8 GB</td><td>16 GB</td></tr>
  <tr><td>VRAM</td><td>CPU OK</td><td>4 GB GPU</td></tr>
  <tr><td>Storage</td><td>10 GB</td><td>20 GB SSD</td></tr>
  <tr><td>Training time (15 epochs)</td><td>~4h CPU</td><td>~30m GPU</td></tr>
</table>

<h2>📁 Project Structure</h2>
<pre style="background:#1E2D3D; padding:12px; border-radius:6px; font-size:11px; color:#FFD54F;">
illegal-mining-detector/
├── app/
│   ├── main.py              ← Launch the app
│   ├── core/
│   │   ├── dataset.py       ← Data loading + augmentation
│   │   ├── model.py         ← EfficientNet-B0 architecture
│   │   ├── trainer.py       ← Training loop
│   │   ├── predictor.py     ← Inference + Grad-CAM
│   │   └── downloader.py    ← Kaggle download helper
│   └── utils/
│       ├── config.py        ← All paths and constants
│       └── logger.py        ← Logging
├── data/raw/                ← Dataset lives here
├── models/                  ← Saved checkpoints
├── logs/                    ← Training logs + history
└── results/                 ← Batch prediction exports
</pre>

<h2>📚 References</h2>
<ul>
  <li>Tan &amp; Le (2019) — <i>EfficientNet: Rethinking Model Scaling for CNNs</i></li>
  <li>Selvaraju et al. (2017) — <i>Grad-CAM: Visual Explanations from Deep Networks</i></li>
  <li>Planet Labs / Kaggle (2017) — <i>Understanding the Amazon from Space</i></li>
  <li>Asner et al. (2013) — <i>Airborne mapping of illegal Amazon mining</i>, Science</li>
</ul>

</body>
</html>
"""


class AboutTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(GUIDE_HTML)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setContentsMargins(16, 16, 16, 16)

        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        layout.addWidget(scroll)
