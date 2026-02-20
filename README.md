# Raspi Cat Sentry
> An automated, Raspberyy Pi based security system desgigned to detect and monitor feline activity using computer vision and automated data pipelines.

**Raspi Cat Sentry** is a specialized monitoring tool that leverages the power of the Raspberry Pi to capture video, detect cats in real-time, and manage a training pipeline for improving computer vision models.

---

## Key Features

### End-to-End Data Pipeline
The project features a robust multi-step pipeline to transform raw video data into a labeled dataset for training:
* **Step 1 (Frame Splitter):** Automatically extracts discrete frames from raw video captures for analysis.
* **Step 2 (Auto-Labeler):** Uses pre-trained Yolo26x to identify cats and generate bounding box labels automatically, significantly reducing manual work.
* **Step 3 (Data Splitter):** Intelligently organizes labeled data into training and validation sets to prepare for custom model training.

### Inteligent Inference
* **Real-time Detection:** Optimized to run inference on edge devices, identifying feline intruders or pets as they move through the frame.
* **Automated Training:** Includes dedicated scripts to train custom YOLO detection model specifically on your own environment's data.

### Local Data Management
* **Configuration-Driven:** Uses `data.yaml` to manage class names and dataset paths.
* **Efficient Storage:** Designed to manage large amounts of image data locally.

--- 

## Tech Stack
* **Hardware:** Raspberry Pi (external camera required)
* **Language:** Python 3.12
* **Computer Vision:** OpenCV, YOLO

---

## Project Structure
```
raspi-cat-sentry/
├── data/
│   └── data.yaml             # Dataset configuration (classes, paths)
├── scripts/
│   ├── pipeline/             # The core automated data processing workflow
│   │   ├── run_pipeline.py           # Orchestrator for the data pipeline
│   │   ├── step01_frame_splitter.py  # Video-to-frame extraction
│   │   ├── step02_auto_labeler.py    # Automated AI-assisted labeling
│   │   └── step03_train_val_split.py # Training/Validation set organization
│   └── training/
│       └── train.py          # Custom model training script
├── src/
│   └── run_inference.py      # Main script for real-time cat detection
├── LICENSE                 
└── README.md       
```

---

## Getting Started

### Prerequisites
1. **Raspberry Pi** with an external camera.
2. **Python 3.12** installled.

### Installation
1. **Clone the repo:**
```bash
git clone https://github.com/kacperf04/raspi-cat-sentry.git
cd raspi-cat-sentry
```

2. **Install packages:**
```bash
pip install requirements.txt
```

3. **Run the Data Pipeline:**
```bash
python scripts/pipeline/run_pipeline.py
```

4. **Start Detection:**
```bash
python src/run_inference.py
```

---

## Roadmap
- [x] Create automated frame extraction.
- [x] Implemented AI-assisted auto-labeling.
- [x] Build training/validation split logic.
- [ ] Optimize inference.
- [ ] Implement data collection logic.
- [ ] Implement local notification system.
- [ ] Develop a web dashboard for viewing detection logs.
