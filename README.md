# Drone Gesture Control System  
**Hand Gesture Recognition for Drone Navigation using Mediapipe & Deep Learning**

Control a drone using simple hand gestures captured in real-time via webcam — no physical remote needed!

This project implements multiple approaches for robust hand gesture classification:

- Random Forest (baseline)
- 1D CNN (convolutional approach on landmark coordinates)
- ANN (simple feed-forward network)
- Few-Shot Learning prototype (prototype for adding new gestures with minimal data)

## Project Structure
Drone_AIMS_Scratch/
├── creating_data.py          # Collect raw hand landmark data (Mediapipe → CSV)
├── data_aug.py               # Data augmentation (noise, flip, rotate, translate)
├── crate_model_and_run.py    # Random Forest model + real-time inference
├── ann_model.py              # Simple ANN (MLP) model + real-time inference
├── cnn_model.py              # 1D CNN model (best accuracy) + real-time inference
├── fsl.py                    # Few-Shot Learning prototype (add new gestures easily)
├── data.csv                  # Raw collected data (do NOT commit large files)
├── data_aug.csv              # Augmented dataset (used for training)
├── model.joblib              # Saved Random Forest model
├── ann.pth                   # Saved ANN (PyTorch)
├── cnn.pth                   # Saved 1D CNN (PyTorch)
├── scaler_ann.pkl            # StandardScaler for ANN
├── scaler_cnn.pkl            # StandardScaler for CNN
└── data_fsl.pth              # Few-shot learning reference embeddings (optional)



## Supported Gestures & Drone Commands

| Gesture              | Label in model       | Drone Command    | Notes                     |
|----------------------|----------------------|------------------|---------------------------|
| Thumbs Up            | Thumbs_Up            | Up               |                           |
| Thumbs Down          | Thumbs_Down          | Down             |                           |
| Index Point Up       | Index_Point_Up       | Up               |                           |
| Index Point Down     | Index_Point_Down     | Down             |                           |
| Open Hand            | Open_Hand            | Forward          |                           |
| Closed Fist          | Closed_Fist          | Backward         |                           |
| Victory ✌️           | Victory              | Landing          |                           |
| Yo (call me gesture) | Yo                   | BackFlip         | Fun / trick command       |
| Little Finger Up     | Little_Finger_Up     | None             | Neutral / placeholder     |
| Thumb Left           | Thumb_Left           | Left             |                           |
| Thumb Right          | Thumb_Right          | Right            |                           |
| Dead (flat hand?)    | Dead                 | None             | Neutral / stop            |

## Features

- Real-time hand landmark detection using **MediaPipe Hands**
- Multiple classification backbones: Random Forest, ANN, 1D CNN
- Strong data augmentation pipeline (noise, flip, rotation, translation)
- Option to collect your own gestures (`creating_data.py`)
- Prototype few-shot learning mode — add new gestures with few examples (`fsl.py`)
- Models saved & loaded automatically

## Requirements

```bash
pip install opencv-python mediapipe pandas numpy scikit-learn torch torchvision torchaudio tqdm joblib

