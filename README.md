# Stealthy and Practical Multi-Modal Attacks on Mixed Reality Tracking

This repository contains the official codebase for the AIVR 2024 paper:  
**"Stealthy and Practical Multi-Modal Attacks on Mixed Reality Tracking"**  
by *Yasra Chandio, Noman Bashir, and Fatima M. Anwar*.

📄 [Paper Link](https://yasrachandio.github.io/assets/pdfs/chandio_AIVR_24.pdf)]

---

## 🔍 Overview

Mixed Reality (MR) systems rely on sensor fusion for tracking. This code demonstrates a set of stealthy and effective attacks that simultaneously manipulate visual and inertial streams to bypass fusion-based tracking.

These attacks:
- Exploit **spatiotemporal vulnerabilities**
- Are **parameterizable**, enabling controlled deviation using **Right Frame Selection (RFS)** strategy

The project includes utilities for launching, configuring, and analyzing:
- Frame drop / duplication (temporal)
- Inertial signal perturbation (amplitude, orientation)
- Zero-displacement and path deviation attacks (spatial)

---

## Code Organization

| File | Description |
|------|-------------|
| `attack_frames.py` | Visual frame manipulation (drop/duplication) |
| `attack_trajectory.py` | Injects targeted trajectory distortions |
| `orientationattack.py` | Alters orientation using inertial misalignment |
| `deviationattack.py` | Performs trajectory deviation attack |
| `distance_enlargement.py`, `distance_reduction.py` | Spatial perturbation via amplitude modulation |
| `zero_displacement-hist.py` | Precise redirect attack using histogram shifts |
| `process_all.py`, `trajectory.py`, `speed.py` | Batch processing and evaluation tools |
| `project_hand_eye_to_pv.py` | Transforms tracking data to projected view space |
| `data_loader.py` | Loads synchronized visual and inertial frames |
| `ifc.py`, `allaboutimage.py` | Image similarity and perceptual hashing |
| `save_result_to_file.py` | Exports manipulated trajectory and metrics |

---

## 🧪 Dataset

This code is designed for use with the [**HoloSet**](https://zenodo.org/records/7200131#.ZBCnt2QpDVY) dataset, which includes:
- RGB and grayscale frames (5–30 fps)
- IMU data (12–20 Hz)
- Headset-collected sequences for indoor/outdoor scenes

To run the code, clone this repo and organize your data as:
data/
├── images/
└── imu/

## ⚙️ Dependencies

- Python 3.7+
- PyTorch
- NumPy
- OpenCV
- SciPy
- tqdm

## Install via:
pip install -r requirements.txt



## Launch a speed manipulation attack: 

python attack_trajectory.py --mode speedup --length 10 --warmup 5

## 📜 Citation

> If you use this codebase or HoloSet, please cite our paper:
> 
@inproceedings{chandio2024stealthy,
  title={Stealthy and Practical Multi-Modal Attacks on Mixed Reality Tracking},
  author={Chandio, Yasra and Bashir, Noman and Anwar, Fatima M.},
  booktitle={Proceedings of the IEEE International Conference on Artificial Intelligence and Virtual Reality (AIxVR)},
  year={2024}
}




