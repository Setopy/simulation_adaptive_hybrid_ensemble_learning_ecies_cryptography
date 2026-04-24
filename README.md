
# Simulation of Adaptive Hybrid Ensemble Learning and Elliptic Curve Encryption for Cybersecurity

This work presents a simulation-based study on network intrusion detection using an adaptive hybrid ensemble of machine learning models, combined with Elliptic Curve Integrated Encryption Scheme (ECIES) cryptography to secure network traffic in transit.

The system trains six models on the UNSW-NB15 benchmark dataset and deploys them together as an ensemble that continuously adjusts its internal weights during simulation based on recent detection performance.

---

## Authors

**Seyi Tope Ogunji, MSc** (PhD Candidate)
Instituto Politecnico Nacional, CITEDI Centre

**Dr. Moises Sanchez Adame**
Instituto Politecnico Nacional, CITEDI Centre

**Dr. Oscar Montiel Ross**
Instituto Politecnico Nacional, CITEDI Centre

---

## Associated Publication

This repository accompanies a research article currently under preparation. A full citation will be added upon publication. If you use this work before the article is available, please cite as:

> Ogunji, S. T., Sanchez Adame, M., & Montiel Ross, O. (in preparation). Simulation of Adaptive Hybrid Ensemble Learning and Elliptic Curve Encryption for Cybersecurity. Instituto Politecnico Nacional, CITEDI Centre, Mexico.

---

## Academic Use and Credit

This codebase is made available for research and educational reference. If you use, adapt, or build on any part of this work, you are expected to give proper credit to the authors as listed above. Reproducing substantial portions of this code without attribution is not permitted.

---

## Background

Network intrusion detection remains a difficult problem because attack patterns change over time and no single model handles all attack types well. This work addresses that by combining three deep learning models (CNN, LSTM, DNN) and three traditional classifiers (XGBoost, Random Forest, SVM) into an ensemble that adapts its voting weights based on how well each model has performed in a recent time window.

The ECIES cryptography layer is applied to network traffic as it arrives, simulating a deployment scenario where traffic is transmitted securely and decrypted only when a closer inspection is needed during intrusion analysis.

---

## How the System Works

The project has two distinct phases.

**Phase 1 — Training**

Run main.py to train all six models on the UNSW-NB15 dataset. Each model is trained independently and saved to results/models/. The neural models are saved as .pth files (PyTorch) and the traditional models as .joblib files (scikit-learn and XGBoost).

**Phase 2 — Simulation**

Run sim_main.py to start the detection simulation. The system loads the pretrained models and generates synthetic network traffic. Each packet is encrypted using ECIES as soon as it arrives, simulating secure transmission across the network. The ensemble then runs an initial detection pass on the original traffic features. If a packet is flagged as a potential intrusion, the encrypted copy is decrypted and a second, more detailed detection pass is carried out on the decrypted data. If the intrusion is confirmed, an alert is recorded along with the traffic pattern analysis.

The ensemble starts with fixed initial weights and switches to a meta-learner (Logistic Regression stacking) once it has collected enough labeled samples to update the weights based on recent model performance.

---

## Project Structure

├── main.py                        # Entry point for training all six models
├── sim_main.py                    # Entry point for the simulation pipeline
├── sim_ids_monitor.py             # Adaptive ensemble engine and IDS logic
├── sim_network_simulator.py       # Network traffic simulation and monitoring
├── sim_crypto_manager.py          # ECIES encryption and decryption of network traffic
├── sim_traffic_generator.py       # Synthetic traffic generation
├── sim_config.py                  # Simulation configuration and thresholds
├── config.py                      # Training configuration
│
├── models/                        # Model architecture definitions
│   ├── base_model.py
│   ├── cnn_model.py
│   ├── lstm_model.py
│   ├── dnn_model.py
│   └── traditional_models.py
│
├── trainers/                      # Training logic
│   ├── neural_trainer.py
│   └── traditional_trainer.py
│
├── utils/                         # Data processing and metrics tracking
│
├── data/                          # UNSW-NB15 training and testing datasets
│
├── results/
│   ├── models/                    # Saved pretrained model files (.pth, .joblib)
│   ├── metrics/                   # Training metrics
│   ├── logs/                      # Training logs
│   └── simulation_results/        # Per-run simulation output
│
│
└── archive/                       # Simulation run outputs, plots, and logs
    ├── images/
    ├── logs/
    └── sim_runs/


---

## Requirements

- Python 3.8 or higher
- PyTorch
- scikit-learn
- XGBoost
- joblib
- NumPy
- pandas
- SciPy
- cryptography

Install dependencies:


pip install torch scikit-learn xgboost joblib numpy pandas scipy cryptography


---

## Running the Project

Train the models (skip this if using the pretrained files already in results/models/):

```
python main.py
```

Run the simulation:

```
python sim_main.py
```

Each run creates a timestamped folder under results/simulation_results/ containing traffic records, detection alerts, ensemble metrics, feature importances, and cryptography performance data.

Analyse results after a simulation run:

```
python analysis/adapt_visual.py
```

---

## Dataset

The UNSW-NB15 dataset was used for training and evaluation. It is a publicly available benchmark dataset for network intrusion detection research, created by the Australian Centre for Cyber Security.

If you use this dataset in your own work, please cite the original paper:

Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems. Proceedings of the Military Communications and Information Systems Conference (MilCIS), Canberra, Australia.


## Models in the Ensemble

| Model         | Type                                         | Saved Format |
|---------------|----------------------------------------------|--------------|
| CNN           | Deep learning (1D Convolutional + Attention) | .pth         |
| LSTM          | Deep learning (Long Short-Term Memory)       | .pth         |
| DNN           | Deep learning (Dense Neural Network)         | .pth         |
| XGBoost       | Gradient boosted trees                       | .joblib      |
| Random Forest | Bagged decision trees                        | .joblib      |
| SVM           | Linear Support Vector Machine                | .joblib      |

## Contact

For questions about this research or collaboration inquiries, contact the corresponding author:

**Dr. Oscar Montiel Ross**
Instituto Politecnico Nacional, CITEDI Centre
oross@ipn.mx


