# 🌊 StreamFlow-ML

A modular machine learning framework for streamflow prediction using deep learning architectures such as LSTM, BiLSTM, and GRU. The framework supports flexible configuration, training, hyperparameter tuning, and evaluation of hydrologic time-series models.

---

## 📁 Project Structure

```
StreamFlow-ML/
├── config/             # YAML configuration files
├── data/               # Input data (not committed due to size)
├── hpt/                # Hyperparameter tuning scripts/configs
├── models/             # Deep learning model architectures
│   ├── bilstm_model.py
│   ├── gru_model.py
│   └── lstm_model.py
├── notebooks/          # Jupyter notebooks for experiments/visuals
├── results/            # Evaluation results and metrics
├── saved_models/       # Saved trained model weights
├── utils/              # Utilities (data loading, evaluation)
│   ├── data_loader.py
│   └── evaluation.py
├── train.py            # Script to train models
├── tune_model.py       # Script for hyperparameter tuning
├── predict.py          # Script for prediction/inference
└── README.md           # Project documentation
```

---

## ⚙️ Setup

Make sure Python ≥ 3.7 is installed. Then:

```bash
pip install -r requirements.txt
```

---

## 🚂 Training

```bash
python train.py
```

You can also customize the training process by editing YAML configs under `config/`.

---

## 🔍 Evaluation

```bash
python -m utils.evaluation
```

This runs model evaluation (e.g., NSE, RMSE, Kling-Gupta) and logs results into the `results/` folder.

---

## 🧪 Hyperparameter Tuning

```bash
python tune_model.py
```

This runs your HPT strategy defined in the `hpt/` folder.

---

## 📈 Models Supported

- LSTM
- BiLSTM
- GRU  
(Architectures are located in `models/`)

---

## 📌 Notes

- Large `.csv`/dataset files were removed from GitHub due to file size limits.
- Compatible with site-specific and multi-site streamflow prediction.
- All models are configurable via external YAML files.

---

## 🙋‍♂️ Maintainer

Developed and maintained by [@Almehedi06](https://github.com/Almehedi06)

---

## 📜 License

MIT License
