# 👁️ DocSight — Diabetic Retinopathy Detection

AI-powered Diabetic Retinopathy (DR) screening from fundus photographs using EfficientNet-B4 with Grad-CAM explainability.

🔗 **Live Demo**: [DocSight DR on Hugging Face Spaces](https://huggingface.co/spaces/Salonideshmukh/docsight-dr-detection)

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| Architecture | EfficientNet-B4 |
| Classes | DR / No DR |
| Threshold | 0.35 |
| Explainability | Grad-CAM heatmap overlay |
| Model Hosting | Hugging Face Hub |

### How It Works
- **EfficientNet-B4 backbone** processes the full fundus image
- **Ben Graham normalization** + CLAHE preprocessing (identical to training)
- **Grad-CAM heatmap** highlights retinal regions influencing the prediction
- Model weights are downloaded automatically from Hugging Face Hub at startup

---

## 🚀 Usage

1. Open the app
2. Upload a fundus camera image (`.jpg` / `.png`)
3. Click **Analyze** — get result + Grad-CAM in seconds

### Output
- **Prediction**: DR / Borderline / No DR
- **Confidence score** (0–1)
- **Grad-CAM heatmap** showing attention areas
- **Overlay** of heatmap on original image

---

## 🏗️ Tech Stack

- **Backend**: Flask + Gunicorn
- **Deep Learning**: PyTorch, timm
- **Preprocessing**: OpenCV, Albumentations
- **Model Hosting**: Hugging Face Hub
- **Deployment**: Docker on Hugging Face Spaces

---

## 📁 Project Structure
├── app.py                      # Flask server
├── model.py                    # Model architecture + inference
├── Dockerfile                  # Container setup
├── requirements.txt
└── templates/
├── dr_detection_gui.html   # Main UI
└── eye_info.html           # About / info page

---

## ⚠️ Disclaimer

This tool is a **screening aid only** and does not replace clinical diagnosis by a qualified ophthalmologist. Always refer patients for formal retinal examination.

---

## 👩‍💻 Made By

**Saloni Deshmukh**
🔗 [GitHub](https://github.com/SaloniD8)

---

## 📜 License

MIT License
