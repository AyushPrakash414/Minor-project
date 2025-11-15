# 🥔 Potato Disease Detection System

<div align="center">

<p align="center">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black" alt="JavaScript"/>
</p>

**AI-Powered Plant Health Analysis for Early Disease Detection**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API](#-api-documentation) • [Model](#-model-training) • [Contributing](#-contributing)

</div>

---

## 🌟 Overview

The **Potato Disease Detection System** is an end-to-end AI-powered solution designed to assist farmers and agricultural experts in identifying potato plant diseases with high accuracy. Using state-of-the-art deep learning models, the system can classify potato leaf images into three categories:

- 🦠 **Early Blight** - Caused by *Alternaria solani*
- 🍂 **Late Blight** - Caused by *Phytophthora infestans*
- ✅ **Healthy** - No disease detected

The system provides real-time predictions through an intuitive web interface with bilingual support (English/Hindi), making it accessible to a wider range of users in agricultural communities.

---

## ✨ Features

### 🎯 Core Capabilities
- **Real-time Disease Detection** - Upload potato leaf images and get instant predictions
- **High Accuracy Model** - CNN-based architecture achieving **94.4% test accuracy**
- **Multi-class Classification** - Detects Early Blight, Late Blight, and Healthy plants
- **Confidence Scoring** - Provides prediction confidence percentage

### 🌐 User Experience
- **Bilingual Interface** - Seamless English ↔ Hindi translation
- **Responsive Design** - Works flawlessly on desktop, tablet, and mobile
- **Modern UI/UX** - Gradient backgrounds, smooth animations, toast notifications
- **Keyboard Shortcuts** - Quick actions with Ctrl+U (upload) and Escape (reset)
- **Image Preview** - Real-time preview before prediction

### 🛠️ Technical Features
- **TensorFlow Serving** - Optimized model deployment for production
- **RESTful API** - FastAPI backend with comprehensive endpoints
- **CORS Support** - Secure cross-origin resource sharing
- **Error Handling** - Robust validation and user-friendly error messages
- **Translation API** - Dynamic text translation with fallback mechanisms
- **Model Versioning** - Support for multiple model versions

### 🤖 AI Chatbot Features (Optional)
- **Ollama + Llama 3.2** - Local AI assistant for agricultural advice
- **Context-Aware** - Knows disease predictions and provides specific treatment recommendations
- **Bilingual Support** - Responds in English or Hindi based on user preference
- **Conversation Memory** - Maintains chat history within session for contextual responses
- **Quick Replies** - Pre-defined questions for instant answers
- **100% Free & Private** - No API costs, data stays local, works offline

---

## 🏗️ Architecture

### Modular Separated Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Web Browser                         │
│  ┌──────────────────┐       ┌────────────────────┐     │
│  │  index.html      │       │  chatbot.html      │     │
│  │  (Main App)      │◄─────►│  (iframe)          │     │
│  └────────┬─────────┘       └─────────┬──────────┘     │
└───────────┼───────────────────────────┼────────────────┘
            │                           │
            │ /predict                  │ /chat
            │ /translate                │ /chat/clear
            ▼                           ▼
┌───────────────────────────────────────────────────────┐
│              FastAPI Server (Port: 8000)              │
│  ┌──────────────────┐       ┌──────────────────┐     │
│  │   main_fixed.py  │       │   chatbot.py     │     │
│  │   (Core API)     │◄──────┤   (AI Module)    │     │
│  │                  │import │                  │     │
│  │ • /predict       │       │ • /chat          │     │
│  │ • /translate     │       │ • /chat/clear    │     │
│  │ • /ping          │       │ • Session Mgmt   │     │
│  └────┬─────────────┘       └─────────┬────────┘     │
└───────┼──────────────────────────────┼──────────────┘
        │                              │
        ▼                              ▼
┌─────────────────┐          ┌──────────────────┐
│  TensorFlow     │          │  Ollama + Llama  │
│  Serving        │          │  (Port: 11434)   │
│  (Port: 8501)   │          │                  │
│  • CNN Model    │          │  • Context-aware │
│  • Batch Pred   │          │  • Bilingual     │
└─────────────────┘          └──────────────────┘
        │
        ▼
┌─────────────────┐
│  Deep Translator│
│  (Google/Libre) │
└─────────────────┘
```

---

## � Project Structure

```
Minor-project/
├── backend/                      # FastAPI server
│   ├── main_fixed.py            # Main backend (RUNNING - with normalization fix)
│   ├── chatbot.py               # AI chatbot module
│   └── .env.example             # Environment variables template
├── frontend/                     # Web interface
│   ├── index.html               # Main UI
│   └── chatbot.html             # Chatbot UI
├── models/                       # Trained models
│   └── 3/                       # 94.4% accuracy model
├── dataset/                      # Training images
│   └── PlantVillage/
│       ├── Potato___Early_blight/
│       ├── Potato___Late_blight/
│       └── Potato___healthy/
├── model-training-improved.ipynb # Fixed training notebook
├── models.config                # TensorFlow Serving config
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Tech Stack

### Machine Learning
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing

### Backend
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Pillow (PIL)** - Image processing
- **Deep Translator** - Multi-language translation
- **TensorFlow Serving** - Model serving infrastructure
- **Ollama + Llama 3.2** - Local AI chatbot (optional)

### Frontend
- **HTML5** - Semantic markup
- **CSS3** - Modern styling with gradients, animations
- **JavaScript (ES6+)** - Dynamic interactions
- **Font Awesome 6.4** - Icon library

### DevOps
- **Docker** - Containerization (TensorFlow Serving)
- **Git** - Version control
- **Python 3.8+** - Runtime environment

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Docker (for TensorFlow Serving)
- Git
- Ollama (optional, for AI chatbot)

### Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start TensorFlow Serving (Docker)**
   ```powershell
   # Windows
   docker run -d -p 8501:8501 `
     --mount type=bind,source=C:\Users\sarva\Downloads\Minor-project\models\3,target=/models/potatoes_model/1 `
     -e MODEL_NAME=potatoes_model -t tensorflow/serving
   ```
   
   ```bash
   # Linux/Mac
   docker run -d -p 8501:8501 \
     --mount type=bind,source=$(pwd)/models/3,target=/models/potatoes_model/1 \
     -e MODEL_NAME=potatoes_model -t tensorflow/serving
   ```

3. **Start Ollama (Optional - for chatbot)**
   ```bash
   # Install from https://ollama.ai
   ollama pull llama3.2
   ollama serve
   ```

4. **Start Backend**
   ```bash
   cd backend
   python main_fixed.py
   ```

5. **Open Frontend**
   - Open `frontend/index.html` in your browser
   - Or use: `python -m http.server 5500` in frontend directory

6. **Verify System**
   ```bash
   cd scripts
   python system_status.py
   ```

---

## 💻 Usage

### Web Interface

1. **Open Application** - Navigate to `http://localhost:5500/index.html`
2. **Upload Image** - Click "Choose Image" or drag & drop a potato leaf image
3. **Get Prediction** - Click "Detect Disease" button
4. **View Results** - See disease class and confidence percentage
5. **Switch Language** - Click "हिन्दी" for Hindi or "English" to switch back
6. **Use AI Chatbot** - Click green button in bottom-right (if Ollama installed)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + U` | Upload new image |
| `Escape` | Reset/Clear current prediction |

---

## 📡 API Documentation

### Base URL: `http://localhost:8000`

#### Health Check
```http
GET /ping
```

#### Predict Disease
```http
POST /predict
Content-Type: multipart/form-data
Body: file (image)

Response:
{
  "class": "Early Blight",
  "class_index": 0,
  "confidence": 0.9876
}
```

#### Translate Text
```http
POST /translate
Content-Type: application/json

{
  "texts": ["Hello", "World"],
  "target": "hi"
}

Response:
{
  "translations": ["नमस्ते", "दुनिया"]
}
```

#### AI Chat (Optional - Requires Ollama)
```http
POST /chat
Content-Type: application/json

{
  "message": "What is Early Blight?",
  "session_id": "session_123",
  "language": "en"
}
```

---

## 🧠 Model Training

### Dataset
- **Source**: PlantVillage Dataset (Cornell University)
- **Total Images**: ~2,000
- **Classes**: Early Blight, Late Blight, Healthy
- **Split**: 70% Training, 15% Validation, 15% Testing

### Model Architecture (CNN)
```
Input (256x256x3)
  ↓
Conv2D (32 filters) + ReLU + MaxPooling
  ↓
Conv2D (64 filters) + ReLU + MaxPooling
  ↓
Conv2D (64 filters) + ReLU + MaxPooling
  ↓
Flatten → Dense (64) + ReLU → Dense (3) + Softmax
```

### Performance Metrics

| Metric | Value |
|--------|-------|
| Test Accuracy | 94.4% |
| Precision | 94.52% |
| Recall | 94.39% |
| F1 Score | 94.33% |

**Per-Class Accuracy:**
- Early Blight: 98.7% (148/150)
- Late Blight: 87.3% (131/150)
- Healthy: 97.3% (142/146)

### Training Configuration
- Batch Size: 32
- Image Size: 256x256
- Epochs: 100
- Optimizer: Adam
- Learning Rate: 0.001
- Data Augmentation: Random flip, rotation

### Retrain Model
1. Open `model-training-improved.ipynb` in Google Colab
2. Upload dataset or mount Google Drive
3. Run all cells (~30-60 min with T4 GPU)
4. Download model and place in `models/4/`
5. Update `models.config` and restart TensorFlow Serving

---

## 🛠️ Troubleshooting

### Common Issues

**Docker won't start?**
- Stop old containers: `docker stop $(docker ps -q)`
- Check path in mount command matches your system
- Verify port 8501 is free: `netstat -an | findstr 8501`

**Backend errors?**
- Check port 8000 is available
- Install all dependencies: `pip install -r requirements.txt`
- Verify TensorFlow Serving is running

**Chatbot not responding?**
- Check Ollama is running: `curl http://localhost:11434/api/version`
- Verify Llama 3.2 is installed: `ollama list`
- Check backend logs for errors

**Wrong predictions?**
- Ensure using `main_fixed.py` (has normalization fix)
- Verify model version 3 is loaded in Docker
- Check image is clear and well-lit

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Make your changes with clear commits
4. Push to your fork: `git push origin feature/AmazingFeature`
5. Open a Pull Request

Please ensure:
- Code follows existing style
- All tests pass
- Documentation is updated
- Commit messages are descriptive

---

## 📄 License

This project is licensed under the **MIT License** - see the LICENSE file for details.

---

## 👥 Authors

**Sarvagya Gupta** • **Sarwagya Shah** • **Ayush Prakash Tiwari**

- GitHub: [@sarvagya-019](https://github.com/sarvagya-019) • [@SARWAGYASHAH](https://github.com/SARWAGYASHAH) • [@AyushPrakash414](https://github.com/AyushPrakash414)
- Repository: [Minor-project](https://github.com/AyushPrakash414/Minor-project.git)

---

## 🙏 Acknowledgments

- **PlantVillage Dataset** - Cornell University for comprehensive potato disease dataset
- **TensorFlow Team** - Excellent deep learning framework
- **FastAPI** - Modern, fast web framework
- **Ollama & Llama 3.2** - Local AI capabilities

---

## 🔮 Future Enhancements

- [ ] Support for additional potato diseases (Blackleg, Common Scab)
- [ ] Mobile app (React Native/Flutter)
- [ ] Integration with weather APIs for risk prediction
- [ ] User authentication and prediction history
- [ ] Batch image processing
- [ ] Multi-crop support (tomato, pepper, etc.)
- [ ] Offline model support with TensorFlow Lite
- [ ] Voice input for chatbot

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for the agricultural community

**Version 2.0** • Last Updated: November 2025

</div>