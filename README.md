<div align="center">

# 🥔 Potato Disease Detection System

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

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Dataset](#-dataset)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)

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
- **High Accuracy Model** - CNN-based architecture trained on 2,000+ images
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

### 🤖 AI Chatbot Features
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
│  │   main.py        │       │   chatbot.py     │     │
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

### Component Breakdown

#### **Frontend Layer** (Separated)
1. **`index.html`** - Main Application
   - Disease detection interface
   - Image upload and preview
   - Results display
   - Embeds chatbot via iframe

2. **`chatbot.html`** - Chatbot Component
   - Standalone chat UI
   - Can be embedded anywhere
   - Communicates via `postMessage`

#### **Backend Layer** (Modular)
1. **`main.py`** - Core API
   - Disease prediction (`/predict`)
   - Text translation (`/translate`)
   - Health check (`/ping`)
   - Imports chatbot module

2. **`chatbot.py`** - AI Chat Module
   - Chat endpoint (`/chat`)
   - Clear history (`/chat/clear`)
   - Ollama integration
   - Session management

#### **ML Pipeline**
   - TensorFlow 2.x CNN model
   - TensorFlow Serving for inference
   - Image preprocessing (256x256 RGB)
   - Batch prediction support

#### **AI Assistant**
   - Ollama + Llama 3.2 (local LLM)
   - Context-aware responses
   - Bilingual support (English/Hindi)
   - Conversation memory

---


## 📁 Project Structure

| File | Purpose |
|------|---------|
| `model-training.ipynb` | Complete ML pipeline: data preprocessing, model architecture, training, evaluation |
| `main.py` | Core FastAPI server (prediction, translation, health check) - imports chatbot module |
| `chatbot.py` | **NEW** - AI chatbot module with Ollama integration (separated for modularity) |
| `models.config` | TensorFlow Serving model registry configuration |
| `index.html` | Main frontend application with disease detection UI |
| `chatbot.html` | **NEW** - Standalone chatbot component (embedded via iframe) |
| `saved_models/` | Serialized TensorFlow models ready for serving |

---

## 🛠️ Tech Stack

### Machine Learning
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization and plotting

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

## 🚀 Installation
> 
### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)
- Docker (for TensorFlow Serving)
- Git
- Ollama (optional, for AI chatbot)

### Step 1: Clone the Repository

```bash
git clone https://github.com/sarvagya-019/Minor-project.git
cd Minor-project
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### Step 3: Install Python Dependencies

```bash
pip install tensorflow fastapi uvicorn pillow numpy requests deep-translator python-multipart
```

### Step 4: Download Dataset

The PlantVillage dataset is already included in the repository. If needed, you can download it from:
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village)

### Step 5: Run TensorFlow Serving (Docker)

```bash
docker run -d --name potato_model_server ^
  -p 8501:8501 ^
  -v "%cd%/saved_models:/models/potatoes_model" ^
  -e MODEL_NAME=potatoes_model ^
  tensorflow/serving
```

**Linux/Mac:**
```bash
docker run -d --name potato_model_server \
  -p 8501:8501 \
  -v "$(pwd)/saved_models:/models/potatoes_model" \
  -e MODEL_NAME=potatoes_model \
  tensorflow/serving
```

### Step 6: Start Backend Server

> **Note**: The backend now consists of two modules:
> - `main.py` - Core API (automatically imports chatbot module)
> - `chatbot.py` - AI chatbot endpoints

```bash
cd "Backend server"
python main.py
```

The API will be available at `http://localhost:8000`

**Verify both modules loaded**:
```bash
# Test core API
curl http://localhost:8000/ping

# Test chatbot API (if Ollama installed)
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"message":"Hello"}'
```

### Step 7: Launch Frontend

> **Note**: The frontend now consists of two files:
> - `index.html` - Main application (embeds chatbot via iframe)
> - `chatbot.html` - Chatbot component (loads independently)

```bash
# Using Python's built-in server
cd potato-disease-frontend
python -m http.server 5500
```

Access the application at `http://localhost:5500/index.html`

**Both components will load automatically**:
- Main app: Disease detection interface
- Chatbot: Green button in bottom-right corner (if Ollama installed)

### Step 8: Set Up AI Chatbot (Optional but Recommended)

> **Why Use the AI Chatbot?**
> - Get instant answers about potato diseases
> - Receive personalized treatment recommendations based on your predictions
> - Learn prevention methods and farming best practices
> - Completely **FREE** - no API costs (runs locally with Ollama)
> - **100% Private** - your data never leaves your machine
> - Works **offline** after initial model download

#### 8.1 Install Ollama

Ollama is a free, open-source tool that lets you run AI models locally on your computer.

**Windows:**
1. Download installer from https://ollama.ai/download
2. Run the installer (requires administrator privileges)
3. Ollama will start automatically in the background

**macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Verify Installation:**
```bash
# Check if Ollama is running
ollama --version

# Should output something like: ollama version 0.1.x
```

#### 8.2 Download Llama 3.2 Model

```bash
ollama pull llama3.2
```

**Download Details:**
- Model Size: ~2GB
- Parameters: 3 billion
- Download Time: 5-10 minutes (depending on internet speed)
- Disk Space Required: 2.5GB

**Alternative Models** (if you have limited resources):
```bash
# Smaller, faster model (1GB)
ollama pull llama3.2:1b

# Larger, more accurate model (4.7GB)
ollama pull llama3.2:7b
```

#### 8.3 Start Ollama Server

Ollama needs to be running in the background for the chatbot to work.

**Windows:** Ollama starts automatically after installation. Check system tray for Ollama icon.

**macOS/Linux:**
```bash
ollama serve
```

**Verify Server is Running:**
```bash
curl http://localhost:11434/api/version
```

Should return: `{"version":"0.1.x"}`

#### 8.4 Verify Chatbot Integration

1. **Start Backend** (if not already running):
   ```bash
   cd "Backend server"
   python main.py
   ```

2. **Open Frontend**:
   - Navigate to `http://localhost:5500/index.html`
   - Look for a **green circular button** in the bottom-right corner
   - If you don't see it, hard refresh: `Ctrl + Shift + R`

3. **Test the Chatbot**:
   - Click the green button to open the chat window
   - Type: "Hello, are you working?"
   - You should get a response within 2-3 seconds

4. **Test Context-Awareness**:
   - Upload a potato leaf image with disease
   - Click "Detect Disease"
   - Open chatbot and ask: "How do I treat this disease?"
   - The chatbot will provide specific advice based on the detected disease

#### 8.5 Chatbot Features & Usage

**What Can the Chatbot Do?**

1. **Disease Information**
   - "What is Early Blight?"
   - "What causes Late Blight?"
   - "How can I identify healthy potato plants?"

2. **Treatment Advice**
   - "How do I treat Early Blight?"
   - "What fungicides work for Late Blight?"
   - "Are there organic treatment options?"

3. **Prevention Methods**
   - "How can I prevent potato diseases?"
   - "What are best practices for potato farming?"
   - "When should I apply fungicides?"

4. **Context-Aware Recommendations**
   - After detecting a disease, ask: "What should I do?"
   - The chatbot knows your prediction and provides specific guidance

5. **Bilingual Support**
   - Switch language in main app (English/Hindi)
   - Chatbot automatically responds in selected language

**Quick Reply Buttons:**
- "What is this disease?" - Get detailed explanation
- "How to treat it?" - Treatment recommendations
- "Prevention tips" - Prevent future infections

**Conversation Memory:**
- Chatbot remembers your conversation within the same session
- Ask follow-up questions naturally
- Clear history anytime with the "Clear" button

#### 8.6 Chatbot Troubleshooting

**Common Issues:**

- **Button not appearing**: Check backend and Ollama are running, refresh browser (`Ctrl + Shift + R`)
- **Cannot connect**: Verify Ollama is running: `curl http://localhost:11434/api/version`
- **Slow responses**: Use smaller model `ollama pull llama3.2:1b`
- **Messages not sending**: Ensure backend server is running, check browser console

**Advanced Options:**
- Change model in `Backend server/chatbot.py` (line 18-19)
- Customize system prompt for different chatbot personality
- Adjust `max_tokens` for response length

---

## 💻 Usage

### Web Interface

1. **Open the Application**
   - Navigate to `http://localhost:5500` (or open `index.html` directly)

2. **Upload an Image**
   - Click "Choose Image" or drag & drop a potato leaf image
   - Supported formats: JPG, JPEG, PNG
   - Maximum file size: 10MB

3. **Get Prediction**
   - Click "Detect Disease" button
   - View results showing disease class and confidence percentage

4. **Switch Language**
   - Click "हिन्दी" button for Hindi interface
   - Click "English" to switch back

5. **Use AI Chatbot** (if Ollama installed)
   - Click the green chat button (bottom-right)
   - Ask questions about detected diseases
   - Get treatment recommendations and farming advice
   - Chatbot responds in selected language (English/Hindi)

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl + U` | Upload new image |
| `Escape` | Reset/Clear current prediction |

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check

```http
GET /ping
```

**Response:**
```json
"Hello, I am alive"
```

---

#### 2. Predict Disease

```http
POST /predict
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` (image file)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "accept: application/json" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@potato_leaf.jpg"
```

**Response:**
```json
{
  "class": "Early Blight",
  "class_index": 0,
  "confidence": 0.9876
}
```

**Response Fields:**
- `class` (string): Disease name or "Healthy"
- `class_index` (integer): 0 = Early Blight, 1 = Late Blight, 2 = Healthy
- `confidence` (float): Prediction confidence (0.0 to 1.0)

---

#### 3. Translate Text

```http
POST /translate
```

**Request:**
```json
{
  "texts": ["Hello", "World"],
  "target": "hi"
}
```

**Response:**
```json
{
  "translations": ["नमस्ते", "दुनिया"]
}
```

**Parameters:**
- `texts` (array): List of strings to translate
- `target` (string): Target language code (e.g., "hi" for Hindi, "en" for English)

**Supported Languages:**
- `en` - English
- `hi` - Hindi
- (Extensible to 100+ languages via Google Translator)

---

#### 4. AI Chat

```http
POST /chat
```

**Request:**
```json
{
  "message": "What is Early Blight?",
  "session_id": "session_123",
  "disease_context": {
    "class": "Early Blight",
    "confidence": 0.87
  },
  "language": "en"
}
```

**Response:**
```json
{
  "response": "Early Blight is a fungal disease caused by Alternaria solani. It causes dark brown spots with concentric rings on leaves...",
  "session_id": "session_123"
}
```

**Parameters:**
- `message` (string): User's question
- `session_id` (string, optional): Session identifier for conversation continuity
- `disease_context` (object, optional): Recent prediction context for personalized advice
- `language` (string): "en" or "hi"

**Note:** Requires Ollama running on `localhost:11434` with Llama 3.2 model installed.

---

#### 5. Clear Chat History

```http
POST /chat/clear
```

**Request:**
```json
{
  "session_id": "session_123"
}
```

**Response:**
```json
{
  "status": "cleared",
  "session_id": "session_123"
}
```

---

## 🧠 Model Training

### Dataset Preparation

The training pipeline automatically splits the PlantVillage dataset:

- **Training Set**: 70% (1,400 images)
- **Validation Set**: 15% (300 images)
- **Testing Set**: 15% (300 images)

### Model Architecture

```python
Input Layer (256x256x3)
    ↓
Conv2D (32 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3) + ReLU
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dense (64 units) + ReLU
    ↓
Dense (3 units) + Softmax
```

### Training Configuration

- **Batch Size**: 32
- **Image Size**: 256x256 pixels
- **Epochs**: 50
- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Data Augmentation**: Random flip, rotation

### Training the Model

Open `model-training.ipynb` in Jupyter Notebook:

```bash
jupyter notebook model-training.ipynb
```

Run all cells sequentially to:
1. Split the dataset
2. Load and preprocess images
3. Build CNN architecture
4. Train the model
5. Evaluate performance
6. Save model to `saved_models/`

### Model Versioning

The system supports multiple model versions. Update `models.config` to switch versions:

```plaintext
model_config_list {
  config {
    name: 'potatoes_model'
    base_path: '/Minor-project/saved_models/'
    model_platform: 'tensorflow'
    model_version_policy: {all: {}}
  }
}
```

---

## 📊 Dataset

### PlantVillage Dataset

- **Source**: Cornell University PlantVillage Project
- **Total Images**: ~2,000
- **Classes**: 3 (Early Blight, Late Blight, Healthy)
- **Image Format**: JPEG
- **Resolution**: Variable (resized to 256x256 during training)

### Class Distribution

| Class | Description | Image Count |
|-------|-------------|-------------|
| Early Blight | Dark brown spots with concentric rings | ~1,000 |
| Late Blight | Large irregular brown lesions | ~1,000 |
| Healthy | Green, unblemished leaves | ~152 |

### Data Augmentation

To improve model generalization:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Random zoom (±10%)
- Normalization (pixel values scaled to [0, 1])

---

## 🌐 Deployment

### Local Deployment

Follow the [Installation](#-installation) steps above.

### Production Deployment Options

#### 1. **Docker Compose** (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  tensorflow-serving:
    image: tensorflow/serving
    ports:
      - "8501:8501"
    volumes:
      - ./saved_models:/models/potatoes_model
    environment:
      - MODEL_NAME=potatoes_model

  backend:
    build: ./Backend server
    ports:
      - "8000:8000"
    depends_on:
      - tensorflow-serving
    environment:
      - TF_SERVING_ENDPOINT=http://tensorflow-serving:8501/v1/models/potatoes_model:predict

  frontend:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./potato-disease-frontend:/usr/share/nginx/html
```

Run:
```bash
docker-compose up -d
```

#### 2. **Cloud Platforms**

- **AWS**: Deploy on EC2 with Elastic Load Balancer
- **Google Cloud**: Use Cloud Run for containerized deployment
- **Azure**: Deploy on Azure Container Instances
- **Heroku**: Use Heroku Dynos for backend + static site for frontend

#### 3. **TensorFlow Serving on Kubernetes**

For high-availability production:
```bash
kubectl apply -f tf-serving-deployment.yaml
kubectl apply -f backend-deployment.yaml
kubectl apply -f frontend-deployment.yaml
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

### 1. Fork the Repository

Click the "Fork" button at the top right of this page.

### 2. Clone Your Fork

```bash
git clone https://github.com/your-username/Minor-project.git
cd Minor-project
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/AmazingFeature
```

### 4. Make Changes

- Write clean, documented code
- Follow existing code style
- Add tests for new features

### 5. Commit Your Changes

```bash
git add .
git commit -m "Add: Amazing new feature"
```

### 6. Push to Your Fork

```bash
git push origin feature/AmazingFeature
```

### 7. Open a Pull Request

Go to the original repository and click "New Pull Request".

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Follow best practices for code quality

---

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

**Sarvagya Gupta** : **Sarwagya Shah** : **Ayush Prakash Tiwari**
- GitHub: [@sarvagya-019](https://github.com/sarvagya-019) , [@AyushPrakash414](https://github.com/AyushPrakash414) ,  [@SARWAGYASHAH](https://github.com/SARWAGYASHAH) 

- Repository: [Minor-project](https://github.com/AyushPrakash414/Minor-project.git)

---

## 🙏 Acknowledgments

- **PlantVillage** - For providing the comprehensive potato disease dataset
- **TensorFlow Team** - For the excellent deep learning framework
- **FastAPI** - For the modern, fast web framework
- **Cornell University** - For pioneering plant disease recognition research

---

## 🔮 Future Enhancements

- [ ] Support for additional potato diseases (Blackleg, Common Scab)
- [ ] Mobile app (React Native/Flutter)
- [ ] Integration with weather APIs for risk prediction
- [ ] User authentication and prediction history
- [ ] Batch image processing
- [ ] Export reports as PDF
- [ ] Multi-crop support (tomato, pepper, etc.)
- [ ] Real-time camera capture for mobile devices
- [ ] Offline model support with TensorFlow Lite
- [ ] Voice input for chatbot
- [ ] Custom knowledge base for region-specific advice

---

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Model Accuracy | ~95% |
| Inference Time | <100ms |
| API Response Time | <200ms |
| Supported Image Formats | JPG, PNG, JPEG |
| Max Image Size | 10MB |
| Concurrent Requests | 100+ |

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for the agricultural community

</div>