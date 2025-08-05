# Multimodal-RAG

A small project I built while preparing for AIC. It combines **retrieval-augmented generation (RAG)** with **multimodal data** (text, images, audio) to allow personalized retrieval tasks over your own dataset.

👉 Try the demo: [Hugging Face Space](https://huggingface.co/spaces/trungmin/Multimodal-RAG)

---

## 📦 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Multimodal-RAG.git
cd Multimodal_RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python main.py
```

After running, a local Gradio link will be generated (e.g. `http://127.0.0.1:7860`). Open it in your browser.

---

## 📁 Data Structure

Upload your own data in a `.zip` file with the following folder structure:

```
data.zip
├── images/
├── audios/
└── texts/
```

Each folder can be empty, but **must exist** for the app to work properly.
After uploading, the app will index your data for multimodal retrieval.
