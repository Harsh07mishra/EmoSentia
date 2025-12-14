# EmoSentia
🚀 EmoSentia — Real-Time Multimodal Emotion & Sentiment Detection
<p align="center"> <img src="https://img.shields.io/badge/AI-Emotion%20Recognition-blueviolet?style=for-the-badge" /> <img src="https://img.shields.io/badge/Multimodal-Analysis-brightgreen?style=for-the-badge" /> <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge" /> <img src="https://img.shields.io/badge/License-MIT-orange?style=for-the-badge" /> </p> <p align="center"> <b>EmoSentia</b> is an AI-powered system that performs <b>real-time emotion and sentiment detection</b> from video-based speech using <b>facial expressions, vocal signals, and NLP-driven text analysis</b>. </p>
📌 Key Highlights

🔥 Real-time multimodal emotion recognition

🎥 Facial expression detection using CNN / FER models

🎙️ Speech emotion classification via prosodic + deep audio features

📝 Sentiment scoring using transformers (BERT / Sentence-BERT)

🔗 Fusion model for unified emotional understanding

💡 Ideal for virtual interviews, e-learning, HCI systems, and mental health analytics

⚡ Optimized for real-time performance

🏗️ System Architecture
flowchart LR
    A[Video Input] --> B[Face Detection & Landmarks]
    B --> C[Facial Expression Model]

    A2[Audio Stream] --> D[Feature Extraction (Librosa/OpenSMILE)]
    D --> E[Speech Emotion Model]

    A3[ASR Module] --> F[Text Output]
    F --> G[NLP Sentiment Model]

    C --> H[Multimodal Fusion Engine]
    E --> H
    G --> H

    H --> I[Final Emotion + Sentiment Prediction]

    📊 Supported Datasets

Facial Expression: FER2013, AffectNet, RAF-DB

Speech Emotion: RAVDESS, CREMA-D, SAVEE

Text Sentiment: SST-2, IMDB, custom ASR transcripts

🚀 Future Enhancements

Cross-modal transformer-based fusion

Explainable AI for transparent scoring

Deployment on mobile/edge devices

Fairness-aware emotion modeling

Cultural adaptation modules

🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue first to discuss proposals.

👤 Author

Harsh Mishra
Infosys Springboard — AI/ML Intern
Project: EmoSentia: Real-Time Emotion & Sentiment Detection in Video Speech
