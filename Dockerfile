# 🎙️ IA Réunions - Dockerfile avec WhisperX
# =============================================

# 1. Image PyTorch officielle 2.5 avec CUDA 12.4 et cuDNN 9 intégré
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# 2. Installation des dépendances système
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Préparation du répertoire de travail
WORKDIR /app

# 4. Copie des requirements et installation
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Installation de WhisperX (dernière version git)
# Note: On ignore les dépendances torch car PyTorch est déjà installé
RUN pip install --no-cache-dir --no-deps git+https://github.com/m-bain/whisperX.git

# 6. Installation des dépendances de WhisperX (sans torch)
RUN pip install --no-cache-dir \
    faster-whisper>=1.1.1 \
    ctranslate2>=4.5.0 \
    pyannote-audio>=3.3.2 \
    transformers>=4.36.0

# 7. Copie du code source
COPY . .

# 8. Création du dossier de transcriptions
RUN mkdir -p /app/transcriptions

# 9. Exposition du port
EXPOSE 8000

# 10. Commande de démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]