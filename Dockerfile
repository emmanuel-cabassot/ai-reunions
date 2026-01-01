# 🎙️ IA Réunions - Dockerfile avec support GPU et NeMo Diarization
# ===================================================================

# 1. Image officielle NeMo dev de NVIDIA (inclut NeMo pré-configuré avec CUDA)
FROM nvcr.io/nvidia/nemo:dev

# OPTIMISATION : Force Python à afficher les logs instantanément
ENV PYTHONUNBUFFERED=1

# Désactiver les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive

# 2. Installation de ffmpeg pour Whisper
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3. Préparation du plan de travail
WORKDIR /app

# 4. Installation des dépendances Python additionnelles
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    python-multipart>=0.0.6 \
    openai-whisper>=20231117 \
    pydub>=0.25.1 \
    python-dotenv>=1.0.0 \
    loguru>=0.7.0

# 5. Ajout du code
COPY . .

# 6. Créer le dossier de transcriptions
RUN mkdir -p /app/transcriptions

# 7. La commande de service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]