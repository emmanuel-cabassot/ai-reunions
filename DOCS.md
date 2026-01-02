# 📚 Documentation et Ressources

Collection de liens utiles pour le projet IA Réunions.

---

## 🎤 WhisperX - Transcription & Diarisation

| Ressource | Description |
|-----------|-------------|
| [📖 GitHub - WhisperX](https://github.com/m-bain/whisperX) | Repo officiel WhisperX |
| [📦 PyPI - whisperx](https://pypi.org/project/whisperx/) | Package Python |
| [📄 Paper - WhisperX](https://arxiv.org/abs/2303.00747) | Article scientifique (alignement mot à mot) |

### Fonctionnalités WhisperX

| Fonctionnalité | Description |
|----------------|-------------|
| **Transcription** | Basée sur faster-whisper (CTranslate2) |
| **Alignement** | Alignement mot à mot précis |
| **Diarisation** | Via pyannote-audio |
| **Multi-GPU** | Support du traitement distribué |

---

## 🔊 pyannote-audio - Diarisation

| Ressource | Description |
|-----------|-------------|
| [📖 GitHub - pyannote-audio](https://github.com/pyannote/pyannote-audio) | Repo officiel pyannote |
| [🤗 Hugging Face Models](https://huggingface.co/pyannote) | Modèles pré-entraînés |
| [📄 Documentation](https://pyannote.github.io/) | Documentation complète |

### Modèles pyannote utilisés

| Modèle | Description | Usage |
|--------|-------------|-------|
| `pyannote/segmentation-3.0` | Segmentation des locuteurs | Détection des changements de voix |
| `pyannote/embedding` | Speaker Embedding | Extraction d'empreintes vocales |

> [!IMPORTANT]
> L'utilisation des modèles pyannote nécessite d'accepter les conditions sur Hugging Face et de configurer un token d'accès.

---

## 🎙️ OpenAI Whisper

| Ressource | Description |
|-----------|-------------|
| [📖 GitHub - Whisper](https://github.com/openai/whisper) | Repo officiel OpenAI Whisper |
| [📦 PyPI - openai-whisper](https://pypi.org/project/openai-whisper/) | Package Python |
| [📊 Model Card](https://github.com/openai/whisper/blob/main/model-card.md) | Détails des modèles disponibles |

### Modèles Whisper disponibles

| Modèle | Paramètres | VRAM | Vitesse |
|--------|-----------|------|---------|
| tiny | 39M | ~1GB | ~32x |
| base | 74M | ~1GB | ~16x |
| small | 244M | ~2GB | ~6x |
| medium | 769M | ~5GB | ~2x |
| large-v3 | 1550M | ~10GB | 1x |
| **turbo** | 809M | ~6GB | 6x ⭐ |

---

## 🐳 Docker & GPU

| Ressource | Description |
|-----------|-------------|
| [📦 PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch) | Images Docker PyTorch officielles |
| [📖 NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) | Installation GPU pour Docker |

---

*Dernière mise à jour : Janvier 2026*
