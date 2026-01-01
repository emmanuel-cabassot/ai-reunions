# 🎙️ IA Réunions

API de transcription audio intelligente utilisant **Whisper** (OpenAI) pour la transcription et **NeMo** (NVIDIA) pour la diarisation (identification des locuteurs).

Transformez vos enregistrements de réunions en texte avec :
- 🎯 Détection automatique de la langue
- 👥 Identification des locuteurs (Speaker 1, Speaker 2, etc.)
- ⏱️ Horodatage précis des segments
- 💾 Sauvegarde automatique des transcriptions

---

## 📋 Prérequis

| Composant | Version minimale | Vérification |
|-----------|------------------|--------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| nvidia-container-toolkit | 1.14+ | `dpkg -l \| grep nvidia-container-toolkit` |

> [!NOTE]
> Ce projet utilise le GPU pour accélérer les transcriptions et la diarisation. Une carte NVIDIA avec au moins 6GB de VRAM est recommandée.

---

## 🚀 Démarrage rapide

### 1. Cloner le projet

```bash
git clone <url-du-repo>
cd ia_reunions
```

### 2. Lancer l'application

```bash
# Build et démarre le conteneur (première fois : 10-15 min avec NeMo)
docker compose up --build
```

### 3. Accéder à l'API

- 🌐 **API** : http://localhost:8001
- 📚 **Documentation Swagger** : http://localhost:8001/docs
- 📖 **Documentation ReDoc** : http://localhost:8001/redoc

---

## 🎤 Fonctionnalités

### Transcription avec diarisation

```bash
curl -X POST "http://localhost:8001/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@reunion.mp3"
```

**Exemple de réponse :**

```json
{
  "success": true,
  "language": "fr",
  "speakers": ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
  "segments": [
    {"speaker": "SPEAKER_00", "start": 0.0, "end": 5.2, "text": "Bonjour à tous..."},
    {"speaker": "SPEAKER_01", "start": 5.2, "end": 12.5, "text": "Merci, passons au sujet..."},
    {"speaker": "SPEAKER_02", "start": 12.5, "end": 18.0, "text": "J'ai une question..."}
  ],
  "metadata": {
    "processing_time_seconds": 45.2,
    "speaker_count": 3,
    "audio_duration_seconds": 300.0,
    "speed_ratio": 6.64
  }
}
```

### Transcription sans diarisation (plus rapide)

```bash
curl -X POST "http://localhost:8001/transcribe?diarize=false" \
  -F "file=@audio.mp3"
```

### Lister les transcriptions sauvegardées

```bash
curl http://localhost:8001/transcriptions
```

### Récupérer une transcription spécifique

```bash
curl http://localhost:8001/transcriptions/20260101_120000_reunion.json
```

---

## 🐳 Commandes Docker

```bash
# Démarrer en arrière-plan
docker compose up -d

# Voir les logs
docker compose logs -f

# Redémarrer après modification
docker compose restart

# Arrêter
docker compose down
```

---

## ⚙️ Configuration

### Variables d'environnement

| Variable | Défaut | Description |
|----------|--------|-------------|
| `WHISPER_MODEL` | `base` | Modèle Whisper (tiny, base, small, medium, large-v3) |
| `ENABLE_DIARIZATION` | `true` | Activer/désactiver la diarisation |
| `LOG_LEVEL` | `INFO` | Niveau de log |

### Modèles Whisper disponibles

| Modèle | Paramètres | VRAM | Qualité |
|--------|------------|------|---------|
| `tiny` | 39M | ~1GB | ⭐ |
| `base` | 74M | ~1GB | ⭐⭐ |
| `small` | 244M | ~2GB | ⭐⭐⭐ |
| `medium` | 769M | ~5GB | ⭐⭐⭐⭐ |
| `large-v3` | 1550M | ~10GB | ⭐⭐⭐⭐⭐ |

---

## 📁 Structure du projet

```
ia_reunions/
├── main.py              # API FastAPI
├── requirements.txt     # Dépendances Python
├── Dockerfile          # Image Docker
├── docker-compose.yml  # Orchestration
├── README.md           # Ce fichier
├── ROADMAP.md          # Idées futures
└── transcriptions/     # Transcriptions sauvegardées
```

---

## 🔧 Développement

### Hot-reload activé

Le code est monté en volume, les modifications sont appliquées automatiquement.

### Ajouter une dépendance

1. Modifier `requirements.txt`
2. Rebuild : `docker compose up --build`

---

## 📜 Licences

| Composant | Licence | Usage commercial |
|-----------|---------|------------------|
| Whisper (OpenAI) | MIT | ✅ Libre |
| NeMo (NVIDIA) | Apache 2.0 | ✅ Libre |
| FastAPI | MIT | ✅ Libre |

---

## 🐛 Troubleshooting

### GPU non détecté

```bash
# Vérifier que le GPU est visible
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```

### Port déjà utilisé

Modifier le port dans `docker-compose.yml` :
```yaml
ports:
  - "8002:8000"  # Changer 8001 en 8002
```

### Mémoire GPU insuffisante

Utiliser un modèle plus petit :
```yaml
environment:
  - WHISPER_MODEL=tiny
```
