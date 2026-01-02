# 🐳 Guide Docker - IA Réunions

Guide complet pour construire, lancer et gérer l'application via Docker.

---

## 📋 Prérequis

| Composant | Version minimale | Commande de vérification |
|-----------|------------------|--------------------------|
| Docker | 20.10+ | `docker --version` |
| Docker Compose | 2.0+ | `docker compose version` |
| NVIDIA Driver | 525+ | `nvidia-smi` |
| nvidia-container-toolkit | 1.14+ | `dpkg -l | grep nvidia-container-toolkit` |

> [!IMPORTANT]
> Le GPU NVIDIA est **obligatoire** pour cette application. Vérifiez avec `nvidia-smi` que votre carte est détectée.

---

## 🏗️ Construction de l'image

### Build standard

```bash
# Build l'image (première fois : 10-15 min)
docker compose build
```

### Build avec options

```bash
# Voir les logs détaillés pendant le build
docker compose build --progress=plain

# Rebuild complet (ignore le cache Docker)
docker compose build --no-cache

# Build + démarrage en une commande
docker compose up --build
```

### Vérifier que l'image est construite

```bash
docker images | grep ia_reunions
```

**Résultat attendu :**
```
ia_reunions-whisper-api   latest   abc123def456   2 hours ago   15.2GB
```

> [!NOTE]
> L'image fait ~8GB car elle inclut WhisperX et les modèles de diarisation pyannote.

---

## 🚀 Lancer le conteneur

### Démarrage simple

```bash
# Avec logs visibles (Ctrl+C pour arrêter)
docker compose up

# En arrière-plan (daemon)
docker compose up -d
```

### Vérifier que le conteneur tourne

```bash
# Liste des conteneurs actifs
docker ps
```

**Résultat attendu :**
```
CONTAINER ID   IMAGE                       STATUS         PORTS
abc123def456   ia_reunions-whisper-api     Up 2 minutes   0.0.0.0:8001->8000/tcp
```

### Tester que l'API fonctionne

```bash
# Test simple
curl http://localhost:8001/

# Réponse attendue :
# {"message":"🎙️ IA Réunions API","status":"running"...}
```

### Vérifier l'accès GPU

```bash
docker compose exec whisper-api nvidia-smi
```

---

## 📋 Voir les logs

```bash
# Tous les logs
docker compose logs

# Suivre en temps réel
docker compose logs -f

# Dernières 100 lignes seulement
docker compose logs --tail=100

# Logs d'un service spécifique
docker compose logs whisper-api
```

---

## 🛑 Arrêter le conteneur

```bash
# Arrêt simple
docker compose down

# Arrêt + suppression des volumes
# ⚠️ Ne supprime PAS les transcriptions (montées en volume local)
docker compose down -v
```

---

## 🔧 Commandes utiles

### Gestion du conteneur

```bash
# Redémarrer (après modif du code)
docker compose restart

# Entrer dans le conteneur (bash)
docker compose exec whisper-api bash

# Exécuter une commande dans le conteneur
docker compose exec whisper-api python --version
```

### Nettoyage

```bash
# Supprimer les images non utilisées
docker image prune

# Supprimer tout ce qui n'est pas utilisé (images, conteneurs, volumes)
docker system prune -a
```

---

## ⚠️ Dépannage

### Le conteneur ne démarre pas

```bash
# 1. Voir les logs d'erreur
docker compose logs

# 2. Vérifier l'état du conteneur
docker ps -a

# 3. Rebuild complet si nécessaire
docker compose build --no-cache
docker compose up
```

### GPU non détecté

```bash
# 1. Vérifier que le GPU est visible sur l'hôte
nvidia-smi

# 2. Tester Docker + GPU
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# 3. Si ça ne marche pas, installer nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

### Port 8001 déjà utilisé

```bash
# Voir quel processus utilise le port
lsof -i :8001

# Tuer le processus (remplacer PID)
kill -9 <PID>

# OU changer le port dans docker-compose.yml :
# ports:
#   - "8002:8000"  # Utiliser 8002 au lieu de 8001
```

### Erreur "out of memory" (GPU)

Utiliser un modèle Whisper plus petit :

```yaml
# Dans docker-compose.yml
environment:
  - WHISPER_MODEL=tiny   # Au lieu de base/small/medium
```

### L'API ne répond pas après le démarrage

Le chargement des modèles peut prendre **1-2 minutes** au premier lancement. Attendez de voir ce message dans les logs :

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 📊 Tableau récapitulatif

| Action | Commande |
|--------|----------|
| Build | `docker compose build` |
| Démarrer | `docker compose up -d` |
| Logs | `docker compose logs -f` |
| Arrêter | `docker compose down` |
| Redémarrer | `docker compose restart` |
| État | `docker ps` |
| Entrer dans le conteneur | `docker compose exec whisper-api bash` |
| Vérifier GPU | `docker compose exec whisper-api nvidia-smi` |

---

## 📁 Fichiers Docker du projet

| Fichier | Description |
|---------|-------------|
| `Dockerfile` | Définition de l'image (base PyTorch + WhisperX) |
| `docker-compose.yml` | Configuration du service (ports, GPU, volumes) |
| `.dockerignore` | Fichiers exclus du build |

---

## 🔗 Liens utiles

- [Documentation Docker Compose](https://docs.docker.com/compose/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [README principal](./README.md)
