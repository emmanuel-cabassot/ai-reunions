# 🏗️ Architecture Technique (Vision Fin 2025)

Ce document résume la stratégie architecturale pour la transcription et l'identification du locuteur locale, optimisée pour une contrainte de **12 Go de VRAM**.

## 🎯 Objectif Stratégique
Atteindre une précision de niveau industriel tout en garantissant la confidentialité et une latence faible sur du matériel grand public (RTX 3060/4070).

## 🛠️ La "Winning Stack" (Architecture Modulaire)

L'approche modulaire est privilégiée par rapport aux modèles unifiés (type Canary) pour sa flexibilité et sa gestion fine des identités.

### 1. Transcription (ASR) : Whisper Large v3 Turbo
*   **Pourquoi ?** Le modèle Whisper Large v3 standard occupe ~10 Go de VRAM. La version **Turbo** réduit le nombre de couches du décodeur (32 → 4), offrant une vitesse **6x supérieure** pour une perte de précision négligeable (< 1% WER).
*   **Optimisation :** Utilisation de `faster-whisper` avec quantification **INT8** pour descendre l'empreinte mémoire à **~2-3 Go**.

### 2. Diarisation & Identification : WhisperX + pyannote
*   **VAD (Voice Activity Detection) :** Intégré dans WhisperX via pyannote.
*   **Segmentation :** Modèle pyannote/segmentation-3.0 pour la détection des changements de locuteur.
*   **Embeddings :** pyannote/embedding pour les empreintes vocales robustes.
*   **Clustering :** Regroupement automatique pour identifier les locuteurs distincts.

### 3. Gestion des Identités : Base Vectorielle
*   **Stockage :** `ChromaDB` (léger, local) pour stocker les centroïdes vocaux.
*   **Identification :** Comparaison par **Similarité Cosinus** entre le segment actuel et la base d'enrôlement.

---

## 📊 Comparaison des Approches

| Critère | WhisperX (Choisi) | Stack NeMo |
| :--- | :--- | :--- |
| **VRAM** | ~3-4 Go | ~8-10 Go |
| **Diarisation** | Précise (pyannote) | Complexe à configurer |
| **Installation** | Simple (pip) | Lourde (NGC container) |
| **Alignement** | Word-level natif | Post-traitement requis |

## 🚀 Pipeline d'Enrôlement (Enrollment)
1. **Prétraitement :** Audio 16kHz Mono, normalisé à -20 LUFS.
2. **Filtrage :** VAD strict pour éliminer le silence (qui dilue l'identité).
3. **Agrégation :** Calcul du centroïde sur 3 échantillons de 10s pour une empreinte stable.
4. **Validation :** Rejet si la variance inter-échantillons est trop élevée.

---
> [!TIP]
> Cette architecture permet de traiter des réunions complexes avec plusieurs locuteurs tout en restant fluide sur une station de travail locale.
