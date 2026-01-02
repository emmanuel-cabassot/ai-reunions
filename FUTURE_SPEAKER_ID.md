# Identification de Locuteurs avec WhisperX/pyannote (Projet Futur)

Ce document résume les possibilités techniques pour transformer la **diarisation** actuelle (SPEAKER_00, SPEAKER_01...) en **identification nominative** (ex: "Corine", "Jean").

## 🚀 Pourquoi WhisperX + pyannote est le bon choix ?

### 1. Simplicité d'installation
*   **Installation pip** : Pas besoin de conteneur NVIDIA NGC lourd.
*   **Modèles Hugging Face** : Téléchargement automatique des modèles pré-entraînés.
*   **Maintenance active** : WhisperX et pyannote sont activement maintenus.

### 2. Performance et Ressources
*   **VRAM légère** : ~3-4 Go pour transcription + diarisation.
*   **Alignement mot à mot** : Natif dans WhisperX, pas de post-traitement complexe.
*   **Qualité pyannote** : État de l'art pour la diarisation (DER < 10% sur benchmarks).

---

## 🛠️ Architecture de l'Identification (Enrôlement)

L'idée est de passer d'un système qui sépare les voix à un système qui les reconnaît.

### Étape 1 : Créer la Base de Données des Voix
Pour chaque personne que vous voulez reconnaître :
1.  Enregistrer un échantillon de 30-60 secondes de la personne seule.
2.  Extraire les embeddings avec `pyannote/embedding`.
3.  Récupérer un **Embedding** (un tableau de nombres qui représente la voix).
4.  Stocker ce tableau dans un fichier `speakers.json` :
    ```json
    {
      "Corine": [0.12, -0.45, 0.88, ...],
      "Jean": [0.33, 1.02, -0.12, ...]
    }
    ```

### Étape 2 : Inférence (Lors d'une réunion)
1.  **Diarisation WhisperX** : Le système sépare les voix en clusters (`SPEAKER_00`, `SPEAKER_01`).
2.  **Extraction** : Pour chaque cluster, on calcule son embedding moyen avec pyannote.
3.  **Comparaison (Similarité Cosine)** :
    *   On compare l'embedding de `SPEAKER_00` avec celui de "Corine" et "Jean" dans notre JSON.
    *   Si la ressemblance est > 80% avec "Corine", on remplace automatiquement `SPEAKER_00` par **Corine**.

---

## 📅 Prochaines étapes suggérées
Ce projet restant "de côté pour plus tard", voici les composants à implémenter le moment venu :
1.  **Route API `/enroll`** : Pour uploader un échantillon de voix et l'ajouter à la base.
2.  **Gestionnaire de base de données** : Un simple fichier JSON ou une base vectorielle légère (ChromaDB/FAISS) si vous avez des centaines de personnes.
3.  **Post-processing** : Une fonction dans `main.py` qui rebaptise les speakers avant de renvoyer la réponse finale.

---
*Document généré le 02/01/2026 pour le projet IA Réunions.*
