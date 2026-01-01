# 💡 Idées & Fonctionnalités Futures

Liste des améliorations potentielles pour IA Réunions.

---

## 🎤 Diarisation & Reconnaissance de Locuteurs

### ✅ Diarisation basique (Speaker 1, Speaker 2, etc.) - IMPLÉMENTÉ
- **Outil utilisé** : NeMo (NVIDIA) avec TitaNet
- **Licence** : Apache 2.0 ✅ Commercial OK
- **Sortie** : Segments avec identification `SPEAKER_00`, `SPEAKER_01`, etc.
- **Endpoint** : `POST /transcribe` avec `diarize=true` (par défaut)

### Reconnaissance de locuteurs (avec BDD)
- **Concept** : Enrôler des personnes avec ~30 sec de leur voix
- **Outils** : 
  - SpeechBrain (Apache 2.0) ✅
  - Resemblyzer (Apache 2.0) ✅
  - NeMo TitaNet (Apache 2.0) ✅
- **Stockage** : PostgreSQL avec pgvector ou SQLite
- **Workflow** :
  1. `POST /speakers/enroll` → Enregistrer un locuteur avec son nom + audio
  2. `POST /transcribe` → Reconnaissance automatique des personnes connues
  3. Les inconnus restent `INCONNU_1`, `INCONNU_2`, etc.

---

## 📝 Résumé automatique

- Utiliser un LLM (Mistral, Llama, GPT) pour résumer la transcription
- Extraire les points clés, décisions, actions à faire
- Générer un compte-rendu structuré

---

## 📺 Export sous-titres

- Format **SRT** (compatible YouTube, VLC)
- Format **VTT** (web)
- Format **ASS** (stylisé)

---

## 🔍 Recherche dans les transcriptions

- Indexation full-text des transcriptions
- Recherche par mot-clé avec contexte
- Recherche sémantique avec embeddings

---

## 🌐 Interface Web

- Dashboard pour voir toutes les transcriptions
- Upload drag & drop
- Lecteur audio synchronisé avec la transcription
- Édition manuelle des segments

---

## 📊 Analytics

- Temps de parole par locuteur
- Mots les plus utilisés
- Sentiment analysis
- Détection de sujets/thèmes

---

## 🔗 Intégrations

- **Webhook** : Notification quand transcription terminée
- **Slack/Discord** : Poster le résumé automatiquement
- **Google Drive / Dropbox** : Sync des fichiers audio
- **Calendrier** : Lier aux événements de réunion

---

## 🏷️ Priorités

| Fonctionnalité | Priorité | Complexité |
|----------------|----------|------------|
| Diarisation basique | 🔴 Haute | Moyenne |
| Export SRT/VTT | 🟡 Moyenne | Facile |
| Résumé automatique | 🟡 Moyenne | Moyenne |
| Reconnaissance locuteurs (BDD) | 🟢 Basse | Élevée |
| Interface Web | 🟢 Basse | Élevée |
