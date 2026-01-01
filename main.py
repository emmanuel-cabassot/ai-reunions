"""
🎙️ IA Réunions - API de Transcription Audio avec Diarisation
=============================================================
Utilise Whisper (OpenAI) pour transcrire et NeMo (NVIDIA) pour identifier les locuteurs.
"""

import os
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
import subprocess

import whisper
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

# ====== CONFIGURATION ======
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/transcriptions"))
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"

# Création du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== INITIALISATION ======
app = FastAPI(
    title="IA Réunions",
    description="API de transcription audio avec Whisper et diarisation NeMo",
    version="2.0.0"
)

# Détection GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

# Chargement du modèle Whisper au démarrage
logger.info(f"🔄 Chargement du modèle Whisper '{WHISPER_MODEL}' sur {DEVICE}...")
whisper_model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
logger.success(f"✅ Modèle Whisper '{WHISPER_MODEL}' chargé avec succès sur {DEVICE} !")

# Chargement du modèle NeMo pour la diarisation
nemo_diarizer = None
if ENABLE_DIARIZATION:
    try:
        from nemo.collections.asr.models import ClusteringDiarizer
        logger.info("🔄 Chargement du modèle NeMo pour la diarisation...")
        # Le modèle sera initialisé à la demande pour économiser la mémoire
        logger.success("✅ NeMo diarization disponible !")
    except ImportError as e:
        logger.warning(f"⚠️ NeMo non disponible, diarisation désactivée: {e}")
        ENABLE_DIARIZATION = False


def run_diarization(audio_path: str) -> List[Dict]:
    """
    Exécute la diarisation avec NeMo.
    Retourne une liste de segments avec speaker_id.
    """
    try:
        from omegaconf import OmegaConf
        from nemo.collections.asr.models import ClusteringDiarizer
        
        # Créer un dossier temporaire pour NeMo
        temp_dir = tempfile.mkdtemp()
        manifest_path = os.path.join(temp_dir, "manifest.json")
        output_dir = os.path.join(temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer le manifest pour NeMo
        manifest_entry = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "num_speakers": None,  # Auto-detect
            "rttm_filepath": None,
            "uem_filepath": None
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest_entry, f)
            f.write("\n")
        
        # Configuration NeMo
        config = OmegaConf.create({
            "diarizer": {
                "manifest_filepath": manifest_path,
                "out_dir": output_dir,
                "oracle_vad": False,
                "collar": 0.25,
                "ignore_overlap": True,
                "vad": {
                    "model_path": "vad_multilingual_marblenet",
                    "parameters": {
                        "onset": 0.8,
                        "offset": 0.6,
                        "min_duration_on": 0.1,
                        "min_duration_off": 0.1,
                    }
                },
                "speaker_embeddings": {
                    "model_path": "titanet_large",
                    "parameters": {
                        "window_length_in_sec": 1.5,
                        "shift_length_in_sec": 0.75,
                        "multiscale_weights": [1, 1, 1, 1, 1]
                    }
                },
                "clustering": {
                    "parameters": {
                        "oracle_num_speakers": False,
                        "max_num_speakers": 8,
                        "enhanced_count_thres": 80,
                        "max_rp_threshold": 0.25,
                        "sparse_search_volume": 30
                    }
                }
            }
        })
        
        # Initialiser et exécuter le diarizer
        diarizer = ClusteringDiarizer(cfg=config)
        diarizer.diarize()
        
        # Lire les résultats RTTM
        rttm_file = os.path.join(output_dir, "pred_rttms", 
                                  os.path.basename(audio_path).replace(
                                      os.path.splitext(audio_path)[1], ".rttm"))
        
        diarization_segments = []
        if os.path.exists(rttm_file):
            with open(rttm_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start = float(parts[3])
                        duration = float(parts[4])
                        speaker = parts[7]
                        diarization_segments.append({
                            "start": start,
                            "end": start + duration,
                            "speaker": speaker
                        })
        
        # Nettoyage
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return diarization_segments
        
    except Exception as e:
        logger.error(f"Erreur diarisation: {e}")
        return []


def merge_transcription_with_diarization(
    whisper_segments: List[Dict], 
    diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Fusionne les segments Whisper avec les informations de diarisation.
    """
    if not diarization_segments:
        # Pas de diarisation, retourner les segments originaux
        return [{"speaker": "SPEAKER_00", **seg} for seg in whisper_segments]
    
    merged = []
    for seg in whisper_segments:
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_mid = (seg_start + seg_end) / 2
        
        # Trouver le speaker qui parle au milieu du segment
        speaker = "UNKNOWN"
        for diar_seg in diarization_segments:
            if diar_seg["start"] <= seg_mid <= diar_seg["end"]:
                speaker = diar_seg["speaker"]
                break
        
        merged.append({
            "speaker": speaker,
            "start": seg_start,
            "end": seg_end,
            "text": seg["text"]
        })
    
    return merged


# ====== ROUTES ======

@app.get("/")
async def root():
    """Page d'accueil - Vérifie que l'API fonctionne."""
    return {
        "message": "🎙️ Bienvenue sur IA Réunions !",
        "status": "online",
        "model": WHISPER_MODEL,
        "device": DEVICE,
        "gpu": GPU_NAME,
        "diarization_enabled": ENABLE_DIARIZATION,
        "version": "2.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Endpoint de santé pour vérifier que tout fonctionne."""
    return {
        "status": "healthy",
        "model_loaded": whisper_model is not None,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "diarization_available": ENABLE_DIARIZATION
    }


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    save: bool = Query(True, description="Sauvegarder la transcription sur le disque"),
    diarize: bool = Query(True, description="Activer la diarisation (identification des locuteurs)")
):
    """
    📝 Transcrit un fichier audio en texte avec identification des locuteurs.
    
    Formats supportés : mp3, wav, m4a, ogg, flac, webm, mp4
    
    Retourne :
    - text : La transcription complète
    - language : La langue détectée
    - segments : Les segments avec timestamps ET speaker
    - speakers : Liste des locuteurs détectés
    """
    
    # Vérification du type de fichier
    allowed_extensions = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {file_ext}. Formats acceptés : {', '.join(allowed_extensions)}"
        )
    
    logger.info(f"📁 Fichier reçu : {file.filename} ({file.content_type})")
    
    try:
        # Sauvegarde temporaire du fichier
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
            file_size = len(content)
        
        # Mesure du temps de traitement
        start_time = time.time()
        
        # ====== ÉTAPE 1: Transcription Whisper ======
        logger.info(f"🎵 Transcription Whisper en cours sur {DEVICE}...")
        whisper_result = whisper_model.transcribe(
            temp_path,
            language=None,
            task="transcribe"
        )
        whisper_time = time.time() - start_time
        logger.success(f"✅ Transcription Whisper terminée en {whisper_time:.2f}s")
        
        # ====== ÉTAPE 2: Diarisation NeMo (optionnel) ======
        diarization_segments = []
        diarization_time = 0
        speakers_detected = []
        
        if diarize and ENABLE_DIARIZATION:
            logger.info("🎤 Diarisation NeMo en cours...")
            diar_start = time.time()
            diarization_segments = run_diarization(temp_path)
            diarization_time = time.time() - diar_start
            speakers_detected = list(set(seg["speaker"] for seg in diarization_segments))
            logger.success(f"✅ Diarisation terminée en {diarization_time:.2f}s - {len(speakers_detected)} locuteurs détectés")
        
        # ====== ÉTAPE 3: Fusion des résultats ======
        whisper_segments = [
            {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
            for seg in whisper_result["segments"]
        ]
        
        merged_segments = merge_transcription_with_diarization(
            whisper_segments, diarization_segments
        )
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Calcul des statistiques
        audio_duration = whisper_result["segments"][-1]["end"] if whisper_result["segments"] else 0
        word_count = len(whisper_result["text"].split())
        segment_count = len(merged_segments)
        
        # Nettoyage du fichier temporaire
        os.unlink(temp_path)
        
        # Préparation de la réponse
        response_data = {
            "success": True,
            "filename": file.filename,
            "language": whisper_result["language"],
            "text": whisper_result["text"],
            "speakers": speakers_detected if speakers_detected else ["SPEAKER_00"],
            "segments": merged_segments,
            "metadata": {
                "processing_time_seconds": processing_time,
                "whisper_time_seconds": round(whisper_time, 2),
                "diarization_time_seconds": round(diarization_time, 2),
                "audio_duration_seconds": round(audio_duration, 2),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "word_count": word_count,
                "segment_count": segment_count,
                "speaker_count": len(speakers_detected) if speakers_detected else 1,
                "model_used": WHISPER_MODEL,
                "diarization_enabled": diarize and ENABLE_DIARIZATION,
                "device": DEVICE,
                "gpu": GPU_NAME,
                "speed_ratio": round(audio_duration / processing_time, 2) if processing_time > 0 else 0,
                "transcribed_at": datetime.now().isoformat()
            }
        }
        
        # Sauvegarde sur le disque si demandé
        if save:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = Path(file.filename).stem
            output_filename = f"{timestamp}_{base_name}.json"
            output_path = OUTPUT_DIR / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Transcription sauvegardée : {output_path}")
            response_data["saved_to"] = str(output_path)
        
        return response_data
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de la transcription : {e}")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transcriptions")
async def list_transcriptions():
    """📂 Liste toutes les transcriptions sauvegardées."""
    transcriptions = []
    
    for file_path in sorted(OUTPUT_DIR.glob("*.json"), reverse=True):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                transcriptions.append({
                    "filename": file_path.name,
                    "original_file": data.get("filename", "unknown"),
                    "language": data.get("language", "unknown"),
                    "speakers": data.get("speakers", []),
                    "speaker_count": data.get("metadata", {}).get("speaker_count", 1),
                    "duration": data.get("metadata", {}).get("audio_duration_seconds", 0),
                    "word_count": data.get("metadata", {}).get("word_count", 0),
                    "transcribed_at": data.get("metadata", {}).get("transcribed_at", "unknown"),
                    "path": str(file_path)
                })
        except Exception as e:
            logger.warning(f"Erreur lecture {file_path}: {e}")
    
    return {
        "count": len(transcriptions),
        "transcriptions": transcriptions
    }


@app.get("/transcriptions/{filename}")
async def get_transcription(filename: str):
    """📄 Récupère une transcription sauvegardée par son nom de fichier."""
    file_path = OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Transcription non trouvée : {filename}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/models")
async def list_models():
    """Liste les modèles disponibles et leur configuration."""
    return {
        "whisper": {
            "current_model": WHISPER_MODEL,
            "device": DEVICE,
            "gpu": GPU_NAME,
            "available_models": {
                "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x"},
                "base": {"params": "74M", "vram": "~1GB", "speed": "~16x"},
                "small": {"params": "244M", "vram": "~2GB", "speed": "~6x"},
                "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x"},
                "large": {"params": "1550M", "vram": "~10GB", "speed": "1x"},
                "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "1x"},
                "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "1x"}
            }
        },
        "diarization": {
            "enabled": ENABLE_DIARIZATION,
            "engine": "NeMo" if ENABLE_DIARIZATION else None,
            "models": {
                "vad": "vad_multilingual_marblenet",
                "speaker_embedding": "titanet_large"
            }
        }
    }


# ====== DÉMARRAGE ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
