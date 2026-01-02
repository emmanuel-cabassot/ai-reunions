"""
🎙️ IA Réunions - API de Transcription Audio avec Diarisation
=============================================================
Utilise WhisperX pour la transcription et pyannote pour la diarisation.
"""

import os
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from loguru import logger

# ====== FIX PYTORCH 2.6+ COMPATIBILITY ======
# PyTorch 2.6+ a changé weights_only=True par défaut, ce qui casse pyannote
# On force weights_only=False pour les modèles de confiance (pyannote, whisperx)
_torch_load_original = torch.load
def _torch_load_patched(*args, **kwargs):
    kwargs['weights_only'] = False  # Force toujours False
    return _torch_load_original(*args, **kwargs)
torch.load = _torch_load_patched
# ====== CONFIGURATION ======
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/transcriptions"))
ENABLE_DIARIZATION = os.getenv("ENABLE_DIARIZATION", "true").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Création du dossier de sortie
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== INITIALISATION ======
app = FastAPI(
    title="IA Réunions",
    description="API de transcription audio avec WhisperX et diarisation pyannote",
    version="3.0.0"
)

# Détection GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

# Chargement de WhisperX au démarrage
whisperx_model = None
align_model = None
align_metadata = None
diarize_model = None

def load_models():
    """Charge les modèles WhisperX et pyannote."""
    global whisperx_model, align_model, align_metadata, diarize_model
    
    import whisperx
    
    # 1. Modèle de transcription WhisperX
    logger.info(f"🔄 Chargement du modèle WhisperX '{WHISPER_MODEL}' sur {DEVICE}...")
    whisperx_model = whisperx.load_model(
        WHISPER_MODEL, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE
    )
    logger.success(f"✅ Modèle WhisperX '{WHISPER_MODEL}' chargé !")
    
    # 2. Modèle de diarisation (si token HF disponible)
    if ENABLE_DIARIZATION and HF_TOKEN:
        logger.info("🔄 Chargement du modèle de diarisation pyannote...")
        try:
            from whisperx.diarize import DiarizationPipeline
            diarize_model = DiarizationPipeline(
                use_auth_token=HF_TOKEN,
                device=DEVICE
            )
            logger.success("✅ Modèle de diarisation pyannote chargé !")
        except Exception as e:
            logger.error(f"❌ Erreur chargement diarisation: {e}")
            logger.warning("⚠️ Diarisation désactivée")
    elif ENABLE_DIARIZATION and not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN non défini - Diarisation désactivée")
        logger.info("💡 Définissez HF_TOKEN dans docker-compose.yml pour activer la diarisation")

# Charger les modèles au démarrage
load_models()


def transcribe_with_whisperx(audio_path: str, diarize: bool = True) -> Dict:
    """
    Transcrit un fichier audio avec WhisperX.
    Retourne la transcription avec alignement et diarisation.
    """
    import whisperx
    
    # 1. Charger l'audio
    audio = whisperx.load_audio(audio_path)
    
    # 2. Transcription
    logger.info(f"🎵 Transcription en cours sur {DEVICE}...")
    result = whisperx_model.transcribe(audio, batch_size=16)
    detected_language = result["language"]
    logger.info(f"📝 Langue détectée: {detected_language}")
    
    # 3. Alignement (word-level timestamps)
    logger.info("🔄 Alignement des mots...")
    global align_model, align_metadata
    
    # Charger le modèle d'alignement pour cette langue
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language, 
        device=DEVICE
    )
    result = whisperx.align(
        result["segments"], 
        align_model, 
        align_metadata, 
        audio, 
        DEVICE,
        return_char_alignments=False
    )
    
    # 4. Diarisation (si activée et disponible)
    speakers_detected = []
    if diarize and diarize_model is not None:
        logger.info("🎤 Diarisation en cours...")
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Extraire les locuteurs uniques
        speakers_detected = list(set(
            seg.get("speaker", "SPEAKER_00") 
            for seg in result["segments"] 
            if seg.get("speaker")
        ))
        speakers_detected.sort()
    
    return {
        "segments": result["segments"],
        "language": detected_language,
        "speakers": speakers_detected if speakers_detected else ["SPEAKER_00"]
    }


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
        "diarization_enabled": ENABLE_DIARIZATION and diarize_model is not None,
        "version": "3.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Endpoint de santé."""
    return {
        "status": "healthy",
        "model_loaded": whisperx_model is not None,
        "device": DEVICE,
        "gpu_available": torch.cuda.is_available(),
        "diarization_available": diarize_model is not None
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
        
        # ====== TRANSCRIPTION WHISPERX ======
        should_diarize = diarize and diarize_model is not None
        result = transcribe_with_whisperx(temp_path, diarize=should_diarize)
        
        end_time = time.time()
        processing_time = round(end_time - start_time, 2)
        
        # Construire le texte complet
        full_text = " ".join(seg.get("text", "") for seg in result["segments"])
        
        # Formater les segments pour la réponse
        formatted_segments = []
        for seg in result["segments"]:
            formatted_segments.append({
                "speaker": seg.get("speaker", "SPEAKER_00"),
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": seg.get("text", "").strip()
            })
        
        # Calcul des statistiques
        audio_duration = formatted_segments[-1]["end"] if formatted_segments else 0
        word_count = len(full_text.split())
        segment_count = len(formatted_segments)
        
        # Nettoyage du fichier temporaire
        os.unlink(temp_path)
        
        # Préparation de la réponse
        response_data = {
            "success": True,
            "filename": file.filename,
            "language": result["language"],
            "text": full_text,
            "speakers": result["speakers"],
            "segments": formatted_segments,
            "metadata": {
                "processing_time_seconds": processing_time,
                "audio_duration_seconds": round(audio_duration, 2),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "word_count": word_count,
                "segment_count": segment_count,
                "speaker_count": len(result["speakers"]),
                "model_used": WHISPER_MODEL,
                "diarization_enabled": should_diarize,
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
        import traceback
        traceback.print_exc()
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
        "whisperx": {
            "current_model": WHISPER_MODEL,
            "device": DEVICE,
            "compute_type": COMPUTE_TYPE,
            "gpu": GPU_NAME,
            "available_models": {
                "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x"},
                "base": {"params": "74M", "vram": "~1GB", "speed": "~16x"},
                "small": {"params": "244M", "vram": "~2GB", "speed": "~6x"},
                "medium": {"params": "769M", "vram": "~3GB", "speed": "~4x"},
                "large-v2": {"params": "1550M", "vram": "~5GB", "speed": "~2x"},
                "large-v3": {"params": "1550M", "vram": "~5GB", "speed": "~2x"}
            }
        },
        "diarization": {
            "enabled": ENABLE_DIARIZATION,
            "available": diarize_model is not None,
            "engine": "pyannote-audio" if diarize_model else None,
            "hf_token_set": HF_TOKEN is not None
        }
    }


# ====== DÉMARRAGE ======
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
