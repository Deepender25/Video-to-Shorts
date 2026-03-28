import os
import threading
from flask import Flask, render_template, request, jsonify, send_from_directory
from config import DOWNLOADS_DIR, OUTPUTS_DIR
from pipeline import create_job, run_download_phase, run_analysis_phase, get_job

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_video():
    """Start Phase 1: download video + parse transcript."""
    data = request.get_json()
    url = data.get("url", "").strip()

    if not url:
        return jsonify({"error": "Please provide a YouTube URL."}), 400

    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify({"error": "Please provide a valid YouTube URL."}), 400

    # Create job and run Phase 1 in background
    job_id = create_job(url)
    thread = threading.Thread(target=run_download_phase, args=(job_id,), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def job_status(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    return jsonify({
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "video_title": job.get("video_title", ""),
        "duration": job.get("duration", 0),
        "clips": [
            {
                "title": c.get("title", "Untitled"),
                "hook": c.get("hook", ""),
                "duration": c.get("duration", 0),
                "filename": c.get("filename", ""),
                "burned_filename": c.get("burned_filename", ""),
                "start": c.get("start", 0),
                "end": c.get("end", 0),
                "segments": c.get("segments", []),
                "segment_count": len(c.get("segments", [])),
            }
            for c in job.get("clips", [])
        ],
    })


@app.route("/api/preview/<job_id>")
def preview_video(job_id):
    """Serve the downloaded video file for the preview player."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    video_path = job.get("video_path")
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Video file not found."}), 404

    directory = os.path.dirname(video_path)
    filename = os.path.basename(video_path)
    return send_from_directory(directory, filename)


@app.route("/api/transcript/<job_id>")
def get_transcript(job_id):
    """Return the parsed transcript segments as JSON."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    segments = job.get("transcript_segments", [])
    return jsonify({
        "segments": [
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
            }
            for seg in segments
        ],
        "total": len(segments),
    })


@app.route("/api/continue/<job_id>", methods=["POST"])
def continue_processing(job_id):
    """User approved — start Phase 2 (AI analysis + cutting)."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404

    if job["status"] != "review":
        return jsonify({"error": "Job is not in review stage."}), 400

    # Mark as continuing and run Phase 2 in background
    job["status"] = "analyzing"
    job["progress"] = 50
    job["message"] = "Starting AI analysis..."

    thread = threading.Thread(target=run_analysis_phase, args=(job_id,), daemon=True)
    thread.start()

    return jsonify({"ok": True})


@app.route("/api/download/<job_id>/<filename>")
def download_clip(job_id, filename):
    # Added ?dl=1 support to allow inline playback for React video player
    as_attachment = request.args.get("dl") == "1"
    clip_dir = os.path.join(OUTPUTS_DIR, job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({"error": "Job not found."}), 404
    return send_from_directory(clip_dir, filename, as_attachment=as_attachment)


# ==============================================================================
# Editor & Subtitle Routes (Merged from Subtitle_gen)
# ==============================================================================
import video_editor

@app.route("/editor/<job_id>/<filename>")
def editor_view(job_id, filename):
    """Serve the React Vite editor."""
    return send_from_directory("static/react_editor", "index.html")

import transcriber

@app.route("/api/transcribe_clip/<job_id>/<filename>")
def transcribe_clip(job_id, filename):
    """Dynamically transcribes the cut clip for the editor UI using Whisper/Hinglish Apex."""
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found."}), 404
        
    clip_path = os.path.join(OUTPUTS_DIR, job_id, filename)
    if not os.path.exists(clip_path):
         return jsonify({"error": "Clip not found."}), 404
         
    try:
        # Heavily process the video dynamically on-demand
        result = transcriber.transcribe_video(clip_path, model_size="medium")
        
        # We can also update the job's memory to cache this if the user refreshes
        for c in job.get("clips", []):
            if c.get("filename") == filename:
                c["word_segments"] = result.get("segments", [])
                job["subtitle_lang"] = result.get("language", "en")
                break
                
        return jsonify({
             "segments": result.get("segments", []),
             "language": result.get("language", "en")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/burn", methods=["POST"])
def burn_video():
    data = request.json
    job_id = data.get("job_id")
    filename = data.get("filename")
    segments = data.get("segments")
    video_format = data.get("format", "mp4")
    
    if not job_id or not filename or not segments:
        return jsonify({"error": "Job ID, Filename, and segments required"}), 400

    input_path = os.path.join(OUTPUTS_DIR, job_id, filename)
    if not os.path.exists(input_path):
        return jsonify({"error": "Input file not found"}), 404
        
    # Generate unique output name
    base_name = os.path.splitext(filename)[0]
    import uuid
    unique_suffix = str(uuid.uuid4())[:8]
    output_filename = f"custom_sub_{base_name}_{unique_suffix}.{video_format}"
    output_path = os.path.join(OUTPUTS_DIR, job_id, output_filename)
    
    # We still need SRT for fallback if ASS is not supported or requested
    srt_filename = f"{base_name}.srt"
    srt_path = os.path.join(OUTPUTS_DIR, job_id, srt_filename)
    video_editor.generate_srt(segments, srt_path)
    
    style_config = data.get("styleConfig")
    success = video_editor.burn_subtitles(input_path, srt_path, output_path, style_config, segments)
    
    if success:
        return jsonify({
            "message": "Subtitles burned successfully",
            "download_url": f"/api/download/{job_id}/{output_filename}"
        })
    else:
        return jsonify({"error": "Failed to burn subtitles"}), 500

@app.route("/save_srt", methods=["POST"])
def save_srt():
    data = request.json
    job_id = data.get("job_id")
    filename = data.get("filename")
    segments = data.get("segments")
    
    if not job_id or not filename or not segments:
        return jsonify({"error": "Job ID, Filename, and segments required"}), 400

    base_name = os.path.splitext(filename)[0]
    srt_filename = f"{base_name}.srt"
    srt_path = os.path.join(OUTPUTS_DIR, job_id, srt_filename)
    
    try:
        video_editor.generate_srt(segments, srt_path)
        return jsonify({
            "message": "SRT generated successfully",
            "download_url": f"/api/download/{job_id}/{srt_filename}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/export_soft_subs", methods=["POST"])
def export_soft_subs():
    data = request.json
    job_id = data.get("job_id")
    filename = data.get("filename")
    segments = data.get("segments")
    
    if not job_id or not filename or not segments:
        return jsonify({"error": "Job ID, Filename, and segments required"}), 400

    input_path = os.path.join(OUTPUTS_DIR, job_id, filename)
    if not os.path.exists(input_path):
        return jsonify({"error": "Input file not found"}), 404

    base_name = os.path.splitext(filename)[0]
    srt_filename = f"{base_name}.srt"
    srt_path = os.path.join(OUTPUTS_DIR, job_id, srt_filename)
    
    video_editor.generate_srt(segments, srt_path)
    
    output_filename = f"softsubs_{base_name}.mkv"
    output_path = os.path.join(OUTPUTS_DIR, job_id, output_filename)
    
    success = video_editor.embed_soft_subtitles(input_path, srt_path, output_path)
    
    if success:
        return jsonify({
            "message": "Soft subtitles exported successfully",
            "download_url": f"/api/download/{job_id}/{output_filename}"
        })
    else:
        return jsonify({"error": "Failed to export soft subtitles"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
