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
    clip_dir = os.path.join(OUTPUTS_DIR, job_id)
    if not os.path.isdir(clip_dir):
        return jsonify({"error": "Job not found."}), 404
    return send_from_directory(clip_dir, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
