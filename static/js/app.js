// ── DOM Elements ────────────────────────────────────
const form = document.getElementById('processForm');
const urlInput = document.getElementById('urlInput');
const submitBtn = document.getElementById('submitBtn');
const heroSection = document.getElementById('hero');
const progressSection = document.getElementById('progressSection');
const reviewSection = document.getElementById('reviewSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const progressBar = document.getElementById('progressBar');
const progressBadge = document.getElementById('progressBadge');
const progressMessage = document.getElementById('progressMessage');
const progressTitle = document.getElementById('progressTitle');
const clipsGrid = document.getElementById('clipsGrid');
const videoTitle = document.getElementById('videoTitle');
const errorMessage = document.getElementById('errorMessage');
const newVideoBtn = document.getElementById('newVideoBtn');
const retryBtn = document.getElementById('retryBtn');

// Review section elements
const reviewVideo = document.getElementById('reviewVideo');
const reviewTitle = document.getElementById('reviewTitle');
const reviewSubtitle = document.getElementById('reviewSubtitle');
const transcriptBody = document.getElementById('transcriptBody');
const transcriptCount = document.getElementById('transcriptCount');
const proceedBtn = document.getElementById('proceedBtn');
const cancelReviewBtn = document.getElementById('cancelReviewBtn');

// ── State ───────────────────────────────────────────
let currentJobId = null;
let pollInterval = null;

// ── Step tracking ───────────────────────────────────
const STEP_ORDER = ['downloading', 'parsing', 'review', 'analyzing', 'validating', 'cutting'];

function updateSteps(currentStatus) {
    const currentIdx = STEP_ORDER.indexOf(currentStatus);

    STEP_ORDER.forEach((step, idx) => {
        const el = document.getElementById(`step-${step}`);
        if (!el) return;

        el.classList.remove('active', 'completed');
        if (idx < currentIdx) {
            el.classList.add('completed');
        } else if (idx === currentIdx) {
            el.classList.add('active');
        }
    });
}

// ── Show/Hide Sections ──────────────────────────────
function showSection(section) {
    [heroSection, progressSection, reviewSection, resultsSection, errorSection].forEach(s => {
        s.classList.add('hidden');
    });
    section.classList.remove('hidden');
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function resetToHero() {
    showSection(heroSection);
    urlInput.value = '';
    submitBtn.disabled = false;
    submitBtn.querySelector('.btn-text').textContent = 'Generate Shorts';
    progressBar.style.width = '0%';
    clipsGrid.innerHTML = '';
    currentJobId = null;

    // Reset review section
    reviewVideo.removeAttribute('src');
    reviewVideo.load();
    transcriptBody.innerHTML = '<p class="transcript-loading">Loading transcript...</p>';
}

// ── Format Time ─────────────────────────────────────
function formatTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

// ── Form Submission ─────────────────────────────────
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = urlInput.value.trim();

    if (!url) {
        urlInput.focus();
        return;
    }

    // Disable button, show loading
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').textContent = 'Starting...';

    try {
        const res = await fetch('/api/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url }),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Failed to start processing.');
        }

        currentJobId = data.job_id;

        // Switch to progress view and start polling
        showSection(progressSection);
        pollStatus(data.job_id);

    } catch (err) {
        showError(err.message);
    }
});

// ── Poll Status ─────────────────────────────────────
function pollStatus(jobId) {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        try {
            const res = await fetch(`/api/status/${jobId}`);
            const data = await res.json();

            if (!res.ok) {
                clearInterval(pollInterval);
                showError(data.error || 'Job not found.');
                return;
            }

            // Update progress UI
            progressBar.style.width = `${data.progress}%`;
            progressBadge.textContent = `${data.progress}%`;
            progressMessage.textContent = data.message;
            progressTitle.textContent = data.video_title || 'Processing...';

            // Update step indicators
            updateSteps(data.status);

            // ── Phase 1 complete — show review ──
            if (data.status === 'review') {
                clearInterval(pollInterval);
                showReview(jobId, data);
            }
            // ── Phase 2 complete — show results ──
            else if (data.status === 'done') {
                clearInterval(pollInterval);
                showResults(data, jobId);
            }
            // ── Error ──
            else if (data.status === 'error') {
                clearInterval(pollInterval);
                showError(data.message);
            }

        } catch (err) {
            // Network error — keep polling
            console.error('Polling error:', err);
        }
    }, 1500);
}

// ── Show Review Section ─────────────────────────────
async function showReview(jobId, statusData) {
    // Set video source
    reviewVideo.src = `/api/preview/${jobId}`;
    reviewVideo.load();

    // Update title
    const title = statusData.video_title || 'Untitled Video';
    const duration = statusData.duration ? ` · ${formatTime(statusData.duration)}` : '';
    reviewTitle.textContent = title;
    reviewSubtitle.textContent = `Download complete! Review the video and transcript before AI analysis.${duration}`;

    // Fetch and display transcript
    try {
        const res = await fetch(`/api/transcript/${jobId}`);
        const data = await res.json();

        if (data.segments && data.segments.length > 0) {
            transcriptCount.textContent = `${data.total} segments`;
            transcriptBody.innerHTML = '';

            data.segments.forEach(seg => {
                const row = document.createElement('div');
                row.className = 'transcript-row';
                row.innerHTML = `
                    <span class="transcript-time">${formatTime(seg.start)}</span>
                    <span class="transcript-text">${escapeHtml(seg.text)}</span>
                `;
                // Click to seek video to that timestamp
                row.addEventListener('click', () => {
                    reviewVideo.currentTime = seg.start;
                    reviewVideo.play();
                });
                transcriptBody.appendChild(row);
            });
        } else {
            transcriptBody.innerHTML = '<p class="transcript-empty">No transcript segments found.</p>';
            transcriptCount.textContent = '0 segments';
        }
    } catch (err) {
        transcriptBody.innerHTML = '<p class="transcript-empty">Failed to load transcript.</p>';
        console.error('Transcript fetch error:', err);
    }

    showSection(reviewSection);
}

// ── Proceed to Phase 2 ─────────────────────────────
proceedBtn.addEventListener('click', async () => {
    if (!currentJobId) return;

    proceedBtn.disabled = true;
    proceedBtn.querySelector('.btn-text').textContent = 'Starting AI Analysis...';

    try {
        const res = await fetch(`/api/continue/${currentJobId}`, {
            method: 'POST',
        });
        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.error || 'Failed to continue.');
        }

        // Switch back to progress view for Phase 2
        showSection(progressSection);
        pollStatus(currentJobId);

    } catch (err) {
        showError(err.message);
    }
});

// ── Cancel Review ───────────────────────────────────
cancelReviewBtn.addEventListener('click', resetToHero);

// ── Show Results ────────────────────────────────────
function showResults(data, jobId) {
    videoTitle.textContent = data.video_title || '';
    clipsGrid.innerHTML = '';

    data.clips.forEach((clip, idx) => {
        const card = document.createElement('div');
        card.className = 'clip-card';
        card.innerHTML = `
            <div class="clip-info">
                <div class="clip-number">Clip ${idx + 1}</div>
                <div class="clip-title">${escapeHtml(clip.title)}</div>
                ${clip.hook ? `<div class="clip-hook">"${escapeHtml(clip.hook)}"</div>` : ''}
                <div class="clip-meta">
                    <span>
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="12" height="12">
                            <circle cx="12" cy="12" r="10"/>
                            <polyline points="12 6 12 12 16 14"/>
                        </svg>
                        ${clip.duration.toFixed(0)}s
                    </span>
                    <span>${formatTime(clip.start)} — ${formatTime(clip.end)}</span>
                </div>
            </div>
            <div class="clip-actions">
                <a href="/api/download/${jobId}/${clip.filename}" class="btn-download" download>
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="7 10 12 15 17 10"/>
                        <line x1="12" y1="15" x2="12" y2="3"/>
                    </svg>
                    Download
                </a>
            </div>
        `;
        clipsGrid.appendChild(card);
    });

    showSection(resultsSection);
}

// ── Show Error ──────────────────────────────────────
function showError(message) {
    errorMessage.textContent = message;
    showSection(errorSection);
}

// ── Escape HTML ─────────────────────────────────────
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ── Button Handlers ─────────────────────────────────
newVideoBtn.addEventListener('click', resetToHero);
retryBtn.addEventListener('click', resetToHero);
