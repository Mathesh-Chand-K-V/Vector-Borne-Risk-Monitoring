// Dataset Generator Web Application Logic
let ws = null;
let isGenerating = false;

const elements = {
  form: document.getElementById('config-form'),
  btnStart: document.getElementById('btn-start'),
  btnPause: document.getElementById('btn-pause'),
  btnResume: document.getElementById('btn-resume'),
  btnCancel: document.getElementById('btn-cancel'),
  statusBadge: document.getElementById('backend-status-badge'),
  statePill: document.getElementById('gen-state-pill'),
  progressFill: document.getElementById('progress-fill'),
  progressPct: document.getElementById('progress-percentage-text'),
  progressRatio: document.getElementById('progress-ratio-text'),
  statTarget: document.getElementById('stat-target'),
  statCompleted: document.getElementById('stat-completed'),
  statRejected: document.getElementById('stat-rejected'),
  statFailed: document.getElementById('stat-failed'),
  statRemaining: document.getElementById('stat-remaining'),
  activeSample: document.getElementById('active-sample-id'),
  activeStage: document.getElementById('active-stage'),
  galleryGrid: document.getElementById('gallery-grid'),
  galleryCount: document.getElementById('gallery-count'),
  logConsole: document.getElementById('log-console'),
  btnClearLogs: document.getElementById('clear-logs'),
};

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/progress`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    elements.statusBadge.innerHTML = '<span class="dot"></span> Connected';
    elements.statusBadge.classList.remove('badge-error');
  };

  ws.onclose = () => {
    elements.statusBadge.innerHTML = '<span class="dot" style="background:#ef4444;box-shadow:none;"></span> Disconnected';
    setTimeout(connectWebSocket, 3000);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === 'progress') {
        updateProgressUI(data.data);
      } else if (data.type === 'log') {
        appendLog(data.message, data.level || 'info');
      } else if (data.type === 'sample_added') {
        addSampleToGallery(data.sample);
      }
    } catch (err) {
      console.error('Error handling WS message:', err);
    }
  };
}

function updateProgressUI(data) {
  if (!data) return;
  const target = data.target || 0;
  const completed = data.completed || 0;
  const rejected = data.rejected || 0;
  const failed = data.failed || 0;
  const remaining = Math.max(0, target - completed);
  const pct = target > 0 ? ((completed / target) * 100).toFixed(1) : '0.0';

  elements.statTarget.textContent = target.toLocaleString();
  elements.statCompleted.textContent = completed.toLocaleString();
  elements.statRejected.textContent = rejected.toLocaleString();
  elements.statFailed.textContent = failed.toLocaleString();
  elements.statRemaining.textContent = remaining.toLocaleString();

  elements.progressPct.textContent = `${pct}%`;
  elements.progressRatio.textContent = `${completed} / ${target} valid samples`;
  elements.progressFill.style.width = `${Math.min(100, pct)}%`;

  if (data.current_sample) {
    elements.activeSample.textContent = data.current_sample;
  }
  if (data.current_stage) {
    elements.activeStage.textContent = data.current_stage;
  }

  if (data.is_paused) {
    elements.statePill.textContent = 'Paused';
    elements.statePill.className = 'pill pill-paused';
    elements.btnPause.style.display = 'none';
    elements.btnResume.style.display = 'inline-flex';
    elements.btnResume.disabled = false;
  } else if (completed >= target && target > 0) {
    elements.statePill.textContent = 'Completed';
    elements.statePill.className = 'pill pill-idle';
    setControlsRunning(false);
  } else if (isGenerating) {
    elements.statePill.textContent = 'Running';
    elements.statePill.className = 'pill pill-running';
    elements.btnPause.style.display = 'inline-flex';
    elements.btnPause.disabled = false;
    elements.btnResume.style.display = 'none';
  }
}

function appendLog(msg, level = 'info') {
  const line = document.createElement('div');
  line.className = `log-line ${level}`;
  const timeStr = new Date().toLocaleTimeString();
  line.textContent = `[${timeStr}] ${msg}`;
  elements.logConsole.appendChild(line);
  elements.logConsole.scrollTop = elements.logConsole.scrollHeight;
}

function addSampleToGallery(sample) {
  // Remove empty state if present
  const empty = elements.galleryGrid.querySelector('.empty-state');
  if (empty) empty.remove();

  const card = document.createElement('div');
  card.className = 'sample-thumb-card';
  card.innerHTML = `
    <img src="/api/image/${sample.sample_id}?dir=${encodeURIComponent(sample.output_dir || 'dataset')}" alt="${sample.sample_id}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\'http://www.w3.org/2000/svg\' width=\'100\' height=\'100\'><rect width=\'100%\' height=\'100%\' fill=\'%231f2937\'/><text x=\'50%\' y=\'50%\' fill=\'%239ca3af\' font-size=\'10\' text-anchor=\'middle\' dy=\'.3em\'>Patch</text></svg>'">
    <div class="thumb-info">
      <div class="thumb-id">${sample.sample_id}</div>
      <div class="thumb-meta">${sample.date || ''} • ${sample.cloud_coverage || 0}% cloud</div>
      <div class="thumb-meta">${sample.temp || ''}°C</div>
    </div>
  `;
  elements.galleryGrid.prepend(card);
  const total = elements.galleryGrid.querySelectorAll('.sample-thumb-card').length;
  elements.galleryCount.textContent = `${total} items`;
}

function setControlsRunning(running) {
  isGenerating = running;
  elements.btnStart.disabled = running;
  elements.btnPause.disabled = !running;
  elements.btnResume.disabled = true;
  elements.btnResume.style.display = 'none';
  elements.btnPause.style.display = 'inline-flex';
  elements.btnCancel.disabled = !running;

  if (running) {
    elements.statePill.textContent = 'Running';
    elements.statePill.className = 'pill pill-running';
  } else {
    elements.statePill.textContent = 'Idle';
    elements.statePill.className = 'pill pill-idle';
  }
}

// Event Listeners
elements.form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(elements.form);
  const payload = {
    location: formData.get('location'),
    radius_km: parseFloat(formData.get('radius_km')),
    target_samples: parseInt(formData.get('target_samples'), 10),
    start_date: formData.get('start_date'),
    end_date: formData.get('end_date'),
    max_cloud_coverage: parseFloat(formData.get('max_cloud_coverage')),
    max_date_difference_days: parseInt(formData.get('max_date_difference_days'), 10),
    image_size: parseInt(formData.get('image_size'), 10),
    workers: parseInt(formData.get('workers'), 10),
    random_seed: parseInt(formData.get('random_seed'), 10),
    output_dir: formData.get('output_dir')
  };

  try {
    setControlsRunning(true);
    appendLog(`Starting generation task for ${payload.location} (${payload.target_samples} samples)...`, 'info');
    const resp = await fetch('/api/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const res = await resp.json();
    if (!resp.ok) {
      appendLog(`Error starting generation: ${res.detail || 'Unknown error'}`, 'error');
      setControlsRunning(false);
    } else {
      appendLog('Generation task initialized on server.', 'success');
    }
  } catch (err) {
    appendLog(`Network error: ${err.message}`, 'error');
    setControlsRunning(false);
  }
});

elements.btnPause.addEventListener('click', async () => {
  try {
    await fetch('/api/pause', { method: 'POST' });
    appendLog('Requested pause...', 'warn');
  } catch (err) {
    appendLog(`Failed to pause: ${err.message}`, 'error');
  }
});

elements.btnResume.addEventListener('click', async () => {
  try {
    await fetch('/api/resume', { method: 'POST' });
    appendLog('Requested resume...', 'success');
  } catch (err) {
    appendLog(`Failed to resume: ${err.message}`, 'error');
  }
});

elements.btnCancel.addEventListener('click', async () => {
  try {
    await fetch('/api/cancel', { method: 'POST' });
    appendLog('Requested cancellation...', 'warn');
    setControlsRunning(false);
  } catch (err) {
    appendLog(`Failed to cancel: ${err.message}`, 'error');
  }
});

elements.btnClearLogs.addEventListener('click', () => {
  elements.logConsole.innerHTML = '';
});

// Initialization
connectWebSocket();
