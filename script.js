/* ═══════════════════════════════════════════════════════════
   FLUIDSENSE — Gesture Control Dashboard
   Customer-Focused Interface
   ═══════════════════════════════════════════════════════════ */

// ── Application State ──
const appState = {
    gestures: [
        { id: 'palm', name: 'Palm', icon: 'fa-hand' },
        { id: 'fist', name: 'Fist', icon: 'fa-hand-fist' },
        { id: 'peace', name: 'Peace', icon: 'fa-hand-peace' },
        { id: 'thumb', name: 'Thumb', icon: 'fa-thumbs-up' },
        { id: 'index', name: 'Index Finger', icon: 'fa-hand-pointer' },
        { id: 'ok', name: 'OK', icon: 'fa-thumbs-up' },
        { id: 'l', name: 'L', icon: 'fa-l' },
        { id: 'c', name: 'C', icon: 'fa-hand-lizard' },
        { id: 'down', name: 'Down', icon: 'fa-hand-point-down' },
    ],
    actions: [
        { id: 'a1', name: 'Switch Tab Right', icon: 'fa-arrow-right' },
        { id: 'a2', name: 'Switch Tab Left', icon: 'fa-arrow-left' },
        { id: 'a3', name: 'Open New Tab', icon: 'fa-plus' },
        { id: 'a4', name: 'Close Tab', icon: 'fa-xmark' },
        { id: 'a5', name: 'Play / Pause Video', icon: 'fa-play' },
        { id: 'a6', name: 'Mute / Unmute', icon: 'fa-volume-xmark' },
        { id: 'a7', name: 'Open Calculator', icon: 'fa-calculator' },
        { id: 'a8', name: 'Take Screenshot', icon: 'fa-camera' },
        { id: 'a9', name: 'Volume Up', icon: 'fa-volume-high' },
        { id: 'a10', name: 'Volume Down', icon: 'fa-volume-low' },
        { id: 'a11', name: 'Refresh Page', icon: 'fa-rotate-right' },
        { id: 'a12', name: 'Energy Saving Mode', icon: 'fa-leaf' },
        { id: 'a13', name: 'Shut Down PC', icon: 'fa-power-off' },
        { id: 'a14', name: 'Lock Screen', icon: 'fa-lock' },
    ],
    mappings: [],          // { id, gestureId, actionId }
    recognition: false,
    selectedMapGesture: null,
    selectedMapAction: null,
    deletingMappingId: null,
    activities: [],
};

// ── Init ──
document.addEventListener('DOMContentLoaded', async () => {
    loadState();
    await syncGesturesFromBackend();
    await syncMappingsFromBackend();
    renderAll();
    updateStats();
    updateQuickStart();
    startSSEListener();
    pollBackendStatus();

    // Click outside mapping lists → deselect
    document.addEventListener('click', (e) => {
        const mappingSection = document.getElementById('section-mapping');
        if (!mappingSection) return;
        const insideGestures = e.target.closest('#mapping-gesture-list');
        const insideActions = e.target.closest('#mapping-action-list');
        const insideCenter = e.target.closest('.mapping-center');
        if (mappingSection.contains(e.target) && !insideGestures && !insideActions && !insideCenter) {
            document.querySelectorAll('#mapping-gesture-list .mapping-card, #mapping-action-list .mapping-card').forEach(c => c.classList.remove('selected'));
            resetMappingSelection();
        }
    });

    // ESC key to close recording modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const recordingModal = document.getElementById('modal-recording');
            if (recordingModal && recordingModal.classList.contains('show')) {
                cancelRecording();
            }
        }
    });

    // keyboard shortcut for recognition toggle
    document.addEventListener('keydown', (e) => {
        if (e.key === ' ' && e.ctrlKey) { e.preventDefault(); toggleRecognition(); }
    });
});

// ── Built-in gesture IDs (never removed) ──
const BUILTIN_IDS = new Set(appState.gestures.map(g => g.id));

// ── Persistence ──
function saveState() {
    try {
        const customGestures = appState.gestures.filter(g => !BUILTIN_IDS.has(g.id));
        localStorage.setItem('ctrlState', JSON.stringify({
            customGestures,
            activities: appState.activities.slice(0, 50),
        }));
    } catch (e) { }
}

function loadState() {
    try {
        const saved = JSON.parse(localStorage.getItem('ctrlState'));
        if (!saved) return;
        // Merge custom gestures after the built-ins
        if (saved.customGestures?.length) {
            saved.customGestures.forEach(g => {
                if (!appState.gestures.some(x => x.id === g.id)) appState.gestures.push(g);
            });
        }
        // NOTE: mappings are NOT loaded from localStorage — backend is the only source
        if (saved.activities) appState.activities = saved.activities;
    } catch (e) { }
}

// ═══════════════════════════════════════
//  NAVIGATION
// ═══════════════════════════════════════

const sectionTitles = {
    dashboard: { title: 'Dashboard', sub: 'Your gesture control at a glance' },
    mapping: { title: 'Action Mapping', sub: 'Link gestures to desktop actions' },
};

function switchSection(sectionId, navEl) {
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    if (navEl) navEl.classList.add('active');

    document.querySelectorAll('.content-section').forEach(s => s.classList.remove('active'));
    const target = document.getElementById(`section-${sectionId}`);
    if (target) {
        target.classList.add('active');
        target.style.animation = 'none';
        target.offsetHeight;
        target.style.animation = '';
    }

    const info = sectionTitles[sectionId];
    if (info) {
        document.getElementById('page-title-text').textContent = info.title;
        document.getElementById('page-subtitle').textContent = info.sub;
    }

    if (sectionId === 'mapping') renderMappingSection();
    if (sectionId === 'dashboard') { renderDashboardMappings(); updateStats(); }

    document.getElementById('sidebar').classList.remove('open');
}

function toggleSidebar() {
    document.getElementById('sidebar').classList.toggle('open');
}

// ═══════════════════════════════════════
//  RENDER ALL
// ═══════════════════════════════════════

function renderAll() {
    renderMappingSection();
    renderDashboardMappings();
    renderActivityFeed();
}

// ═══════════════════════════════════════
//  STATS
// ═══════════════════════════════════════

function updateStats() {
    document.getElementById('stat-gestures-val').textContent = appState.gestures.length;
    document.getElementById('stat-mappings-val').textContent = appState.mappings.length;
    document.getElementById('stat-actions-val').textContent = appState.actions.length;
    document.getElementById('stat-status-val').textContent = appState.recognition ? 'Active' : 'Off';
}

function updateQuickStart() {
    const s1 = document.getElementById('step-1');
    const s2 = document.getElementById('step-2');
    const s3 = document.getElementById('step-3');

    if (appState.mappings.length > 0) s1.classList.add('done'); else s1.classList.remove('done');
    if (appState.recognition) s2.classList.add('done'); else s2.classList.remove('done');
    // Step 3 is done when recognition is on and mappings exist
    if (appState.recognition && appState.mappings.length > 0) s3.classList.add('done'); else s3.classList.remove('done');
}

// ═══════════════════════════════════════
//  ADD CUSTOM GESTURE
// ═══════════════════════════════════════

let selectedNewIcon = 'none';

function openAddGestureModal() {
    document.getElementById('gesture-name-input').value = '';
    document.getElementById('save-gesture-btn').disabled = true;
    selectedNewIcon = 'none';

    // Reset icon picker
    document.querySelectorAll('#icon-picker .icon-option').forEach(o => o.classList.remove('selected'));
    const noneOpt = document.querySelector('#icon-picker .icon-option[data-icon="none"]');
    if (noneOpt) noneOpt.classList.add('selected');

    openModal('modal-add-gesture');

    document.getElementById('gesture-name-input').oninput = function () {
        document.getElementById('save-gesture-btn').disabled = !this.value.trim();
    };
}

function selectIcon(el) {
    document.querySelectorAll('#icon-picker .icon-option').forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
    selectedNewIcon = el.getAttribute('data-icon');
}

async function saveNewGesture() {
    const name = document.getElementById('gesture-name-input').value.trim();
    if (!name) return;

    if (appState.gestures.some(g => g.name.toLowerCase() === name.toLowerCase())) {
        showToast('A gesture with this name already exists', 'error');
        return;
    }

    // Add to local state immediately for UI responsiveness
    appState.gestures.push({
        id: 'custom_' + Date.now(),
        name: name,
        icon: selectedNewIcon === 'none' ? 'fa-circle' : selectedNewIcon,
    });

    addActivity('add', `Added custom gesture "${name}"`);
    saveState();
    renderAll();
    updateStats();
    closeModal('modal-add-gesture');

    // Call backend to record gesture via webcam and retrain
    const result = await api('/gesture/add', 'POST', { name: name });
    if (result && result.status === 'recording_started') {
        openRecordingModal(name);
    }
}

// ═══════════════════════════════════════
//  RECORDING CAMERA MODAL
// ═══════════════════════════════════════

let recordingPollTimer = null;

function openRecordingModal(gestureName) {
    document.getElementById('recording-gesture-name').textContent = gestureName;
    document.getElementById('recording-status-text').textContent = 'Starting camera...';
    document.getElementById('recording-count').textContent = '0 / 400 samples';
    document.getElementById('recording-progress-bar').style.width = '0%';
    document.getElementById('camera-feed-overlay').style.display = 'flex';

    // Start MJPEG stream
    const feed = document.getElementById('recording-camera-feed');
    feed.src = `${API}/gesture/add/stream`;
    feed.onerror = () => { /* stream ended or not started yet */ };
    feed.onload = () => {
        document.getElementById('camera-feed-overlay').style.display = 'none';
    };

    openModal('modal-recording');

    // Poll progress
    recordingPollTimer = setInterval(async () => {
        const result = await api('/status');
        if (!result) return;

        const task = result.backgroundTask;
        if (task && task.type === 'add') {
            const progress = task.progress || '';
            document.getElementById('recording-status-text').textContent = progress;

            // Parse "Captured X/Y" to update progress bar
            const match = progress.match(/Captured (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1]);
                const total = parseInt(match[2]);
                const pct = Math.round((current / total) * 100);
                document.getElementById('recording-progress-bar').style.width = pct + '%';
                document.getElementById('recording-count').textContent = `${current} / ${total} samples`;

                // Hide overlay once frames start arriving
                if (current > 0) {
                    document.getElementById('camera-feed-overlay').style.display = 'none';
                }
            }

            if (!task.running) {
                // Recording finished (success or cancel or error)
                clearInterval(recordingPollTimer);
                recordingPollTimer = null;
                feed.src = '';

                // Reset cancel button for next time
                const cancelBtn = document.getElementById('cancel-recording-btn');
                if (cancelBtn) {
                    cancelBtn.disabled = false;
                    cancelBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Cancel Recording';
                }

                closeModal('modal-recording');

                if (task.error) {
                    showToast(task.progress, task.error === 'Cancelled by user' ? 'info' : 'error');
                } else {
                    showToast(task.progress, 'success');
                }
                // Refresh gesture list
                syncGesturesFromBackend();
            }
        }
    }, 500);
}

async function cancelRecording() {
    // Disable button to prevent double-clicks
    const btn = document.getElementById('cancel-recording-btn');
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Cancelling...';
    }
    await api('/gesture/add/cancel', 'POST');
    showToast('Cancelling — any recorded data will be erased', 'info');
    // The poll loop will handle closing the modal and re-enabling the button
}

// ═══════════════════════════════════════
//  MAPPING SECTION
// ═══════════════════════════════════════

function renderMappingSection() {
    renderMappingGestures();
    renderMappingActions();
    renderMappingChips();
    resetMappingSelection();
}

function renderMappingGestures() {
    const list = document.getElementById('mapping-gesture-list');
    list.innerHTML = '';
    let available = 0;

    const builtIn = appState.gestures.filter(g => BUILTIN_IDS.has(g.id));
    const custom = appState.gestures.filter(g => !BUILTIN_IDS.has(g.id));

    function renderGestureCard(g, isCustom) {
        const isMapped = appState.mappings.some(m => m.gestureId === g.id);
        const card = document.createElement('div');
        card.className = `mapping-card${isMapped ? ' used' : ''}`;
        card.setAttribute('data-gesture-id', g.id);
        if (!isMapped) {
            card.onclick = (e) => {
                if (e.target.closest('.delete-gesture-btn')) return;
                selectMappingGesture(g.id, card);
            };
            available++;
        }
        const deleteBtn = isCustom
            ? `<button class="delete-gesture-btn" onclick="event.stopPropagation(); deleteCustomGesture('${g.id}')" title="Remove gesture"><i class="fa-solid fa-xmark"></i></button>`
            : '';
        card.innerHTML = `<i class="fa-solid ${g.icon}"></i> <span>${esc(g.name)}</span>${deleteBtn}`;
        return card;
    }

    // Built-in gestures
    builtIn.forEach(g => list.appendChild(renderGestureCard(g, false)));

    // Custom gestures section
    if (custom.length > 0) {
        const divider = document.createElement('div');
        divider.className = 'gesture-section-divider';
        divider.innerHTML = '<span>Custom Gestures</span>';
        list.appendChild(divider);
        custom.forEach(g => list.appendChild(renderGestureCard(g, true)));
    }

    document.getElementById('available-gestures-count').textContent = available;
}

async function deleteCustomGesture(id) {
    const g = appState.gestures.find(x => x.id === id);
    if (!g || BUILTIN_IDS.has(id)) return;

    // Remove any mapping that uses this gesture
    const mapping = appState.mappings.find(m => m.gestureId === id);
    if (mapping) appState.mappings = appState.mappings.filter(m => m.gestureId !== id);

    appState.gestures = appState.gestures.filter(x => x.id !== id);
    addActivity('delete', `Removed custom gesture "${g.name}"`);
    saveState();
    renderAll();
    updateStats();
    updateQuickStart();
    showToast(`"${g.name}" removed`, 'info');

    // Call backend to remove from CSV and retrain
    api('/gesture/delete', 'POST', { name: g.name });
}

function renderMappingActions() {
    const list = document.getElementById('mapping-action-list');
    list.innerHTML = '';
    let available = 0;

    appState.actions.forEach(a => {
        const isMapped = appState.mappings.some(m => m.actionId === a.id);
        const card = document.createElement('div');
        card.className = `mapping-card${isMapped ? ' used' : ''}`;
        card.setAttribute('data-action-id', a.id);
        if (!isMapped) {
            card.onclick = () => selectMappingAction(a.id, card);
            available++;
        }
        card.innerHTML = `<i class="fa-solid ${a.icon}"></i> <span>${esc(a.name)}</span>`;
        list.appendChild(card);
    });

    document.getElementById('available-actions-count').textContent = available;
}

function selectMappingGesture(id, el) {
    document.querySelectorAll('#mapping-gesture-list .mapping-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    appState.selectedMapGesture = id;

    const g = appState.gestures.find(x => x.id === id);
    const label = document.getElementById('selected-gesture-label');
    label.textContent = g ? g.name : 'Pick a gesture';
    label.classList.toggle('filled', !!g);
    checkLinkReady();
}

function selectMappingAction(id, el) {
    document.querySelectorAll('#mapping-action-list .mapping-card').forEach(c => c.classList.remove('selected'));
    el.classList.add('selected');
    appState.selectedMapAction = id;

    const a = appState.actions.find(x => x.id === id);
    const label = document.getElementById('selected-action-label');
    label.textContent = a ? a.name : 'Pick an action';
    label.classList.toggle('filled', !!a);
    checkLinkReady();
}

function checkLinkReady() {
    const btn = document.getElementById('link-btn');
    const text = document.getElementById('link-btn-text');
    const ready = appState.selectedMapGesture && appState.selectedMapAction;
    btn.disabled = !ready;
    btn.classList.toggle('ready', ready);
    text.textContent = ready ? 'Link Now' : 'Select a Pair';
}

async function createMapping() {
    if (!appState.selectedMapGesture || !appState.selectedMapAction) return;

    // One-to-one enforcement
    if (appState.mappings.some(m => m.gestureId === appState.selectedMapGesture)) {
        showToast('This gesture is already mapped', 'warning');
        return;
    }
    if (appState.mappings.some(m => m.actionId === appState.selectedMapAction)) {
        showToast('This action already has a gesture assigned', 'warning');
        return;
    }

    const gName = appState.gestures.find(g => g.id === appState.selectedMapGesture)?.name;
    const aName = appState.actions.find(a => a.id === appState.selectedMapAction)?.name;

    const mapping = {
        id: 'm' + Date.now(),
        gestureId: appState.selectedMapGesture,
        actionId: appState.selectedMapAction,
    };
    appState.mappings.push(mapping);

    addActivity('link', `Mapped "${gName}" → "${aName}"`);
    saveState();
    renderAll();
    updateStats();
    updateQuickStart();
    showToast(`"${gName}" → "${aName}" linked!`, 'success');

    // Sync entire mappings array to backend
    syncMappingsToBackend();
}

function openDeleteMappingModal(mappingId) {
    const m = appState.mappings.find(x => x.id === mappingId);
    if (!m) return;
    appState.deletingMappingId = mappingId;
    const gName = appState.gestures.find(g => g.id === m.gestureId)?.name || '';
    const aName = appState.actions.find(a => a.id === m.actionId)?.name || '';
    document.getElementById('delete-mapping-name').textContent = `${gName} → ${aName}`;
    openModal('modal-confirm-delete');
}

function confirmDeleteMapping() {
    const m = appState.mappings.find(x => x.id === appState.deletingMappingId);
    if (!m) return;
    const gName = appState.gestures.find(g => g.id === m.gestureId)?.name;
    const aName = appState.actions.find(a => a.id === m.actionId)?.name;
    const deletedId = appState.deletingMappingId;

    appState.mappings = appState.mappings.filter(x => x.id !== deletedId);

    addActivity('delete', `Removed "${gName}" → "${aName}"`);
    saveState();
    renderAll();
    updateStats();
    updateQuickStart();
    closeModal('modal-confirm-delete');
    showToast('Mapping removed', 'info');

    // Sync entire mappings array to backend
    syncMappingsToBackend();
}

function removeMapping(mappingId) {
    openDeleteMappingModal(mappingId);
}

function resetMappingSelection() {
    appState.selectedMapGesture = null;
    appState.selectedMapAction = null;
    document.getElementById('selected-gesture-label').textContent = 'Pick a gesture';
    document.getElementById('selected-gesture-label').classList.remove('filled');
    document.getElementById('selected-action-label').textContent = 'Pick an action';
    document.getElementById('selected-action-label').classList.remove('filled');
    checkLinkReady();
}

function renderMappingChips() {
    const container = document.getElementById('mapping-chips');
    const empty = document.getElementById('mapping-chips-empty');
    container.innerHTML = '';

    if (appState.mappings.length === 0) { empty.style.display = ''; return; }
    empty.style.display = 'none';

    appState.mappings.forEach(m => {
        const g = appState.gestures.find(x => x.id === m.gestureId);
        const a = appState.actions.find(x => x.id === m.actionId);
        if (!g || !a) return;

        const chip = document.createElement('div');
        chip.className = 'chip';
        chip.innerHTML = `
            <i class="fa-solid ${g.icon}"></i>
            <span>${esc(g.name)}</span>
            <i class="fa-solid fa-arrow-right-long"></i>
            <span>${esc(a.name)}</span>
            <button class="close-chip" onclick="removeMapping('${m.id}')" title="Remove">✕</button>
        `;
        container.appendChild(chip);
    });
}

// ═══════════════════════════════════════
//  DASHBOARD MAPPINGS TABLE
// ═══════════════════════════════════════

function renderDashboardMappings() {
    const tbody = document.getElementById('dashboard-mappings-body');
    const empty = document.getElementById('mappings-empty');
    const table = document.getElementById('dashboard-mappings-table');
    tbody.innerHTML = '';

    if (appState.mappings.length === 0) {
        empty.style.display = '';
        table.style.display = 'none';
        return;
    }
    empty.style.display = 'none';
    table.style.display = '';

    appState.mappings.forEach(m => {
        const g = appState.gestures.find(x => x.id === m.gestureId);
        const a = appState.actions.find(x => x.id === m.actionId);
        if (!g || !a) return;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><div class="mapping-gesture-cell"><i class="fa-solid ${g.icon}"></i> ${esc(g.name)}</div></td>
            <td><i class="fa-solid fa-arrow-right" style="color:var(--text-tertiary);font-size:0.75rem"></i></td>
            <td><div class="mapping-action-cell"><i class="fa-solid ${a.icon}"></i> ${esc(a.name)}</div></td>
            <td><button class="delete-mapping-btn" onclick="removeMapping('${m.id}')"><i class="fa-solid fa-trash"></i></button></td>
        `;
        tbody.appendChild(row);
    });
}

// ═══════════════════════════════════════
//  RECOGNITION TOGGLE
// ═══════════════════════════════════════

async function toggleRecognition(on) {
    appState.recognition = on;
    const dot = document.getElementById('status-dot');
    const val = document.getElementById('status-value');

    if (on) {
        const result = await api('/recognize/start', 'POST');
        if (result && result.error) {
            showToast(result.error, 'error');
            appState.recognition = false;
            document.getElementById('recognition-toggle').checked = false;
            return;
        }
        dot.className = 'status-dot online';
        val.textContent = 'Active';
        addActivity('link', 'Recognition started');
        showToast('Gesture recognition is active', 'success');
    } else {
        await api('/recognize/stop', 'POST');
        dot.className = 'status-dot offline';
        val.textContent = 'Off';
        addActivity('delete', 'Recognition stopped');
        showToast('Gesture recognition stopped', 'info');
    }
    updateStats();
    updateQuickStart();
}

// ═══════════════════════════════════════
//  ACTIVITY FEED
// ═══════════════════════════════════════

function addActivity(type, message) {
    appState.activities.unshift({ type, message, time: Date.now() });
    if (appState.activities.length > 50) appState.activities.pop();
    saveState();
    renderActivityFeed();
}

function renderActivityFeed() {
    const list = document.getElementById('activity-list');
    const empty = document.getElementById('activity-empty');
    list.innerHTML = '';

    if (appState.activities.length === 0) {
        list.appendChild(empty);
        return;
    }

    const icons = {
        add: { icon: 'fa-plus', cls: 'add' },
        delete: { icon: 'fa-minus', cls: 'delete' },
        link: { icon: 'fa-link', cls: 'link' },
    };

    appState.activities.slice(0, 12).forEach(a => {
        const info = icons[a.type] || icons.add;
        const item = document.createElement('div');
        item.className = 'activity-item';
        item.innerHTML = `
            <div class="activity-icon ${info.cls}"><i class="fa-solid ${info.icon}"></i></div>
            <span class="activity-text">${esc(a.message)}</span>
            <span class="activity-time">${timeAgo(a.time)}</span>
        `;
        list.appendChild(item);
    });
}

function clearActivity() {
    appState.activities = [];
    saveState();
    renderActivityFeed();
    showToast('Activity cleared', 'info');
}

// ═══════════════════════════════════════
//  MODALS
// ═══════════════════════════════════════

function openModal(id) {
    document.getElementById(id).classList.add('show');
}

function closeModal(id) {
    document.getElementById(id).classList.remove('show');
}

document.addEventListener('click', e => {
    if (e.target.classList.contains('modal-overlay')) e.target.classList.remove('show');
});
document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.querySelectorAll('.modal-overlay.show').forEach(m => m.classList.remove('show'));
});

// ═══════════════════════════════════════
//  TOASTS
// ═══════════════════════════════════════

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: 'fa-circle-check', error: 'fa-circle-xmark', warning: 'fa-triangle-exclamation', info: 'fa-circle-info' };
    toast.innerHTML = `<i class="fa-solid ${icons[type] || icons.info}"></i> <span>${esc(message)}</span>`;
    container.appendChild(toast);
    setTimeout(() => { toast.classList.add('removing'); setTimeout(() => toast.remove(), 300); }, 3500);
}

// ═══════════════════════════════════════
//  UTILS
// ═══════════════════════════════════════

function esc(text) {
    const d = document.createElement('div');
    d.textContent = text || '';
    return d.innerHTML;
}

function timeAgo(ts) {
    const sec = Math.floor((Date.now() - ts) / 1000);
    if (sec < 10) return 'Just now';
    if (sec < 60) return sec + 's ago';
    const min = Math.floor(sec / 60);
    if (min < 60) return min + 'm ago';
    const hr = Math.floor(min / 60);
    if (hr < 24) return hr + 'h ago';
    return Math.floor(hr / 24) + 'd ago';
}

// ═══════════════════════════════════════
//  PYTHON BACKEND API
// ═══════════════════════════════════════

const API = 'http://127.0.0.1:5001/api';

async function api(endpoint, method = 'GET', body = null) {
    try {
        const opts = { method, headers: { 'Content-Type': 'application/json' } };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(`${API}${endpoint}`, opts);
        if (!res.ok) throw new Error(res.status);
        return await res.json();
    } catch (e) {
        console.warn(`API ${endpoint}:`, e.message);
        return null;
    }
}

// ═══════════════════════════════════════
//  SYNC GESTURES FROM BACKEND
// ═══════════════════════════════════════

async function syncGesturesFromBackend() {
    const result = await api('/gestures');
    if (!result || !result.gestures) {
        console.log('Backend not available — running offline');
        return;
    }

    const backendClasses = result.gestures.map(g => g.toLowerCase());
    console.log('Backend trained classes:', backendClasses);

    // Remove custom gestures that no longer exist in backend training data
    const before = appState.gestures.length;
    appState.gestures = appState.gestures.filter(g => {
        if (BUILTIN_IDS.has(g.id)) return true;           // keep built-in always
        const nameMatches = backendClasses.includes(g.name.toLowerCase());
        const idMatches = backendClasses.includes(g.id.toLowerCase());
        if (!nameMatches && !idMatches) {
            console.log(`Removing stale custom gesture: "${g.name}" (not in backend)`);
            return false;
        }
        return true;
    });

    // Clean up orphaned mappings (pointing to deleted gestures)
    const validGestureIds = new Set(appState.gestures.map(g => g.id));
    appState.mappings = appState.mappings.filter(m => validGestureIds.has(m.gestureId));

    if (appState.gestures.length < before) {
        saveState();
        renderAll();
        showToast('Synced gestures with backend', 'info');
    } else {
        showToast('Connected to backend', 'success');
    }
}

// ═══════════════════════════════════════
//  SYNC MAPPINGS WITH BACKEND
// ═══════════════════════════════════════

async function syncMappingsFromBackend() {
    const result = await api('/mappings');
    if (!result || !result.mappings) return;   // backend offline — keep localStorage mappings
    appState.mappings = result.mappings;
    saveState();
    console.log('Synced mappings from backend:', appState.mappings.length);
}

function syncMappingsToBackend() {
    // Push entire mappings array to backend — overwrites mappings.json
    api('/mappings', 'PUT', { mappings: appState.mappings });
    console.log('Pushed mappings to backend:', appState.mappings.length);
}

// ═══════════════════════════════════════
//  SSE — REAL-TIME GESTURE DETECTION
// ═══════════════════════════════════════

let sseSource = null;

function startSSEListener() {
    try {
        sseSource = new EventSource(`${API}/recognize/stream`);
        sseSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                handleDetectedGesture(data);
            } catch (e) { /* ignore parse errors */ }
        };
        sseSource.onerror = () => {
            // Backend might not be running; silently retry
            console.log('SSE connection lost, will auto-retry');
        };
    } catch (e) {
        console.log('SSE not available');
    }
}

let _lastToastGesture = '';
let _lastToastTime = 0;

function handleDetectedGesture(data) {
    // data = { gesture: "Palm", confidence: 95, timestamp: ..., action: "Volume Up" | null }
    if (!appState.recognition) return;

    // Show toast popup for detected gestures (max 1 per 10 seconds)
    const now = Date.now();
    if (now - _lastToastTime > 6000) {
        if (data.action) {
            showToast(`🖐 "${data.gesture}" → ${data.action}`, 'success');
        } else {
            showToast(`🖐 "${data.gesture}" detected (${data.confidence}%)`, 'info');
        }
        _lastToastGesture = data.gesture;
        _lastToastTime = now;
    }

    // Find if this gesture is mapped to an action
    const gestureEntry = appState.gestures.find(
        g => g.name.toLowerCase() === data.gesture.toLowerCase() ||
            g.id.toLowerCase() === data.gesture.toLowerCase()
    );
    if (!gestureEntry) return;

    const mapping = appState.mappings.find(m => m.gestureId === gestureEntry.id);
    if (mapping) {
        const action = appState.actions.find(a => a.id === mapping.actionId);
        if (action) {
            addActivity('link', `Detected "${data.gesture}" (${data.confidence}%) → ${action.name}`);
        }
    } else {
        addActivity('link', `Detected "${data.gesture}" (${data.confidence}%)`);
    }
}

// ═══════════════════════════════════════
//  POLL BACKGROUND TASK STATUS
// ═══════════════════════════════════════

let _lastProgress = '';

function pollBackendStatus() {
    setInterval(async () => {
        const result = await api('/status');
        if (!result) return;

        // Update background task progress
        if (result.backgroundTask && result.backgroundTask.running) {
            const prog = result.backgroundTask.progress;
            if (prog !== _lastProgress) {
                _lastProgress = prog;
                showToast(prog, 'info');
            }
        } else if (_lastProgress && result.backgroundTask && !result.backgroundTask.running) {
            // Task just finished
            const prog = result.backgroundTask.progress;
            if (prog !== _lastProgress) {
                _lastProgress = prog;
                const err = result.backgroundTask.error;
                showToast(prog, err ? 'error' : 'success');
                // Refresh gestures from backend
                syncGesturesFromBackend();
            }
        }
    }, 2000);
}