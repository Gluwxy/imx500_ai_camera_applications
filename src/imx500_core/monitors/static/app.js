const DEFAULT_WHEP_PORT = 8889;
const ICE_GATHERING_TIMEOUT_MS = 2000;
const STATUS_URL = '/api/status';
const OVERLAY_URL = '/api/overlay';
const STATUS_INTERVAL_MS = 2000;

const SECTIONS = ['live', 'hardware', 'events'];
const DEFAULT_SECTION = 'live';
const MOBILE_QUERY = '(max-width: 640px)';

// Each gauge reads its value (and max) from the /api/status JSON.
// thresholds are fractions of max: < t1 green, < t2 yellow, < t3 peach, else red.
const GAUGES = [
    {
        id: 'imx500-temp', label: 'IMX500 temp', unit: '°C',
        thresholds: [0.65, 0.80, 0.95], // saturates at 80 °C, then the frontend times out
        max: () => 80,
        read: (s) => s.imx500?.temp_c,
    },
    {
        id: 'soc-temp', label: 'SoC temp (CPU+GPU)', unit: '°C',
        thresholds: [0.70, 0.82, 0.94], // the Pi soft-throttles at ~80 °C
        max: () => 85,
        read: (s) => s.soc?.temp_c,
    },
    {
        id: 'cpu', label: 'CPU', unit: '%',
        thresholds: [0.50, 0.75, 0.90],
        max: () => 100,
        read: (s) => s.cpu?.pct,
    },
    {
        id: 'ram', label: 'RAM', unit: 'MB', showTotal: true,
        thresholds: [0.60, 0.80, 0.90],
        max: (s) => s.ram?.total_mb,
        read: (s) => s.ram?.used_mb,
    },
    {
        id: 'swap', label: 'Swap', unit: 'MB', showTotal: true,
        thresholds: [0.25, 0.50, 0.75], // swapping at all already hurts a Pi Zero
        max: (s) => s.swap?.total_mb || null, // SwapTotal 0: no swap → "—"
        read: (s) => s.swap?.used_mb,
    },
    {
        id: 'disk-read', label: 'Disk read', unit: 'MB/s', decimals: 1,
        thresholds: [0.40, 0.70, 0.90],
        max: () => 25, // typical microSD sequential read
        read: (s) => s.disk?.read_mb_s,
    },
    {
        id: 'disk-write', label: 'Disk write', unit: 'MB/s', decimals: 1,
        thresholds: [0.40, 0.70, 0.90],
        max: () => 10, // typical microSD sequential write
        read: (s) => s.disk?.write_mb_s,
    },
];

const LEVEL_COLORS = ['var(--green)', 'var(--yellow)', 'var(--peach)', 'var(--red)'];

// Rendered clip rows by clip name: rows are created once and only their
// status is patched afterwards, so a clip that is playing is never interrupted.
const eventRows = new Map();

let peerConnection = null;
let statusTimer = null;
let whepPort = DEFAULT_WHEP_PORT;
let pendingOverlay = false; // a toggle is being sent: don't overwrite it

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btnConnect').addEventListener('click', connect);
    document.getElementById('toggle-detections').addEventListener('change', toggleDetections);
    document.getElementById('stream-source').addEventListener('change', () => {
        if (peerConnection) connect(); // switch the live view to the new source
    });
    createGauges();
    setupMenu();

    window.addEventListener('hashchange', showSectionFromHash);
    document.addEventListener('visibilitychange', updatePolling);

    showSectionFromHash();
    updatePolling();
});

/* ---------- Navigation ---------- */

function currentSection() {
    const name = window.location.hash.slice(1);
    return SECTIONS.includes(name) ? name : DEFAULT_SECTION;
}

function showSectionFromHash() {
    const active = currentSection();

    for (const name of SECTIONS) {
        document.getElementById(name).hidden = name !== active;
    }
    for (const link of document.querySelectorAll('.menu a')) {
        const isActive = link.dataset.section === active;
        link.classList.toggle('active', isActive);
        if (isActive) link.setAttribute('aria-current', 'page');
        else link.removeAttribute('aria-current');
    }

    // Leaving the live view frees the WebRTC stream on the Pi
    if (active !== 'live') disconnect();
}

/* ---------- Hamburger menu (≤ 640px) ---------- */

function setupMenu() {
    const navbar = document.querySelector('.navbar');
    const toggle = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    const mobile = window.matchMedia(MOBILE_QUERY);

    const isOpen = () => toggle.getAttribute('aria-expanded') === 'true';
    const setOpen = (open) => {
        toggle.setAttribute('aria-expanded', String(open));
        toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
        menu.classList.toggle('open', open);
    };

    toggle.addEventListener('click', () => setOpen(!isOpen()));

    // Close on link click
    menu.addEventListener('click', (event) => {
        if (event.target.closest('a')) setOpen(false);
    });

    // Close on Escape (and return focus to the toggle)
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && isOpen()) {
            setOpen(false);
            toggle.focus();
        }
    });

    // Close on a click outside the nav
    document.addEventListener('click', (event) => {
        if (isOpen() && !navbar.contains(event.target)) setOpen(false);
    });

    // Close when the viewport grows past the breakpoint
    mobile.addEventListener('change', (event) => {
        if (!event.matches) setOpen(false);
    });
}

/* ---------- Status polling ---------- */

// Polls only while the tab is visible, so a background tab costs the Pi nothing.
function updatePolling() {
    if (document.hidden) {
        clearInterval(statusTimer);
        statusTimer = null;
    } else if (statusTimer === null) {
        refreshStatus();
        statusTimer = setInterval(refreshStatus, STATUS_INTERVAL_MS);
    }
}

async function refreshStatus() {
    try {
        const response = await fetch(STATUS_URL, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const status = await response.json();

        for (const gauge of GAUGES) {
            updateGauge(gauge, gauge.read(status), gauge.max(status));
        }

        const { ok, last_frame_s } = status.watchdog;
        setBadge('badge-watchdog', `Watchdog: ${ok ? 'OK' : 'STALLED'} (last frame ${last_frame_s}s ago)`, ok ? 'ok' : 'bad');

        const throttled = status.soc.throttled;
        if (throttled === null) setBadge('badge-throttling', 'Throttling: unknown', 'unknown');
        else setBadge('badge-throttling', `Throttling: ${throttled ? 'YES' : 'no'}`, throttled ? 'bad' : 'ok');

        updateStreamOptions(status);
        renderEvents(status.clips ?? []);
        setFooter(`Updated ${new Date().toLocaleTimeString()}`, 'ok');
    } catch (error) {
        console.error('Status error:', error);
        setBadge('badge-watchdog', 'Watchdog: app offline', 'bad');
        setFooter('App offline', 'bad');
    }
}

function setBadge(id, text, state) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.dataset.state = state; // 'ok' | 'bad' | 'unknown'
}

function setFooter(text, state) {
    const el = document.getElementById('last-update');
    el.textContent = text;
    el.dataset.state = state;
}

/* ---------- Stream options ---------- */

function updateStreamOptions(status) {
    whepPort = status.stream?.whep_port ?? DEFAULT_WHEP_PORT;

    const select = document.getElementById('stream-source');
    const sources = status.stream?.sources ?? [];
    const current = select.value;
    const existing = [...select.options].map((o) => o.value);
    if (sources.join() !== existing.join()) {
        select.replaceChildren(...sources.map((name) => new Option(name, name)));
        if (sources.includes(current)) select.value = current;
    }

    const toggle = document.getElementById('toggle-detections');
    const shown = status.overlay?.show_detections;
    if (!pendingOverlay && typeof shown === 'boolean') toggle.checked = shown;
    toggle.disabled = typeof shown !== 'boolean'; // camera not running yet
}

// The boxes are drawn into the video itself: the toggle affects every viewer.
async function toggleDetections(event) {
    const toggle = event.target;
    pendingOverlay = true;
    try {
        const response = await fetch(OVERLAY_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ show_detections: toggle.checked }),
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        toggle.checked = (await response.json()).show_detections;
    } catch (error) {
        console.error('Overlay toggle error:', error);
        toggle.checked = !toggle.checked; // revert
    } finally {
        pendingOverlay = false;
    }
}

function streamPath() {
    return document.getElementById('stream-source').value;
}

/* ---------- Gauges ---------- */

function createGauges() {
    const template = document.getElementById('gauge-template');
    const container = document.getElementById('gauges');

    for (const gauge of GAUGES) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.id = gauge.id;
        node.querySelector('.label').textContent = gauge.label;
        container.appendChild(node);
    }
}

function levelColor(fraction, [t1, t2, t3]) {
    if (fraction < t1) return LEVEL_COLORS[0];
    if (fraction < t2) return LEVEL_COLORS[1];
    if (fraction < t3) return LEVEL_COLORS[2];
    return LEVEL_COLORS[3];
}

function updateGauge(gauge, value, max) {
    const el = document.getElementById(gauge.id);
    const fill = el.querySelector('.fill');
    const num = el.querySelector('.value-num');
    const unit = el.querySelector('.value-unit');

    el.querySelector('.scale-max').textContent = typeof max === 'number' ? Math.round(max) : '—';

    if (typeof value !== 'number' || typeof max !== 'number' || max <= 0) {
        el.style.removeProperty('--level');
        fill.style.strokeDasharray = '0 100';
        num.textContent = '—';
        unit.textContent = '';
        return;
    }

    const fraction = Math.min(Math.max(value / max, 0), 1);
    const pct = fraction * 100;

    el.style.setProperty('--level', levelColor(fraction, gauge.thresholds)); // arc + value text
    fill.style.strokeDasharray = `${pct} 100`;
    num.textContent = gauge.showTotal
        ? `${Math.round(value)} / ${Math.round(max)}`
        : value.toFixed(gauge.decimals ?? 0);
    unit.textContent = gauge.unit;
}

/* ---------- Events (alert clips) ---------- */

function renderEvents(clips) {
    const list = document.getElementById('events-list');
    const names = new Set(clips.map((c) => c.name));

    // Drop rows whose clip was pruned from data/clips
    for (const [name, row] of eventRows) {
        if (!names.has(name)) {
            row.remove();
            eventRows.delete(name);
        }
    }

    // Clips arrive newest first; walk them backwards and prepend new ones so
    // the newest stays on top
    for (const clip of [...clips].reverse()) {
        let row = eventRows.get(clip.name);
        if (!row) {
            row = createEventRow(clip);
            eventRows.set(clip.name, row);
            list.prepend(row);
        }
        updateClip(row, clip);
    }

    const empty = list.querySelector('.empty');
    if (eventRows.size === 0 && !empty) {
        const li = document.createElement('li');
        li.className = 'empty';
        li.textContent = 'No clips yet';
        list.appendChild(li);
    } else if (eventRows.size > 0 && empty) {
        empty.remove();
    }
}

// Cards start minimized; the toggle opens/closes the clip at any time.
function createEventRow(clip) {
    const row = document.getElementById('event-template').content.firstElementChild.cloneNode(true);
    const head = row.querySelector('.event-head');
    const body = row.querySelector('.event-body');
    const timeEl = row.querySelector('.event-time');

    body.id = `event-body-${clip.name.replace(/[^\w-]/g, '_')}`;
    head.setAttribute('aria-controls', body.id);
    if (clip.time) {
        const time = new Date(clip.time);
        timeEl.dateTime = time.toISOString();
        timeEl.textContent = time.toLocaleString();
    }
    row.querySelector('.event-text').textContent =
        clip.track_id === null ? clip.event : `${clip.event} · #${clip.track_id}`;
    row.querySelector('.clip-download').href = clip.url;

    head.addEventListener('click', () => setExpanded(row, head.getAttribute('aria-expanded') !== 'true'));
    return row;
}

function setExpanded(row, open) {
    const head = row.querySelector('.event-head');
    const video = row.querySelector('.clip-video');

    head.setAttribute('aria-expanded', String(open));
    head.setAttribute('aria-label', open ? 'Hide clip' : 'Show clip');
    row.querySelector('.event-toggle .material-symbols-outlined').textContent =
        open ? 'arrow_drop_up' : 'arrow_drop_down';
    row.querySelector('.event-body').hidden = !open;
    row.classList.toggle('open', open);

    if (open) {
        loadClip(row);
    } else {
        video.pause();
    }
}

// A clip is only requested from the Pi when its card is opened (preload="none"
// until then), so a long event list never downloads videos by itself.
function loadClip(row) {
    const video = row.querySelector('.clip-video');
    if (row.dataset.clipState === 'ready' && !video.getAttribute('src')) {
        video.preload = 'metadata';
        video.src = row.querySelector('.clip-download').getAttribute('href');
    }
}

const CLIP_STATUS_TEXT = {
    recording: 'Recording clip…',
    missing: 'Clip no longer available',
};

function updateClip(row, clip) {
    const state = clip.state ?? 'missing';
    if (row.dataset.clipState === state) return; // nothing changed

    row.dataset.clipState = state;
    const ready = state === 'ready';
    row.querySelector('.clip-player').hidden = !ready;
    const status = row.querySelector('.clip-status');
    status.hidden = ready;
    status.textContent = CLIP_STATUS_TEXT[state] ?? '';

    // Became ready while the card is open: load it now
    if (ready && row.classList.contains('open')) loadClip(row);
}

/* ---------- WebRTC video (WHEP) ---------- */

// Resolves once ICE candidates are gathered (or after a timeout, to avoid hanging).
function waitForIceGathering(pc) {
    return new Promise((resolve) => {
        if (pc.iceGatheringState === 'complete') return resolve();

        const timeout = setTimeout(resolve, ICE_GATHERING_TIMEOUT_MS);
        pc.addEventListener('icegatheringstatechange', () => {
            if (pc.iceGatheringState === 'complete') {
                clearTimeout(timeout);
                resolve();
            }
        });
    });
}

function disconnect() {
    if (!peerConnection) return;
    peerConnection.close();
    peerConnection = null;
    document.getElementById('videoElement').srcObject = null;
    document.getElementById('btnConnect').textContent = 'Connect';
}

async function connect() {
    const video = document.getElementById('videoElement');
    const button = document.getElementById('btnConnect');

    disconnect(); // close any previous connection before reconnecting

    button.disabled = true;
    button.textContent = 'Connecting...';

    try {
        peerConnection = new RTCPeerConnection();
        peerConnection.addTransceiver('video', { direction: 'recvonly' });
        peerConnection.ontrack = (event) => {
            video.srcObject = event.streams[0];
        };

        const offer = await peerConnection.createOffer();
        await peerConnection.setLocalDescription(offer);
        await waitForIceGathering(peerConnection);

        // Send the offer to MediaMTX using the WHEP standard
        const serverIP = window.location.hostname;
        const response = await fetch(`http://${serverIP}:${whepPort}/${encodeURIComponent(streamPath())}/whep`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/sdp' },
            body: peerConnection.localDescription.sdp,
        });

        if (!response.ok) throw new Error(`WHEP request failed: ${response.status} ${response.statusText}`);

        await peerConnection.setRemoteDescription({
            type: 'answer',
            sdp: await response.text(),
        });

        button.textContent = 'Reconnect';
    } catch (error) {
        console.error('Connection error:', error);
        alert('Connection failed. Check the console for details.');
        button.textContent = 'Connect';
    } finally {
        button.disabled = false;
    }
}
