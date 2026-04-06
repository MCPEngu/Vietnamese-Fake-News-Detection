/**
 * Popup Script - Real-time updates
 */

const API_ENDPOINT = "http://localhost:8000";

// DOM Elements
let enableToggle, contextIcon, contextMode, groupDetail, groupId;
let statTotal, statReal, statFake, barReal, barFake, resetStatsBtn;
let serverStatus, statusDot;

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initDOMReferences();
  loadSettings();
  loadStats();
  loadContext();
  checkServerStatus();
  setupEventListeners();
  setupRealtimeUpdates();
});

function initDOMReferences() {
  enableToggle = document.getElementById('enable-toggle');
  contextIcon = document.getElementById('context-icon');
  contextMode = document.getElementById('context-mode');
  groupDetail = document.getElementById('group-detail');
  groupId = document.getElementById('group-id');
  
  statTotal = document.getElementById('stat-total');
  statReal = document.getElementById('stat-real');
  statFake = document.getElementById('stat-fake');
  barReal = document.getElementById('bar-real');
  barFake = document.getElementById('bar-fake');
  resetStatsBtn = document.getElementById('reset-stats');
  
  serverStatus = document.getElementById('server-status');
  statusDot = document.getElementById('status-dot');
}

// ============================================================================
// SETTINGS
// ============================================================================
function loadSettings() {
  chrome.storage.sync.get(['enabled'], (data) => {
    enableToggle.checked = data.enabled !== false;
  });
}

function saveEnabled(enabled) {
  chrome.storage.sync.set({ enabled });
  
  // Notify content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.url?.includes('facebook.com')) {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'toggleEnabled',
        enabled: enabled
      }).catch(() => {});
    }
  });
}

// ============================================================================
// CONTEXT
// ============================================================================
function loadContext() {
  // Lấy context từ storage (reliable hơn background memory)
  chrome.storage.local.get(['currentContext'], (data) => {
    console.log(`[Popup] loadContext from storage:`, data);
    const context = data.currentContext || { mode: "feed", groupId: null };
    updateContextUI(context);
  });
}

function updateContextUI(context) {
  if (context.mode === 'group' && context.groupId) {
    contextIcon.textContent = '👥';
    contextMode.textContent = 'Nhóm';
    groupId.textContent = `ID: ${context.groupId}`;
    groupDetail.classList.remove('hidden');
  } else {
    contextIcon.textContent = '📰';
    contextMode.textContent = 'Feed';
    groupDetail.classList.add('hidden');
  }
}

// ============================================================================
// STATS
// ============================================================================
function loadStats() {
  chrome.storage.sync.get(['stats'], (data) => {
    updateStatsUI(data.stats || { total: 0, real: 0, fake: 0 });
  });
}

function updateStatsUI(stats) {
  // Animate number changes
  animateNumber(statTotal, stats.total);
  animateNumber(statReal, stats.real);
  animateNumber(statFake, stats.fake);
  
  // Update progress bar
  const total = stats.total || 1; // Avoid division by zero
  const realPercent = (stats.real / total) * 100;
  const fakePercent = (stats.fake / total) * 100;
  
  barReal.style.width = `${realPercent}%`;
  barFake.style.width = `${fakePercent}%`;
}

function animateNumber(element, newValue) {
  const current = parseInt(element.textContent) || 0;
  if (current === newValue) return;
  
  // Simple animation
  element.textContent = newValue;
  element.style.transform = 'scale(1.2)';
  setTimeout(() => {
    element.style.transform = 'scale(1)';
  }, 150);
}

function resetStats() {
  chrome.storage.sync.set({
    stats: { total: 0, real: 0, fake: 0 }
  }, () => {
    loadStats();
  });
}

// ============================================================================
// SERVER STATUS
// ============================================================================
async function checkServerStatus() {
  try {
    const response = await fetch(`${API_ENDPOINT}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });
    
    if (response.ok) {
      serverStatus.textContent = 'Server Online';
      statusDot.className = 'status-dot online';
    } else {
      throw new Error('Server error');
    }
  } catch (e) {
    serverStatus.textContent = 'Server Offline';
    statusDot.className = 'status-dot offline';
  }
}

// ============================================================================
// REAL-TIME UPDATES
// ============================================================================
function setupRealtimeUpdates() {
  // Listen for storage changes (real-time stats + context update)
  chrome.storage.onChanged.addListener((changes, namespace) => {
    console.log(`[Popup] Storage changed: ${namespace}`, changes);
    if (namespace === 'sync' && changes.stats) {
      updateStatsUI(changes.stats.newValue);
    }
    if (namespace === 'local' && changes.currentContext) {
      console.log(`[Popup] Context changed:`, changes.currentContext.newValue);
      updateContextUI(changes.currentContext.newValue);
    }
  });
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
  enableToggle.addEventListener('change', (e) => {
    saveEnabled(e.target.checked);
  });
  
  resetStatsBtn.addEventListener('click', resetStats);
}
