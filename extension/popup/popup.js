/**
 * Popup Script
 * Xử lý UI và tương tác với background.js
 */

const API_ENDPOINT = "http://localhost:8000";

// DOM Elements
let enableToggle, modeRadios, groupInfo, currentGroup;
let statTotal, statReal, statFake, resetStatsBtn;
let manualText, analyzeBtn, manualResult, resultText, resultConfidence;
let serverStatus;

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', () => {
  initDOMReferences();
  loadSettings();
  loadStats();
  checkServerStatus();
  setupEventListeners();
});

function initDOMReferences() {
  enableToggle = document.getElementById('enable-toggle');
  modeRadios = document.querySelectorAll('input[name="mode"]');
  groupInfo = document.getElementById('group-info');
  currentGroup = document.getElementById('current-group');
  
  statTotal = document.getElementById('stat-total');
  statReal = document.getElementById('stat-real');
  statFake = document.getElementById('stat-fake');
  resetStatsBtn = document.getElementById('reset-stats');
  
  manualText = document.getElementById('manual-text');
  analyzeBtn = document.getElementById('analyze-btn');
  manualResult = document.getElementById('manual-result');
  resultText = document.getElementById('result-text');
  resultConfidence = document.getElementById('result-confidence');
  
  serverStatus = document.getElementById('server-status');
}

// ============================================================================
// SETTINGS
// ============================================================================
function loadSettings() {
  chrome.storage.sync.get(['enabled', 'mode', 'groupId'], (data) => {
    // Enable toggle
    enableToggle.checked = data.enabled !== false;
    
    // Mode
    const mode = data.mode || 'feed';
    document.querySelector(`input[name="mode"][value="${mode}"]`).checked = true;
    
    // Group info
    if (mode === 'group' && data.groupId) {
      groupInfo.classList.remove('hidden');
      currentGroup.textContent = data.groupId;
    } else {
      groupInfo.classList.add('hidden');
    }
  });
}

function saveSettings() {
  const mode = document.querySelector('input[name="mode"]:checked').value;
  const enabled = enableToggle.checked;
  
  chrome.storage.sync.set({ enabled, mode }, () => {
    console.log('[Popup] Settings saved');
  });
  
  // Notify content script
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]?.url?.includes('facebook.com')) {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: 'toggleEnabled',
        enabled: enabled
      });
    }
  });
}

// ============================================================================
// STATS
// ============================================================================
function loadStats() {
  chrome.storage.sync.get(['stats'], (data) => {
    const stats = data.stats || { total: 0, real: 0, fake: 0 };
    statTotal.textContent = stats.total;
    statReal.textContent = stats.real;
    statFake.textContent = stats.fake;
  });
}

function resetStats() {
  chrome.storage.sync.set({
    stats: { total: 0, real: 0, fake: 0 }
  }, () => {
    loadStats();
    console.log('[Popup] Stats reset');
  });
}

// ============================================================================
// MANUAL ANALYSIS
// ============================================================================
async function analyzeManualText() {
  const text = manualText.value.trim();
  
  if (!text || text.length < 10) {
    alert('Vui lòng nhập nội dung cần phân tích (ít nhất 10 ký tự)');
    return;
  }
  
  // Show loading
  analyzeBtn.textContent = 'Đang phân tích...';
  analyzeBtn.disabled = true;
  
  try {
    const response = await fetch(`${API_ENDPOINT}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        content_text: text,
        timestamp: new Date().toISOString(),
        mode: 'feed'
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    displayManualResult(result);
    
  } catch (error) {
    console.error('[Popup] Analysis error:', error);
    alert('Lỗi kết nối server. Vui lòng kiểm tra server đang chạy.');
  } finally {
    analyzeBtn.textContent = 'Phân tích';
    analyzeBtn.disabled = false;
  }
}

function displayManualResult(result) {
  manualResult.classList.remove('hidden', 'real', 'fake');
  
  if (result.label === 0) {
    manualResult.classList.add('real');
    resultText.textContent = '✓ Tin thật';
  } else {
    manualResult.classList.add('fake');
    resultText.textContent = '⚠ Nghi ngờ tin giả';
  }
  
  resultConfidence.textContent = `Độ tin cậy: ${(result.confidence * 100).toFixed(1)}%`;
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
      serverStatus.textContent = 'Online ✓';
      serverStatus.classList.add('online');
      serverStatus.classList.remove('offline');
    } else {
      throw new Error('Server error');
    }
  } catch (error) {
    serverStatus.textContent = 'Offline ✗';
    serverStatus.classList.add('offline');
    serverStatus.classList.remove('online');
  }
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================
function setupEventListeners() {
  // Enable toggle
  enableToggle.addEventListener('change', saveSettings);
  
  // Mode selection
  modeRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
      if (e.target.value === 'group') {
        groupInfo.classList.remove('hidden');
      } else {
        groupInfo.classList.add('hidden');
      }
      saveSettings();
    });
  });
  
  // Reset stats
  resetStatsBtn.addEventListener('click', resetStats);
  
  // Manual analysis
  analyzeBtn.addEventListener('click', analyzeManualText);
  
  // Enter key in textarea
  manualText.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      analyzeManualText();
    }
  });
}
