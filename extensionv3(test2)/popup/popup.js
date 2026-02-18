/**
 * Popup script for the Vietnamese Toxic Language Detector Extension
 * Handles user settings, analysis, and account management
 */

/**
 * API Endpoint configuration
 * - For production, inject window.TOXIC_API_ENDPOINT in popup.html or set in chrome.storage.sync
 * - For dev, fallback to localhost
 */
let API_ENDPOINT = "http://localhost:7860";
if (typeof window !== 'undefined' && window.TOXIC_API_ENDPOINT) {
  API_ENDPOINT = window.TOXIC_API_ENDPOINT;
} else if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.sync) {
  chrome.storage.sync.get('api_endpoint', (data) => {
    if (data.api_endpoint) {
      API_ENDPOINT = data.api_endpoint;
    }
  });
}
// Authentication storage key
const AUTH_STORAGE_KEY = "toxicDetector_auth";

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', initializePopup);

// Tab elements
const tabButtons = document.querySelectorAll('.tab-button');
const tabPanes = document.querySelectorAll('.tab-pane');

// DOM references for UI elements
let enableToggle, thresholdSlider, thresholdValue, highlightToxic, saveDataCheckbox;
let platformFacebook, platformYoutube, platformTwitter, platformTiktok, platformZalo;
let statScanned, statClean, statOffensive, statHate, statSpam, resetStatsBtn;
let showClean, showOffensive, showHate, showSpam;
let analyzeText, analyzeButton, resultContainer, resultCategory, resultConfidence;
let progressClean, progressOffensive, progressHate, progressSpam;
let percentClean, percentOffensive, percentHate, percentSpam;
let anonymousMode, regionSelect, reportButton, resultAdvanced, resultEmotion, resultIntent, resultKeywords;
let themeLight, themeDark, themeAuto;
let currentPlatform;

// Authentication UI elements
let loginSection, registerSection, profileSection, resetPasswordSection, changePasswordSection;
let loginForm, registerForm, resetPasswordForm, changePasswordForm;
let showRegisterBtn, showLoginBtn, forgotPasswordBtn, cancelResetBtn, cancelChangePasswordBtn;
let logoutBtn, changePasswordBtn;
let profileUsername, profileEmail, profileName, profileRole;

// Batch Analyze UI elements
let singleModeBtn, batchModeBtn, singleAnalyzeArea, batchAnalyzeArea;
let batchInput, batchDetectBtn, batchFileInput, batchResultContainer, batchResultList;

// Category mappings
const categoryNames = {
  0: "Bình thường",
  1: "Xúc phạm",
  2: "Thù ghét",
  3: "Spam"
};

const categoryClasses = {
  0: "clean",
  1: "offensive",
  2: "hate",
  3: "spam"
};

/**
 * Initialize the popup UI - Enhanced with UX improvements
 */
function initializePopup() {
  // Get DOM references for all elements
  initializeDOMReferences();
  
  // Apply theme immediately to avoid FOUC (Flash of Unstyled Content)
  setTheme('auto');
  
  // Check authentication status and update UI accordingly FIRST
  checkAuthStatus().then(() => {
    // Setup tab navigation AFTER checking auth (to potentially hide tabs)
    setupTabNavigation();

    // Load settings and statistics
    loadSettings();
    loadStats(); // Local stats for anonymous, API stats for logged in

    // Setup event handlers for all UI interactions
    setupEventHandlers();

    // Detect current platform from active tab
    detectCurrentPlatform();

    // Setup batch detect UI and stats tab AFTER checking auth
    setupBatchDetectUI();
    setupStatsTab(); // Load stats for logged-in user if applicable

    // Setup analyze mode switch AFTER checking auth
    setupAnalyzeModeSwitch();
    
    // Setup accessibility features
    setupAccessibility();
    
    // Setup enhanced UI interactions
    setupEnhancedUIInteractions();
  });
}

/**
 * Initialize all DOM references
 */
function initializeDOMReferences() {
  // Settings elements
  enableToggle = document.getElementById('enable-toggle');
  thresholdSlider = document.getElementById('threshold');
  thresholdValue = document.getElementById('threshold-value');
  highlightToxic = document.getElementById('highlight-toxic');
  saveDataCheckbox = document.getElementById('save-data');
  
  // Platform settings
  platformFacebook = document.getElementById('platform-facebook');
  platformYoutube = document.getElementById('platform-youtube');
  platformTwitter = document.getElementById('platform-twitter');
  platformTiktok = document.getElementById('platform-tiktok');
  platformZalo = document.getElementById('platform-zalo');

  // Classification visibility toggles
  showClean = document.getElementById('show-clean');
  showOffensive = document.getElementById('show-offensive');
  showHate = document.getElementById('show-hate');
  showSpam = document.getElementById('show-spam');
  
  // Theme settings
  themeLight = document.getElementById('theme-light');
  themeDark = document.getElementById('theme-dark');
  themeAuto = document.getElementById('theme-auto');
  
  // Statistics elements
  statScanned = document.getElementById('stat-scanned');
  statClean = document.getElementById('stat-clean');
  statOffensive = document.getElementById('stat-offensive');
  statHate = document.getElementById('stat-hate');
  statSpam = document.getElementById('stat-spam');
  resetStatsBtn = document.getElementById('reset-stats');
  
  // Text analysis elements
  analyzeText = document.getElementById('analyze-text');
  analyzeButton = document.getElementById('analyze-button');
  resultContainer = document.getElementById('result-container');
  resultCategory = document.getElementById('result-category');
  resultConfidence = document.getElementById('result-confidence');
  progressClean = document.getElementById('progress-clean');
  progressOffensive = document.getElementById('progress-offensive');
  progressHate = document.getElementById('progress-hate');
  progressSpam = document.getElementById('progress-spam');
  percentClean = document.getElementById('percent-clean');
  percentOffensive = document.getElementById('percent-offensive');
  percentHate = document.getElementById('percent-hate');
  percentSpam = document.getElementById('percent-spam');
  
  // Advanced analysis elements
  anonymousMode = document.getElementById('anonymous-mode');
  regionSelect = document.getElementById('region');
  reportButton = document.getElementById('report-button');
  resultAdvanced = document.getElementById('result-advanced');
  resultEmotion = document.getElementById('result-emotion');
  resultIntent = document.getElementById('result-intent');
  resultKeywords = document.getElementById('result-keywords');
  
  // Platform detection
  currentPlatform = document.getElementById('current-platform');
  
  // Authentication elements
  loginSection = document.getElementById('login-section');
  registerSection = document.getElementById('register-section');
  profileSection = document.getElementById('profile-section');
  resetPasswordSection = document.getElementById('reset-password-section');
  changePasswordSection = document.getElementById('change-password-section');
  
  loginForm = document.getElementById('login-form');
  registerForm = document.getElementById('register-form');
  resetPasswordForm = document.getElementById('reset-password-form');
  changePasswordForm = document.getElementById('change-password-form');
  
  showRegisterBtn = document.getElementById('show-register');
  showLoginBtn = document.getElementById('show-login');
  forgotPasswordBtn = document.getElementById('forgot-password');
  cancelResetBtn = document.getElementById('cancel-reset');
  cancelChangePasswordBtn = document.getElementById('cancel-change-password');
  
  logoutBtn = document.getElementById('logout-btn');
  changePasswordBtn = document.getElementById('change-password-btn');
  
  profileUsername = document.getElementById('profile-username');
  profileEmail = document.getElementById('profile-email');
  profileName = document.getElementById('profile-name');
  profileRole = document.getElementById('profile-role');

  // Batch Analyze elements
  singleModeBtn = document.getElementById('single-mode-btn');
  batchModeBtn = document.getElementById('batch-mode-btn');
  singleAnalyzeArea = document.getElementById('single-analyze-area');
  batchAnalyzeArea = document.getElementById('batch-analyze-area');
  batchInput = document.getElementById('batch-input');
  batchDetectBtn = document.getElementById('batch-detect-btn');
  batchFileInput = document.getElementById('batch-file-input');
  batchResultContainer = document.getElementById('batch-result-container');
  batchResultList = document.getElementById('batch-result-list');
}

/**
 * Setup tab navigation
 */
function setupTabNavigation() {
  if (!tabButtons || !tabButtons.length) return;
  
  const isLoggedIn = document.body.classList.contains('logged-in');
  console.log("setupTabNavigation - isLoggedIn:", isLoggedIn);
  
  // Đảm bảo ít nhất 2 tab luôn được hiển thị (analyze và account)
  let visibleTabsCount = 0;
  
  tabButtons.forEach(button => {
    const requiresAuth = button.classList.contains('requires-auth');
    const tabName = button.getAttribute('data-tab');
    const tabPane = document.getElementById(`${tabName}-tab`);
    
    // Always show analyze and account tabs
    if (tabName === 'analyze' || tabName === 'account') {
      button.style.display = '';
      if (tabPane) tabPane.style.display = '';
      visibleTabsCount++;
    } 
    // Hide tabs that require auth if not logged in
    else if (requiresAuth && !isLoggedIn) {
      button.style.display = 'none';
      if (tabPane) {
        tabPane.style.display = 'none'; // Hide the pane content as well
      }
    } else {
      // Show tabs that don't require auth or when logged in
      button.style.display = '';
      if (tabPane) tabPane.style.display = '';
      visibleTabsCount++;
    }

    button.addEventListener('click', () => {
      // Remove active class from all buttons and panes
      tabButtons.forEach(btn => btn.classList.remove('active'));
      tabPanes.forEach(pane => pane.classList.remove('active'));
      
      // Add active class to clicked button
      button.classList.add('active');
      
      // Show corresponding pane
      const targetTabPane = document.getElementById(`${tabName}-tab`);
      if (targetTabPane) {
        targetTabPane.classList.add('active');
        // Special handling for stats tab to load data when activated
        if (tabName === 'stats' && isLoggedIn) {
           setupStatsTab(); // Reload stats data when tab is clicked
        }
      }
    });
  });

  console.log("Visible tabs count:", visibleTabsCount);
  
  // Đảm bảo luôn có một tab nào đó được hiển thị và active
  // Activate the first visible tab by default if no tab is active
  const activeTab = Array.from(tabButtons).find(btn => btn.classList.contains('active') && btn.style.display !== 'none');
  if (!activeTab) {
    const firstVisibleTab = Array.from(tabButtons).find(btn => btn.style.display !== 'none');
    if (firstVisibleTab) {
      console.log("Activating default tab:", firstVisibleTab.getAttribute('data-tab'));
      firstVisibleTab.click();
    } else {
      // Fallback nếu không tìm thấy tab nào - hiển thị tab analyze
      const analyzeTab = document.querySelector('.tab-button[data-tab="analyze"]');
      if (analyzeTab) {
        console.log("Fallback to analyze tab");
        analyzeTab.style.display = '';
        analyzeTab.click();
      }
    }
  }
}

/**
 * Detect the current platform from the active tab
 */
function detectCurrentPlatform() {
  if (!currentPlatform) return;
  
  chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
    if (tabs.length === 0) return;
    
    const url = tabs[0].url;
    
    // Set platform name based on URL
    if (url.includes('facebook.com')) {
      currentPlatform.textContent = 'Facebook';
      currentPlatform.className = 'platform facebook';
    } else if (url.includes('youtube.com')) {
      currentPlatform.textContent = 'YouTube';
      currentPlatform.className = 'platform youtube';
    } else if (url.includes('twitter.com') || url.includes('x.com')) {
      currentPlatform.textContent = 'Twitter';
      currentPlatform.className = 'platform twitter';
    } else if (url.includes('tiktok.com')) {
      currentPlatform.textContent = 'TikTok';
      currentPlatform.className = 'platform tiktok';
    } else if (url.includes('zalo.me')) {
      currentPlatform.textContent = 'Zalo';
      currentPlatform.className = 'platform zalo';
    } else {
      currentPlatform.textContent = 'Khác';
      currentPlatform.className = 'platform other';
    }
  });
}

/**
 * Load settings from storage
 */
async function loadSettings() {
  chrome.storage.sync.get(null, async function(data) {
    let serverSettings = null;
    const authData = await getAuthData();
    if (authData && authData.access_token) {
      serverSettings = await fetchServerSettings();
    }
    const settings = serverSettings || data;
    // Extension enabled
    if (enableToggle) {
      enableToggle.checked = settings.enabled !== undefined ? settings.enabled : true;
    }
    
    // Threshold value
    if (thresholdSlider && thresholdValue && settings.threshold !== undefined) {
      thresholdSlider.value = settings.threshold;
      thresholdValue.textContent = settings.threshold;
    }
    
    // Highlight toxic
    if (highlightToxic) {
      highlightToxic.checked = settings.highlightToxic !== undefined ? settings.highlightToxic : true;
    }
    
    // Save data option
    if (saveDataCheckbox) {
      saveDataCheckbox.checked = settings.saveData !== undefined ? settings.saveData : true;
    }
    
    // Platform settings
    if (settings.platforms) {
      if (platformFacebook) {
        platformFacebook.checked = settings.platforms.facebook !== undefined ? settings.platforms.facebook : true;
      }
      if (platformYoutube) {
        platformYoutube.checked = settings.platforms.youtube !== undefined ? settings.platforms.youtube : true;
      }
      if (platformTwitter) {
        platformTwitter.checked = settings.platforms.twitter !== undefined ? settings.platforms.twitter : true;
      }
      if (platformTiktok) {
        platformTiktok.checked = settings.platforms.tiktok !== undefined ? settings.platforms.tiktok : false;
      }
      if (platformZalo) {
        platformZalo.checked = settings.platforms.zalo !== undefined ? settings.platforms.zalo : false;
      }
    }
    
    // Display options
    if (settings.displayOptions) {
      if (showClean) {
        showClean.checked = settings.displayOptions.showClean !== undefined ? settings.displayOptions.showClean : true;
      }
      if (showOffensive) {
        showOffensive.checked = settings.displayOptions.showOffensive !== undefined ? settings.displayOptions.showOffensive : true;
      }
      if (showHate) {
        showHate.checked = settings.displayOptions.showHate !== undefined ? settings.displayOptions.showHate : true;
      }
      if (showSpam) {
        showSpam.checked = settings.displayOptions.showSpam !== undefined ? settings.displayOptions.showSpam : true;
      }
    }
    
    // Theme settings
    if (settings.theme) {
      setTheme(settings.theme);
    } else {
      // Default to auto
      setTheme('auto');
    }
  });
}

/**
 * Load statistics for the selected period
 */
async function loadStatsForPeriod(period) {
  try {
    // Get authentication token
    const authData = await getAuthData();
    const isLoggedIn = document.body.classList.contains('logged-in');
    
    if (!authData || !authData.access_token || !isLoggedIn) {
      // Use local stats if not authenticated
      return;
    }
    
    // Make API request for stats
    const response = await fetch(`${API_ENDPOINT}/extension/stats?period=${period}`, {
      headers: {
        'Authorization': `Bearer ${authData.access_token}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    
    const stats = await response.json();
    
    // Update UI with stats data
    if (stats) {
      // Update statistics for the tab stats (with tab- prefix)
      const tabStatScanned = document.getElementById('tab-stat-scanned');
      const tabStatClean = document.getElementById('tab-stat-clean');
      const tabStatOffensive = document.getElementById('tab-stat-offensive');
      const tabStatHate = document.getElementById('tab-stat-hate');
      const tabStatSpam = document.getElementById('tab-stat-spam');
      
      if (tabStatScanned) tabStatScanned.textContent = stats.total_count || 0;
      if (tabStatClean) tabStatClean.textContent = stats.clean_count || 0;
      if (tabStatOffensive) tabStatOffensive.textContent = stats.offensive_count || 0;
      if (tabStatHate) tabStatHate.textContent = stats.hate_count || 0;
      if (tabStatSpam) tabStatSpam.textContent = stats.spam_count || 0;
      
      // Also update common stats if they exist
      if (statScanned) statScanned.textContent = stats.total_count || 0;
      if (statClean) statClean.textContent = stats.clean_count || 0;
      if (statOffensive) statOffensive.textContent = stats.offensive_count || 0;
      if (statHate) statHate.textContent = stats.hate_count || 0;
      if (statSpam) statSpam.textContent = stats.spam_count || 0;
    }
    
  } catch (error) {
    console.error('Error loading stats:', error);
    showNotification(error.message, 'error');
  }
}

/**
 * Save settings to chrome storage
 */
async function saveSettings() {
  const settings = {
    enabled: enableToggle ? enableToggle.checked : true,
    threshold: thresholdSlider ? parseFloat(thresholdSlider.value) : 0.7,
    highlightToxic: highlightToxic ? highlightToxic.checked : true,
    saveData: saveDataCheckbox ? saveDataCheckbox.checked : true,
    platforms: {
      facebook: platformFacebook ? platformFacebook.checked : true,
      youtube: platformYoutube ? platformYoutube.checked : true,
      twitter: platformTwitter ? platformTwitter.checked : true,
      tiktok: platformTiktok ? platformTiktok.checked : false,
      zalo: platformZalo ? platformZalo.checked : false
    },
    displayOptions: {
      showClean: showClean ? showClean.checked : true,
      showOffensive: showOffensive ? showOffensive.checked : true,
      showHate: showHate ? showHate.checked : true,
      showSpam: showSpam ? showSpam.checked : true
    },
    theme: getActiveTheme()
  };
  
  chrome.storage.sync.set(settings, async () => {
    notifyContentScripts(settings);
    showNotification('Cài đặt đã được lưu', 'success');
    const authData = await getAuthData();
    if (authData && authData.access_token) {
      await saveServerSettings(settings);
    }
  });
}

/**
 * Notify content scripts about settings changes
 */
function notifyContentScripts(settings) {
  try {
    chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
      if (tabs && tabs.length > 0 && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: "settingsUpdated",
          settings: settings
        }).catch(err => {
          console.log("Could not send settings update to content script: ", err);
        });
      }
    });
  } catch (err) {
    console.error("Error updating content script: ", err);
  }
}

/**
 * Set theme for the extension
 */
function setTheme(theme) {
  // Remove any active class from theme buttons
  if (themeLight) themeLight.classList.remove('active');
  if (themeDark) themeDark.classList.remove('active');
  if (themeAuto) themeAuto.classList.remove('active');
  
  // Add active class to selected theme button
  if (theme === 'light' && themeLight) {
    themeLight.classList.add('active');
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
  } else if (theme === 'dark' && themeDark) {
    themeDark.classList.add('active');
    document.body.classList.remove('light-mode');
    document.body.classList.add('dark-mode');
  } else if (theme === 'auto' && themeAuto) {
    themeAuto.classList.add('active');
    
    // Check system preference
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.body.classList.remove('light-mode');
      document.body.classList.add('dark-mode');
    } else {
      document.body.classList.remove('dark-mode');
      document.body.classList.add('light-mode');
    }
  }
  
  // Save theme setting
  chrome.storage.sync.set({ theme: theme });
}

/**
 * Get active theme
 */
function getActiveTheme() {
  if (themeLight && themeLight.classList.contains('active')) {
    return 'light';
  } else if (themeDark && themeDark.classList.contains('active')) {
    return 'dark';
  } else {
    return 'auto';
  }
}

/**
 * Analyze text using the API
 */
async function processAnalyzeText() {
  const textToAnalyze = analyzeText.value.trim();
  
  if (!textToAnalyze) {
    showNotification('Vui lòng nhập văn bản cần phân tích', 'error');
    return;
  }
  
  try {
    analyzeButton.disabled = true;
    analyzeButton.textContent = 'Đang phân tích...';
    
    // Hiển thị indicator loading nhỏ
    const loadingIndicator = document.createElement('div');
    loadingIndicator.id = 'analysis-loading-indicator';
    loadingIndicator.innerHTML = '<div class="spinner"></div> <span>Đang phân tích...</span>';
    loadingIndicator.style.padding = '10px';
    loadingIndicator.style.textAlign = 'center';
    loadingIndicator.style.color = '#0288d1';
    analyzeButton.after(loadingIndicator);
    
    // Hiển thị sẵn container kết quả
    const resultContainer = document.getElementById('result-container');
    if (resultContainer) {
      resultContainer.classList.remove('hidden');
      resultContainer.style.display = 'block'; // Force show
    }
    
    // Get platform from current tab
    let platform = 'unknown';
    if (currentPlatform && currentPlatform.textContent) {
      platform = currentPlatform.textContent.toLowerCase();
    }
    
    // Get authentication token
    const authData = await getAuthData();
    const token = authData ? authData.access_token : null;
    const isLoggedIn = !!token; // Check if logged in
    
    // Create request headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Add auth token if available
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    // Determine if data should be saved (only if logged in and setting is checked)
    const shouldSaveData = isLoggedIn && saveDataCheckbox && saveDataCheckbox.checked;
    
    // Get anonymization option
    const isAnonymous = anonymousMode ? anonymousMode.checked : false;
    
    // Get region context
    const region = regionSelect ? regionSelect.value : 'all';
    
    console.log('[DEBUG] API Request:', {
      text: textToAnalyze,
      platform: platform,
      save_to_db: shouldSaveData,
      metadata: {
        anonymous: isAnonymous,
        region: region,
        source: 'extension-popup'
      }
    });
    
    // Make API request
    const response = await fetch(`${API_ENDPOINT}/extension/detect`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        text: textToAnalyze,
        platform: platform,
        save_to_db: shouldSaveData, // Use calculated value
        metadata: {
          anonymous: isAnonymous,
          region: region,
          source: 'extension-popup'
        }
      })
    });
    
    if (!response.ok) {
      // Handle potential 401 if token is required but missing/invalid for some reason
      if (response.status === 401) {
         clearAuthData(); // Clear invalid token data
         checkAuthStatus(); // Re-run auth check to update UI
         throw new Error(`Yêu cầu xác thực. Vui lòng đăng nhập lại.`);
      }
      throw new Error(`Lỗi API: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('[DEBUG] API Response:', result);
    
    // Display analysis result
    displayAnalysisResult(result);
    
    // Update local statistics
    updateLocalStats(result.prediction);
    
  } catch (error) {
    console.error('Error analyzing text:', error);
    showNotification(error.message, 'error');
    
    // Hiển thị lỗi trong khu vực kết quả
    const resultContainer = document.getElementById('result-container');
    if (resultContainer) {
      resultContainer.classList.remove('hidden');
      resultContainer.style.display = 'block';
      resultContainer.innerHTML = `<div style="color:#d32f2f; padding: 15px;">
        <h3>Lỗi phân tích</h3>
        <p>${error.message}</p>
        <p>Vui lòng thử lại sau hoặc kiểm tra kết nối mạng.</p>
      </div>`;
    }
    
  } finally {
    analyzeButton.disabled = false;
    analyzeButton.textContent = 'Phân tích';
    
    // Xóa indicator loading
    const loadingIndicator = document.getElementById('analysis-loading-indicator');
    if (loadingIndicator) {
      loadingIndicator.remove();
    }
  }
}

/**
 * Display the analysis result in the UI
 */
function displayAnalysisResult(result) {
  console.log('[DEBUG] displayAnalysisResult input:', result);
  
  // Make sure we are using the single-mode result container
  const resultContainer = document.getElementById('result-container');
  const resultCategory = document.getElementById('result-category'); 
  const resultConfidence = document.getElementById('result-confidence');
  
  if (!resultContainer) {
    console.error('[ERROR] resultContainer not found');
    return;
  }
  
  // Kiểm tra dữ liệu đầu vào
  if (typeof result.prediction === 'undefined' || typeof result.confidence === 'undefined') {
    resultContainer.classList.remove('hidden');
    resultContainer.style.display = 'block'; // Force show for debug
    resultContainer.innerHTML = '<div style="color:#d32f2f;">Không có kết quả phân tích hợp lệ từ API.<br>Chi tiết: ' + JSON.stringify(result) + '</div>';
    console.log('[DEBUG] resultContainer.innerHTML (error):', resultContainer.innerHTML);
    return;
  }
  
  // Show result container
  resultContainer.classList.remove('hidden');
  resultContainer.style.display = 'block'; // Force show for debug
  
  // Render kết quả vào DOM elements từ popup.html
  // Cập nhật category và confidence
  const category = document.getElementById('result-category');
  const confidence = document.getElementById('result-confidence');
  
  if (category && confidence) {
    // Cập nhật category
    category.textContent = categoryNames[result.prediction] || 'Không xác định';
    // Xóa tất cả các CSS class cũ và thêm class mới
    category.className = 'result-value ' + (categoryClasses[result.prediction] || 'unknown');
    // Cập nhật confidence
    confidence.textContent = Math.round(result.confidence * 100) + '%';
  } else {
    console.error('[ERROR] result-category or result-confidence elements not found');
  }
  
  // Cập nhật các progress bar
  let p = result.probabilities;
  let clean = 0, offensive = 0, hate = 0, spam = 0;
  
  if (p) {
    if (typeof p === 'object' && !Array.isArray(p)) {
      clean = p.clean || 0;
      offensive = p.offensive || 0;
      hate = p.hate || 0;
      spam = p.spam || 0;
    } else if (Array.isArray(p)) {
      clean = p[0] || 0;
      offensive = p[1] || 0;
      hate = p[2] || 0;
      spam = p[3] || 0;
    }
  } else {
    clean = result.prediction === 0 ? result.confidence : 0.1;
    offensive = result.prediction === 1 ? result.confidence : 0.1;
    hate = result.prediction === 2 ? result.confidence : 0.1;
    spam = result.prediction === 3 ? result.confidence : 0.1;
  }
  
  // Update progress bars using the existing setProgressBar function
  setProgressBar(document.getElementById('progress-clean'), document.getElementById('percent-clean'), clean);
  setProgressBar(document.getElementById('progress-offensive'), document.getElementById('percent-offensive'), offensive);
  setProgressBar(document.getElementById('progress-hate'), document.getElementById('percent-hate'), hate);
  setProgressBar(document.getElementById('progress-spam'), document.getElementById('percent-spam'), spam);
  
  // Update advanced result section if available
  const resultAdvanced = document.getElementById('result-advanced');
  if (resultAdvanced) {
    resultAdvanced.classList.remove('hidden');
    
    const resultEmotion = document.getElementById('result-emotion');
    const resultIntent = document.getElementById('result-intent');
    const resultKeywords = document.getElementById('result-keywords');
    
    if (resultEmotion) {
      resultEmotion.textContent = result.sentiment && result.sentiment.label ? result.sentiment.label : 'Không xác định';
    }
    
    if (resultIntent) {
      resultIntent.textContent = result.intent || 'Không xác định';
    }
    
    if (resultKeywords) {
      resultKeywords.innerHTML = '';
      if (result.keywords && Array.isArray(result.keywords) && result.keywords.length > 0) {
        result.keywords.forEach(keyword => {
          const chip = document.createElement('span');
          chip.className = 'keyword-chip';
          chip.textContent = keyword;
          resultKeywords.appendChild(chip);
        });
      } else {
        resultKeywords.innerHTML = '<span style="color:#888">Không có từ khóa nổi bật</span>';
      }
    }
  }
  
  // Re-attach event listeners if needed
  const reportButton = document.getElementById('report-button');
  if (reportButton) {
    reportButton.onclick = reportIncorrectAnalysis;
  }
  
  // Log after UI update
  console.log('[DEBUG] Result displayed successfully');
  console.log('[DEBUG] resultContainer.classList:', resultContainer.classList.value, '| style.display:', resultContainer.style.display);
}

/**
 * Set a progress bar value and percentage text
 */
function setProgressBar(progressElement, percentElement, value) {
  if (!progressElement || !percentElement) return;
  
  const percent = Math.round(value * 100);
  progressElement.style.width = `${percent}%`;
  percentElement.textContent = `${percent}%`;
}

/**
 * Update local statistics based on prediction
 */
function updateLocalStats(prediction) {
  // Only update local stats for anonymous users
  const isLoggedIn = document.body.classList.contains('logged-in');
  if (isLoggedIn) return;

  chrome.storage.sync.get('stats', function(data) {
    const stats = data.stats || {
            scanned: 0,
            clean: 0,
            offensive: 0,
            hate: 0,
            spam: 0
          };
          
    // Increment scanned count
    stats.scanned += 1;
    
    // Increment specific category
    switch(prediction) {
      case 0:
        stats.clean += 1;
        break;
      case 1:
        stats.offensive += 1;
        break;
      case 2:
        stats.hate += 1;
        break;
      case 3:
        stats.spam += 1;
        break;
    }
    
    // Save updated stats
    chrome.storage.sync.set({ stats });
    
    // Update stats display for both overview and tab
    if (statScanned) statScanned.textContent = stats.scanned;
    if (statClean) statClean.textContent = stats.clean;
    if (statOffensive) statOffensive.textContent = stats.offensive;
    if (statHate) statHate.textContent = stats.hate;
    if (statSpam) statSpam.textContent = stats.spam;
    
    // Cập nhật cả thống kê trong tab (nếu có)
    const tabStatScanned = document.getElementById('tab-stat-scanned');
    const tabStatClean = document.getElementById('tab-stat-clean');
    const tabStatOffensive = document.getElementById('tab-stat-offensive');
    const tabStatHate = document.getElementById('tab-stat-hate');
    const tabStatSpam = document.getElementById('tab-stat-spam');
    
    if (tabStatScanned) tabStatScanned.textContent = stats.scanned;
    if (tabStatClean) tabStatClean.textContent = stats.clean;
    if (tabStatOffensive) tabStatOffensive.textContent = stats.offensive;
    if (tabStatHate) tabStatHate.textContent = stats.hate;
    if (tabStatSpam) tabStatSpam.textContent = stats.spam;
    
    // Cập nhật cả biểu đồ nếu đã tạo
    initStatsCharts(stats);
  });
}

/**
 * Report incorrect analysis to improve the model
 */
async function reportIncorrectAnalysis() {
  const textContent = analyzeText.value.trim();
  if (!textContent) return;
  
  try {
    // Get authentication token
    const authData = await getAuthData();
    const token = authData ? authData.access_token : null;
    
    // Create request headers
    const headers = {
      'Content-Type': 'application/json'
    };
    
    // Add auth token if available
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    // Make API request to feedback endpoint
    const response = await fetch(`${API_ENDPOINT}/feedback/report`, {
      method: 'POST',
      headers: headers,
      body: JSON.stringify({
        text: textContent,
        source: 'extension-popup',
        reported_by: authData ? authData.user_id : null
      })
    });
    
    if (!response.ok) {
      throw new Error(`Lỗi API: ${response.status}`);
    }
    
    showNotification('Đã gửi báo cáo phân tích sai. Cảm ơn bạn đã giúp cải thiện hệ thống!', 'success');
    
  } catch (error) {
    console.error('Error reporting analysis:', error);
    showNotification(error.message, 'error');
  }
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
  const notification = document.getElementById('notification');
  if (!notification) return;
  
  // Set notification text and type
  notification.textContent = message;
  notification.className = `notification ${type}`;
  
  // Set ARIA attributes for accessibility
  notification.setAttribute('role', 'alert');
  notification.setAttribute('aria-live', 'assertive');
  
  // Show notification
  notification.classList.remove('hidden');
  
  // Hide after 4 seconds
  setTimeout(() => {
    notification.classList.add('hidden');
    
    // Reset ARIA attributes after hiding
    setTimeout(() => {
      notification.setAttribute('aria-live', 'off');
    }, 500);
  }, 4000);
}

/**
 * Setup event handlers for all UI interactions
 */
function setupEventHandlers() {
  // Settings tab handlers
  if (enableToggle) {
    enableToggle.addEventListener('change', function() {
      chrome.storage.sync.set({ enabled: this.checked });
    });
  }
  
  if (thresholdSlider && thresholdValue) {
    thresholdSlider.addEventListener('input', function() {
      thresholdValue.textContent = this.value;
    });
    
    thresholdSlider.addEventListener('change', function() {
      saveSettings();
    });
  }
  
  if (highlightToxic) {
    highlightToxic.addEventListener('change', saveSettings);
  }
  
  if (saveDataCheckbox) {
    saveDataCheckbox.addEventListener('change', saveSettings);
  }
  
  // Platform checkboxes
  const platformCheckboxes = [
    platformFacebook, platformYoutube, platformTwitter, platformTiktok, platformZalo
  ];
  
  platformCheckboxes.forEach(checkbox => {
    if (checkbox) {
      checkbox.addEventListener('change', saveSettings);
    }
  });
  
  // Display options
  const displayOptionCheckboxes = [showClean, showOffensive, showHate, showSpam];
  
  displayOptionCheckboxes.forEach(checkbox => {
    if (checkbox) {
      checkbox.addEventListener('change', saveSettings);
    }
  });
  
  // Theme buttons
  if (themeLight) {
    themeLight.addEventListener('click', () => setTheme('light'));
  }
  
  if (themeDark) {
    themeDark.addEventListener('click', () => setTheme('dark'));
  }
  
  if (themeAuto) {
    themeAuto.addEventListener('click', () => setTheme('auto'));
  }
  
  // Reset stats button
  if (resetStatsBtn) {
    resetStatsBtn.addEventListener('click', resetStats);
  }
  
  // Analysis button for single text
  if (analyzeButton) {
    analyzeButton.addEventListener('click', processAnalyzeText);
  }
  
  // Analysis button for batch text is handled in setupBatchDetectUI
  
  // Report button - using event delegation because it might be dynamically created
  document.addEventListener('click', function(event) {
    if (event.target && event.target.id === 'report-button') {
      reportIncorrectAnalysis();
    }
  });
  
  // Period selector buttons
  const periodButtons = document.querySelectorAll('.period-button');
  if (periodButtons) {
    periodButtons.forEach(button => {
      button.addEventListener('click', () => {
        periodButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
        
        // Load stats for the selected period
        loadStatsForPeriod(button.getAttribute('data-period'));
      });
    });
  }
  
  // Authentication form handlers
  setupAuthHandlers();
}

/**
 * Setup authentication form handlers
 */
function setupAuthHandlers() {
  // Show register form
  if (showRegisterBtn) {
    showRegisterBtn.addEventListener('click', () => {
      showAuthSection('register');
    });
  }
  
  // Show login form
  if (showLoginBtn) {
    showLoginBtn.addEventListener('click', () => {
      showAuthSection('login');
    });
  }
  
  // Show forgot password form
  if (forgotPasswordBtn) {
    forgotPasswordBtn.addEventListener('click', (e) => {
      e.preventDefault();
      showAuthSection('reset-password');
    });
  }
  
  // Cancel reset password
  if (cancelResetBtn) {
    cancelResetBtn.addEventListener('click', () => {
      showAuthSection('login');
    });
  }
  
  // Cancel change password
  if (cancelChangePasswordBtn) {
    cancelChangePasswordBtn.addEventListener('click', () => {
      showAuthSection('profile');
    });
  }
  
  // Logout button
  if (logoutBtn) {
    logoutBtn.addEventListener('click', logout);
  }
  
  // Change password button
  if (changePasswordBtn) {
    changePasswordBtn.addEventListener('click', () => {
      showAuthSection('change-password');
    });
  }
  
  // Login form submission
  if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
      e.preventDefault();
      login();
    });
  }
  
  // Register form submission
  if (registerForm) {
    registerForm.addEventListener('submit', (e) => {
      e.preventDefault();
      register();
    });
  }
  
  // Reset password form submission
  if (resetPasswordForm) {
    resetPasswordForm.addEventListener('submit', (e) => {
      e.preventDefault();
      requestPasswordReset();
    });
  }
  
  // Change password form submission
  if (changePasswordForm) {
    changePasswordForm.addEventListener('submit', (e) => {
      e.preventDefault();
      changePassword();
    });
  }
}

/**
 * Show the specified authentication section and hide others
 */
function showAuthSection(section) {
  // Hide all sections
  const sections = [
    loginSection, registerSection, profileSection, 
    resetPasswordSection, changePasswordSection
  ];
  
  sections.forEach(section => {
    if (section) section.classList.add('hidden');
  });
  
  // Show requested section
  switch (section) {
    case 'login':
      if (loginSection) loginSection.classList.remove('hidden');
      break;
    case 'register':
      if (registerSection) registerSection.classList.remove('hidden');
      break;
    case 'profile':
      if (profileSection) profileSection.classList.remove('hidden');
      break;
    case 'reset-password':
      if (resetPasswordSection) resetPasswordSection.classList.remove('hidden');
      break;
    case 'change-password':
      if (changePasswordSection) changePasswordSection.classList.remove('hidden');
      break;
  }
}

/**
 * Check authentication status and update UI
 */
async function checkAuthStatus() {
  try {
    const authData = await getAuthData();
    
    if (authData && authData.access_token) {
      // Verify token is still valid by making API request
      const response = await fetch(`${API_ENDPOINT}/auth/me`, {
        headers: {
          'Authorization': `Bearer ${authData.access_token}`
        }
      });
      
      if (response.ok) {
        // Token is valid, show profile section
        document.body.classList.add('logged-in');
        document.body.classList.remove('not-logged-in');
        const userData = await response.json();
        updateProfileUI(userData);
        showAuthSection('profile');
        return true;
      } else {
        // Token expired or invalid, show login screen
        document.body.classList.remove('logged-in');
        document.body.classList.add('not-logged-in');
        showAuthSection('login');
        clearAuthData();
        return false;
      }
    } else {
      // No token, show login screen
      document.body.classList.remove('logged-in');
      document.body.classList.add('not-logged-in');
      showAuthSection('login');
      return false;
    }
  } catch (error) {
    console.error('Error checking auth status:', error);
    document.body.classList.remove('logged-in');
    document.body.classList.add('not-logged-in');
    showAuthSection('login');
    return false;
  }
}

/**
 * Apply simple UI updates based on authentication state
 */
function updateAuthDependentUI(isLoggedIn) {
  console.log("Cập nhật UI dựa trên đăng nhập:", isLoggedIn);
  
  // Cập nhật hiển thị tab
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  // Xử lý hiển thị từng tab dựa trên trạng thái đăng nhập
  tabButtons.forEach(button => {
    const tabName = button.getAttribute('data-tab');
    const isAuthRequired = button.classList.contains('requires-auth');
    
    if (isAuthRequired && !isLoggedIn) {
      // Ẩn tab yêu cầu đăng nhập nếu chưa đăng nhập
      button.style.display = 'none';
      
      // Ẩn nội dung tab tương ứng
      const tabPane = document.getElementById(`${tabName}-tab`);
      if (tabPane) {
        tabPane.style.display = 'none';
        tabPane.classList.remove('active');
      }
      
      // Nếu tab đang active bị ẩn đi, bỏ active
      if (button.classList.contains('active')) {
        button.classList.remove('active');
      }
    } else {
      // Hiển thị tab không yêu cầu đăng nhập hoặc đã đăng nhập
      button.style.display = '';
      
      // Hiển thị nội dung tab tương ứng
      const tabPane = document.getElementById(`${tabName}-tab`);
      if (tabPane) {
        tabPane.style.display = '';
      }
    }
  });
  
  // Tìm tab active hiện tại
  let activeTabButton = document.querySelector('.tab-button.active');
  let activePane = document.querySelector('.tab-pane.active');
  
  // Nếu không có tab nào active hoặc tab active không khả dụng
  if (!activeTabButton || activeTabButton.style.display === 'none') {
    // Bỏ active từ tất cả tab và pane
    tabButtons.forEach(btn => btn.classList.remove('active'));
    tabPanes.forEach(pane => pane.classList.remove('active'));
    
    // Tìm tab đầu tiên khả dụng để active
    const visibleTabs = Array.from(tabButtons).filter(btn => btn.style.display !== 'none');
    const firstVisibleTab = visibleTabs.length > 0 ? visibleTabs[0] : null;
    
    if (firstVisibleTab) {
      // Kích hoạt tab đầu tiên khả dụng
      firstVisibleTab.classList.add('active');
      const tabName = firstVisibleTab.getAttribute('data-tab');
      const tabPane = document.getElementById(`${tabName}-tab`);
      if (tabPane) {
        tabPane.classList.add('active');
        tabPane.style.display = '';
        
        // Nếu là tab stats và đã đăng nhập, load dữ liệu stats
        if (tabName === 'stats' && isLoggedIn) {
          setupStatsTab();
        }
      }
    }
  }
  
  // Chế độ phân tích (batch vs single)
  const batchModeBtn = document.getElementById('batch-mode-btn');
  const batchAnalyzeArea = document.getElementById('batch-analyze-area');
  const singleModeBtn = document.getElementById('single-mode-btn');
  const singleAnalyzeArea = document.getElementById('single-analyze-area');
  
  if (batchModeBtn) {
    // Hiển thị/ẩn nút chế độ batch
    batchModeBtn.style.display = isLoggedIn ? '' : 'none';
    
    if (!isLoggedIn && batchModeBtn.classList.contains('active')) {
      // Nếu đang ở chế độ batch mà đăng xuất, chuyển về single
      if (singleModeBtn) {
        singleModeBtn.classList.add('active');
        batchModeBtn.classList.remove('active');
      }
      
      if (singleAnalyzeArea) singleAnalyzeArea.style.display = '';
      if (batchAnalyzeArea) batchAnalyzeArea.style.display = 'none';
    }
  }
  
  // Checkbox "lưu dữ liệu" và các phần chỉ dành cho người dùng đã đăng nhập
  const saveDataCheckbox = document.getElementById('save-data');
  const saveDataLabel = document.querySelector('label[for="save-data"]');
  
  if (saveDataCheckbox) {
    saveDataCheckbox.disabled = !isLoggedIn;
    if (!isLoggedIn) saveDataCheckbox.checked = false;
    
    // Thêm tooltip/message cho người dùng chưa đăng nhập
    if (saveDataLabel) {
      if (!isLoggedIn) {
        saveDataLabel.setAttribute('data-tooltip', 'Yêu cầu đăng nhập');
      } else {
        saveDataLabel.removeAttribute('data-tooltip');
      }
    }
  }
  
  // Nút reset thống kê
  const resetStatsBtn = document.getElementById('reset-stats');
  if (resetStatsBtn) {
    resetStatsBtn.style.display = isLoggedIn ? '' : 'none';
  }
  
  console.log("Cập nhật UI hoàn tất - Tab active:", activeTabButton ? activeTabButton.getAttribute('data-tab') : 'không');
}

/**
 * Update the profile UI with user data
 */
function updateProfileUI(userData) {
  if (!profileUsername || !profileEmail || !profileName || !profileRole) return;
  
  profileUsername.textContent = userData.username || '';
  profileEmail.textContent = userData.email || '';
  profileName.textContent = userData.name || userData.username || '';
  profileRole.textContent = userData.role || 'user';
}

/**
 * Login using API
 */
async function login() {
  if (!loginForm) return;
  
  try {
    const username = document.getElementById('login-username').value;
    const password = document.getElementById('login-password').value;
    const rememberMe = document.getElementById('login-remember').checked;
    
    if (!username || !password) {
      showNotification('Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu', 'error');
      return;
    }
    
    // Create form data for token endpoint (API expects form data)
    const formData = new FormData();
    formData.append('username', username);
    formData.append('password', password);
    
    // Make API request
    const response = await fetch(`${API_ENDPOINT}/auth/token`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        throw new Error('Tên đăng nhập hoặc mật khẩu không đúng');
      }
      throw new Error(`Lỗi đăng nhập: ${response.status}`);
    }
    
    const authData = await response.json();
    
    // Save authentication data
    await saveAuthData(authData, rememberMe);
    
    // Update UI
    const userDataResponse = await fetch(`${API_ENDPOINT}/auth/me`, {
      headers: {
        'Authorization': `Bearer ${authData.access_token}`
      }
    });
    
    if (userDataResponse.ok) {
      // Get user data
      const userData = await userDataResponse.json();
      
      // Update profile and authentication UI
      document.body.classList.add('logged-in');
      document.body.classList.remove('not-logged-in');
      updateProfileUI(userData);
      showAuthSection('profile');
      
      // Refresh tabs to show auth-required tabs
      setupTabNavigation();
      updateAuthDependentUI(true);
      
      // Setup batch detect UI
      setupBatchDetectUI();
      setupAnalyzeModeSwitch();
      
      showNotification('Đăng nhập thành công', 'success');
    } else {
      throw new Error('Không thể lấy thông tin người dùng');
    }
  } catch (error) {
    console.error('Login error:', error);
    showNotification(error.message || 'Lỗi đăng nhập', 'error');
  }
}

/**
 * Register new user
 */
async function register() {
  if (!registerForm) return;
  
  try {
    const username = document.getElementById('register-username').value;
    const email = document.getElementById('register-email').value;
    const name = document.getElementById('register-name').value;
    const password = document.getElementById('register-password').value;
    const confirmPassword = document.getElementById('register-confirm-password').value;
    
    // Validate fields
    if (!username || !email || !password || !confirmPassword) {
      showNotification('Vui lòng điền đầy đủ thông tin', 'error');
      return;
    }
    
    if (password !== confirmPassword) {
      showNotification('Mật khẩu xác nhận không khớp', 'error');
      return;
    }
    
    // Make API request
    const response = await fetch(`${API_ENDPOINT}/auth/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        username,
        email,
        name,
        password
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Lỗi đăng ký');
    }
    
    showNotification('Đăng ký thành công! Vui lòng đăng nhập', 'success');
    showAuthSection('login');
    
    // Pre-fill login form
    if (document.getElementById('login-username')) {
      document.getElementById('login-username').value = username;
    }
  } catch (error) {
    console.error('Registration error:', error);
    showNotification(error.message || 'Lỗi đăng ký', 'error');
  }
}

/**
 * Request password reset
 */
async function requestPasswordReset() {
  if (!resetPasswordForm) return;
  
  try {
    const email = document.getElementById('reset-email').value;
    
    if (!email) {
      showNotification('Vui lòng nhập email', 'error');
      return;
    }
    
    // Make API request
    const response = await fetch(`${API_ENDPOINT}/auth/reset-password-request`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        email
      })
    });
    
    // Always show success even if email doesn't exist (security best practice)
    showNotification('Nếu email tồn tại, bạn sẽ nhận được hướng dẫn khôi phục mật khẩu', 'success');
    showAuthSection('login');
  } catch (error) {
    // Still show success message even on error
    showNotification('Nếu email tồn tại, bạn sẽ nhận được hướng dẫn khôi phục mật khẩu', 'success');
    showAuthSection('login');
  }
}

/**
 * Change password
 */
async function changePassword() {
  if (!changePasswordForm) return;
  
  try {
    const currentPassword = document.getElementById('current-password').value;
    const newPassword = document.getElementById('new-password').value;
    const confirmNewPassword = document.getElementById('confirm-new-password').value;
    
    // Validate fields
    if (!currentPassword || !newPassword || !confirmNewPassword) {
      showNotification('Vui lòng điền đầy đủ thông tin', 'error');
      return;
    }
    
    if (newPassword !== confirmNewPassword) {
      showNotification('Mật khẩu mới xác nhận không khớp', 'error');
      return;
    }
    
    // Get authentication token
    const authData = await getAuthData();
    
    if (!authData || !authData.access_token) {
      showNotification('Bạn cần đăng nhập lại', 'error');
      showAuthSection('login');
      return;
    }
    
    // Make API request
    const response = await fetch(`${API_ENDPOINT}/auth/change-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authData.access_token}`
      },
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Lỗi đổi mật khẩu');
    }
    
    showNotification('Đổi mật khẩu thành công', 'success');
    showAuthSection('profile');
    
    // Clear password fields
    document.getElementById('current-password').value = '';
    document.getElementById('new-password').value = '';
    document.getElementById('confirm-new-password').value = '';
  } catch (error) {
    console.error('Change password error:', error);
    showNotification(error.message || 'Lỗi đổi mật khẩu', 'error');
  }
}

/**
 * Logout the user
 */
async function logout() {
  try {
    // Get authentication token
    const authData = await getAuthData();
    
    if (authData && authData.access_token) {
      // Call logout API
      await fetch(`${API_ENDPOINT}/auth/logout`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${authData.access_token}`
        }
      }).catch(() => {
        // Ignore errors from logout API
      });
    }
    
    // Clear local authentication data
    clearAuthData();
    
    // Update UI for logged-out state
    document.body.classList.remove('logged-in');
    document.body.classList.add('not-logged-in');
    
    // Show login section
    showAuthSection('login');
    
    // Show notification
    showNotification('Đăng xuất thành công', 'success');
    
    // Switch to analyze tab
    const analyzeTabButton = document.querySelector('.tab-button[data-tab="analyze"]');
    if (analyzeTabButton) {
      analyzeTabButton.click();
    }
    
    // Ensure single mode is active
    const singleModeBtn = document.getElementById('single-mode-btn');
    if (singleModeBtn) {
      singleModeBtn.click();
    }
  } catch (error) {
    console.error('Logout error:', error);
    
    // Force logout UI update on error
    document.body.classList.remove('logged-in');
    document.body.classList.add('not-logged-in');
    showAuthSection('login');
  }
}

/**
 * Save authentication data to storage
 */
async function saveAuthData(authData, persistent = false) {
  return new Promise((resolve) => {
    const storage = persistent ? chrome.storage.sync : chrome.storage.local;
    storage.set({ [AUTH_STORAGE_KEY]: authData }, () => {
      resolve();
    });
  });
}

/**
 * Get authentication data from storage
 */
async function getAuthData() {
  return new Promise((resolve) => {
    // First try local storage
    chrome.storage.local.get([AUTH_STORAGE_KEY], (localData) => {
      if (localData && localData[AUTH_STORAGE_KEY]) {
        resolve(localData[AUTH_STORAGE_KEY]);
      } else {
        // If not in local, try sync storage
        chrome.storage.sync.get([AUTH_STORAGE_KEY], (syncData) => {
          resolve(syncData && syncData[AUTH_STORAGE_KEY] ? syncData[AUTH_STORAGE_KEY] : null);
        });
      }
    });
  });
}

/**
 * Clear authentication data from storage
 */
function clearAuthData() {
  chrome.storage.local.remove([AUTH_STORAGE_KEY]);
  chrome.storage.sync.remove([AUTH_STORAGE_KEY]);
}

/**
 * Setup the analyze mode switch between single and batch modes
 */
function setupAnalyzeModeSwitch() {
  // Check if required elements exist
  if (!singleModeBtn || !batchModeBtn || !singleAnalyzeArea || !batchAnalyzeArea) {
    console.warn('Analyze mode switch elements not found');
    return;
  }
  
  // Set up event handlers for switching modes
  singleModeBtn.addEventListener('click', function() {
    // Activate single mode
    singleModeBtn.classList.add('active');
    batchModeBtn.classList.remove('active');
    
    // Update ARIA states
    singleModeBtn.setAttribute('aria-pressed', 'true');
    batchModeBtn.setAttribute('aria-pressed', 'false');
    
    // Show/hide appropriate areas
    singleAnalyzeArea.style.display = 'block';
    batchAnalyzeArea.style.display = 'none';
  });
  
  batchModeBtn.addEventListener('click', function() {
    // Activate batch mode
    batchModeBtn.classList.add('active');
    singleModeBtn.classList.remove('active');
    
    // Update ARIA states
    batchModeBtn.setAttribute('aria-pressed', 'true');
    singleModeBtn.setAttribute('aria-pressed', 'false');
    
    // Show/hide appropriate areas
    batchAnalyzeArea.style.display = 'block';
    singleAnalyzeArea.style.display = 'none';
  });
  
  // Initialize with the current state
  if (singleModeBtn.classList.contains('active')) {
    singleAnalyzeArea.style.display = 'block';
    batchAnalyzeArea.style.display = 'none';
  } else if (batchModeBtn.classList.contains('active')) {
    batchAnalyzeArea.style.display = 'block';
    singleAnalyzeArea.style.display = 'none';
  }
}

/**
 * Setup Batch Detect UI elements and functionality
 */
function setupBatchDetectUI() {
  // Verify batch UI elements exist
  if (!batchInput || !batchDetectBtn || !batchResultContainer || !batchResultList) {
    console.warn('Batch UI elements not found');
    return;
  }

  // Add click event listener for the batch detect button
  batchDetectBtn.addEventListener('click', processBatchText);

  // Setup file input handling for batch processing
  if (batchFileInput) {
    batchFileInput.addEventListener('change', function(e) {
      if (!this.files || this.files.length === 0) return;
      
      const file = this.files[0];
      if (file.type !== 'text/plain' && file.type !== 'text/csv' && 
          !file.name.endsWith('.txt') && !file.name.endsWith('.csv')) {
        showNotification('Chỉ chấp nhận file .txt hoặc .csv', 'error');
        return;
      }
      
      const reader = new FileReader();
      reader.onload = function(event) {
        batchInput.value = event.target.result;
      };
      reader.onerror = function() {
        showNotification('Không thể đọc file', 'error');
      };
      reader.readAsText(file);
    });
  }

  // Clear batch results when input changes
  if (batchInput) {
    batchInput.addEventListener('input', function() {
      if (batchResultContainer) {
        batchResultContainer.classList.add('hidden');
      }
    });
  }
}

/**
 * Setup Statistics tab - simplified version without charts
 */
function setupStatsTab() {
  // Just load local stats when the tab is viewed
  try {
    // Get stats from chrome.storage
    chrome.storage.sync.get('stats', (data) => {
      const stats = data.stats || {
        scanned: 0,
        clean: 0,
        offensive: 0,
        hate: 0,
        spam: 0
      };
      
      // Update the UI with the stats
      const tabStatScanned = document.getElementById('tab-stat-scanned');
      const tabStatClean = document.getElementById('tab-stat-clean');
      const tabStatOffensive = document.getElementById('tab-stat-offensive');
      const tabStatHate = document.getElementById('tab-stat-hate');
      const tabStatSpam = document.getElementById('tab-stat-spam');
      
      if (tabStatScanned) tabStatScanned.textContent = stats.scanned || 0;
      if (tabStatClean) tabStatClean.textContent = stats.clean || 0;
      if (tabStatOffensive) tabStatOffensive.textContent = stats.offensive || 0;
      if (tabStatHate) tabStatHate.textContent = stats.hate || 0;
      if (tabStatSpam) tabStatSpam.textContent = stats.spam || 0;
    });
  } catch (error) {
    console.error('Error in setupStatsTab:', error);
  }
}

/**
 * Setup enhanced UI interactions for better UX 
 */
function setupEnhancedUIInteractions() {
  // Add ripple effect to buttons
  document.querySelectorAll('.btn, .action-button, .primary-button, .secondary-button, .warning-button').forEach(button => {
    button.addEventListener('click', createRippleEffect);
  });
  
  // Setup tooltips behavior for elements with data-tooltip attributes
  setupTooltips();
  
  // Setup responsive form behavior (floating labels, etc)
  setupResponsiveForms();
  
  // Enhance tab indicator animation
  setupTabIndicator();
  
  // Setup form validation with proper error states
  setupFormValidation();
  
  // Setup progess bar animations
  setupProgressAnimations();
}

/**
 * Create ripple effect on button click
 */
function createRippleEffect(e) {
  // Remove any existing ripple elements
  const ripples = this.getElementsByClassName('ripple');
  
  while (ripples.length > 0) {
    ripples[0].parentNode.removeChild(ripples[0]);
  }
  
  // Create new ripple element
  const ripple = document.createElement('span');
  ripple.classList.add('ripple');
  this.appendChild(ripple);
  
  // Position the ripple where the user clicked
  const rect = this.getBoundingClientRect();
  const size = Math.max(rect.width, rect.height);
  
  ripple.style.width = ripple.style.height = size + 'px';
  ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
  ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
  
  // Remove ripple after animation completes
  ripple.addEventListener('animationend', function() {
    if (this.parentNode) {
      this.parentNode.removeChild(this);
    }
  });
}

/**
 * Setup tooltips behavior
 */
function setupTooltips() {
  document.querySelectorAll('[data-tooltip]').forEach(element => {
    // Add keyboard focus support for tooltip display
    element.addEventListener('focus', function() {
      const tooltipText = this.getAttribute('data-tooltip');
      this.setAttribute('aria-label', tooltipText);
    });
    
    // Ensure tooltip is also accessible to screen readers
    if (!element.getAttribute('aria-label')) {
      const tooltipText = element.getAttribute('data-tooltip');
      element.setAttribute('aria-label', tooltipText);
    }
  });
}

/**
 * Setup responsive form behaviors
 */
function setupResponsiveForms() {
  // Setup floating labels effect
  document.querySelectorAll('.form-group.floating').forEach(group => {
    const input = group.querySelector('input');
    const label = group.querySelector('label');
    
    if (input && label) {
      // Check initial state
      if (input.value) {
        label.classList.add('active');
      }
      
      // Add event listeners
      input.addEventListener('focus', () => {
        label.classList.add('active');
      });
      
      input.addEventListener('blur', () => {
        if (!input.value) {
          label.classList.remove('active');
        }
      });
    }
  });
  
  // Setup password visibility toggle
  document.querySelectorAll('.password-toggle').forEach(toggle => {
    toggle.addEventListener('click', function() {
      const input = this.previousElementSibling;
      if (input && input.type === 'password') {
        input.type = 'text';
        this.textContent = '🙈';
        this.setAttribute('aria-label', 'Ẩn mật khẩu');
      } else if (input) {
        input.type = 'password';
        this.textContent = '👁️';
        this.setAttribute('aria-label', 'Hiện mật khẩu');
      }
    });
  });
  
  // Setup password strength meters
  document.querySelectorAll('input[type="password"]').forEach(input => {
    input.addEventListener('input', function() {
      // Find the nearest password strength bar
      const strengthBar = this.parentNode.querySelector('.password-strength-bar');
      if (strengthBar) {
        updatePasswordStrength(this.value, strengthBar);
      }
    });
  });
}

/**
 * Update password strength meter
 */
function updatePasswordStrength(password, strengthBar) {
  // Simple password strength calculation
  let strength = 0;
  
  if (password.length >= 8) strength += 1;
  if (password.match(/[A-Z]/)) strength += 1;
  if (password.match(/[a-z]/)) strength += 1;
  if (password.match(/[0-9]/)) strength += 1;
  if (password.match(/[^A-Za-z0-9]/)) strength += 1;
  
  // Update strength bar
  strengthBar.className = 'password-strength-bar';
  
  if (strength <= 2) {
    strengthBar.classList.add('weak');
  } else if (strength <= 3) {
    strengthBar.classList.add('medium');
  } else {
    strengthBar.classList.add('strong');
  }
}

/**
 * Setup tab indicator animation
 */
function setupTabIndicator() {
  const tabHeader = document.querySelector('.tab-header');
  if (!tabHeader) return;
  
  // Create tab indicator element if it doesn't exist
  let tabIndicator = tabHeader.querySelector('.tab-indicator');
  if (!tabIndicator) {
    tabIndicator = document.createElement('div');
    tabIndicator.classList.add('tab-indicator');
    tabHeader.appendChild(tabIndicator);
  }
  
  // Function to position the indicator
  function positionIndicator(tab) {
    const tabRect = tab.getBoundingClientRect();
    const headerRect = tabHeader.getBoundingClientRect();
    
    tabIndicator.style.left = (tabRect.left - headerRect.left) + 'px';
    tabIndicator.style.width = tabRect.width + 'px';
  }
  
  // Update indicator position when tab is clicked
  tabHeader.querySelectorAll('.tab-button').forEach(button => {
    button.addEventListener('click', function() {
      positionIndicator(this);
    });
  });
  
  // Position indicator on initial load
  const activeTab = tabHeader.querySelector('.tab-button.active');
  if (activeTab) {
    setTimeout(() => positionIndicator(activeTab), 100);
  }
}

/**
 * Setup form validation with error states
 */
function setupFormValidation() {
  const forms = document.querySelectorAll('form');
  
  forms.forEach(form => {
    const inputs = form.querySelectorAll('input:required, textarea:required');
    
    inputs.forEach(input => {
      input.addEventListener('blur', function() {
        validateInput(this);
      });
    });
    
    form.addEventListener('submit', function(e) {
      let isValid = true;
      
      inputs.forEach(input => {
        if (!validateInput(input)) {
          isValid = false;
        }
      });
      
      if (!isValid) {
        e.preventDefault();
      }
    });
  });
}

/**
 * Validate a single input
 */
function validateInput(input) {
  const formGroup = input.closest('.form-group');
  if (!formGroup) return true;
  
  let errorMessage = '';
  
  if (!input.value) {
    errorMessage = 'Trường này là bắt buộc';
  } else if (input.type === 'email' && !validateEmail(input.value)) {
    errorMessage = 'Email không hợp lệ';
  }
  
  if (errorMessage) {
    formGroup.classList.add('error');
    input.setAttribute('aria-invalid', 'true');
    
    // Add or update error message
    let errorEl = formGroup.querySelector('.error-message');
    if (!errorEl) {
      errorEl = document.createElement('span');
      errorEl.classList.add('error-message');
      formGroup.appendChild(errorEl);
    }
    errorEl.textContent = errorMessage;
    
    return false;
  } else {
    formGroup.classList.remove('error');
    input.setAttribute('aria-invalid', 'false');
    
    // Remove error message if exists
    const errorEl = formGroup.querySelector('.error-message');
    if (errorEl) {
      formGroup.removeChild(errorEl);
    }
    
    return true;
  }
}

/**
 * Validate email format
 */
function validateEmail(email) {
  const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return re.test(email);
}

/**
 * Setup progress bar animations for results
 */
function setupProgressAnimations() {
  // No immediate actions needed here - the animation is applied
  // in the updateResults function when results are shown
}

/**
 * Enhanced version of updateResults to include animations
 */
function updateResults(data) {
  // Existing results update code...
  
  // Example of what should be added to the existing updateResults function:
  // Enable animations for the progress bars by first setting to 0 then animating to final value
  progressClean.style.width = '0%';
  progressOffensive.style.width = '0%';
  progressHate.style.width = '0%';
  progressSpam.style.width = '0%';
  
  // Use setTimeout to trigger CSS transitions
  setTimeout(() => {
    progressClean.style.width = data.probabilities.clean + '%';
    progressOffensive.style.width = data.probabilities.offensive + '%';
    progressHate.style.width = data.probabilities.hate + '%';
    progressSpam.style.width = data.probabilities.spam + '%';
    
    // Update ARIA for accessibility
    progressClean.parentElement.setAttribute('aria-valuenow', data.probabilities.clean);
    progressOffensive.parentElement.setAttribute('aria-valuenow', data.probabilities.offensive);
    progressHate.parentElement.setAttribute('aria-valuenow', data.probabilities.hate);
    progressSpam.parentElement.setAttribute('aria-valuenow', data.probabilities.spam);
  }, 50);
}

/**
 * Setup accessibility features
 */
function setupAccessibility() {
  // Skip link functionality
  const skipLink = document.querySelector('.skip-link');
  if (skipLink) {
    skipLink.addEventListener('click', function(e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute('href'));
      if (target) {
        target.setAttribute('tabindex', '-1');
        target.focus();
      }
    });
  }
  
  // Add appropriate ARIA attributes to tabs
  setupTabsAccessibility();
  
  // Ensure form inputs have associated labels
  checkFormAccessibility();
}

/**
 * Setup accessibility for tab navigation
 */
function setupTabsAccessibility() {
  const tabButtons = document.querySelectorAll('.tab-button');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  // Set up the tablist role
  const tabHeader = document.querySelector('.tab-header');
  if (tabHeader) {
    tabHeader.setAttribute('role', 'tablist');
  }
  
  tabButtons.forEach((button, index) => {
    // Set tab attributes
    button.setAttribute('role', 'tab');
    button.setAttribute('id', `tab-${button.getAttribute('data-tab')}`);
    button.setAttribute('aria-controls', `${button.getAttribute('data-tab')}-tab`);
    
    // Update accessibility state on click
    button.addEventListener('click', function() {
      tabButtons.forEach(btn => {
        btn.setAttribute('aria-selected', 'false');
        btn.setAttribute('tabindex', '-1');
      });
      
      this.setAttribute('aria-selected', 'true');
      this.setAttribute('tabindex', '0');
    });
    
    // Add keyboard navigation
    button.addEventListener('keydown', function(e) {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        
        const tabs = Array.from(tabButtons);
        const currentIndex = tabs.indexOf(this);
        
        let nextIndex;
        if (e.key === 'ArrowLeft') {
          nextIndex = currentIndex === 0 ? tabs.length - 1 : currentIndex - 1;
        } else {
          nextIndex = currentIndex === tabs.length - 1 ? 0 : currentIndex + 1;
        }
        
        tabs[nextIndex].click();
        tabs[nextIndex].focus();
      }
    });
  });
  
  // Set tab pane attributes
  tabPanes.forEach(pane => {
    const id = pane.id;
    const tabId = id.replace('-tab', '');
    const associatedButton = document.querySelector(`[data-tab="${tabId}"]`);
    
    pane.setAttribute('role', 'tabpanel');
    pane.setAttribute('aria-labelledby', `tab-${tabId}`);
    
    // Hide inactive panes from screen readers
    if (!pane.classList.contains('active')) {
      pane.setAttribute('hidden', '');
    }
  });
}

/**
 * Check form elements for accessibility issues
 */
function checkFormAccessibility() {
  // Ensure all inputs have associated labels
  document.querySelectorAll('input, textarea, select').forEach(input => {
    const id = input.id;
    if (!id) return;
    
    const label = document.querySelector(`label[for="${id}"]`);
    if (!label && !input.hasAttribute('aria-label')) {
      console.warn(`Input #${id} has no associated label or aria-label`);
    }
  });
}

/**
 * Enhanced show notification message with screen reader support
 */
function showNotification(message, type = 'info') {
  const notification = document.getElementById('notification');
  if (!notification) return;
  
  // Set notification text and type
  notification.textContent = message;
  notification.className = `notification ${type}`;
  
  // Set ARIA attributes for accessibility
  notification.setAttribute('role', 'alert');
  notification.setAttribute('aria-live', 'assertive');
  
  // Show notification
  notification.classList.remove('hidden');
  
  // Hide after 4 seconds
  setTimeout(() => {
    notification.classList.add('hidden');
    
    // Reset ARIA attributes after hiding
    setTimeout(() => {
      notification.setAttribute('aria-live', 'off');
    }, 500);
  }, 4000);
}

/**
 * Reset user statistics
 */
async function resetStats() {
  if (!confirm("Bạn có chắc muốn đặt lại tất cả thống kê không?")) {
    return;
  }

  try {
    // Get authentication token
    const authData = await getAuthData();
    const token = authData ? authData.access_token : null;
    
    if (!token) {
      showNotification('Vui lòng đăng nhập để đặt lại thống kê', 'warning');
      return;
    }
    
    // Nếu là người dùng đăng nhập, gọi API để reset thống kê
    if (document.body.classList.contains('logged-in')) {
      // Make API request to reset stats endpoint
      const response = await fetch(`${API_ENDPOINT}/extension/stats/reset`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });
      
      if (!response.ok) {
        throw new Error(`Lỗi API: ${response.status}`);
      }
      
      showNotification('Đã đặt lại thống kê thành công', 'success');
      
      // Reload stats
      loadStats();
    } 
    // Nếu chưa đăng nhập, reset thống kê local
    else {
      chrome.storage.sync.set({
        stats: {
          scanned: 0,
          clean: 0,
          offensive: 0,
          hate: 0,
          spam: 0
        }
      }, () => {
        showNotification('Đã đặt lại thống kê thành công', 'success');
        loadStats(); // Tải lại thống kê
      });
    }
  } catch (error) {
    console.error('Error resetting stats:', error);
    showNotification(error.message, 'error');
  }
}

/**
 * Process batch text for analysis
 */
async function processBatchText() {
  // Get the batch input text
  const batchInputText = batchInput.value.trim();
  
  if (!batchInputText) {
    showNotification('Vui lòng nhập văn bản cần phân tích', 'warning');
    return;
  }
  
  try {
    // Show loading indicator - save original text to restore later
    const originalButtonText = batchDetectBtn.innerHTML;
    batchDetectBtn.disabled = true;
    batchDetectBtn.innerHTML = '<span class="spinner"></span> Đang xử lý...';
    
    // Split text by newlines to get individual comments
    const allComments = batchInputText.split('\n')
      .map(text => text.trim())
      .filter(text => text.length > 0);
    
    if (allComments.length === 0) {
      showNotification('Không tìm thấy bình luận hợp lệ', 'warning');
      return;
    }
    
    // Determine authentication state
    const authData = await getAuthData();
    const isLoggedIn = !!authData;
    
    // Set headers based on authentication
    const headers = {
      'Content-Type': 'application/json',
    };
    
    if (isLoggedIn) {
      headers['Authorization'] = `Bearer ${authData.access_token}`;
    }
    
    // Get current platform from the currentPlatform element
    let platform = 'unknown';
    if (currentPlatform && currentPlatform.textContent) {
      platform = currentPlatform.textContent.toLowerCase();
    }
    
    // Create a progress indicator in the batch result container
    if (batchResultContainer) {
      batchResultContainer.classList.remove('hidden');
      
      if (batchResultList) {
        batchResultList.innerHTML = `
          <div class="batch-processing-progress">
            <div class="spinner"></div>
            <div id="batch-progress-text">Đang chuẩn bị phân tích ${allComments.length} dòng...</div>
            <div class="progress-bar">
              <div id="batch-progress-fill" class="progress-fill" style="width: 0%"></div>
            </div>
          </div>
        `;
      }
    }
    
    // Determine if we need to process in chunks
    const CHUNK_SIZE = 100;
    const totalChunks = Math.ceil(allComments.length / CHUNK_SIZE);
    const needsChunking = totalChunks > 1;
    
    // Initialize results array to store all processing results
    let allResults = [];
    
    // Process all chunks
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      // Update progress indicator
      const progressPercent = Math.round((chunkIndex / totalChunks) * 100);
      const progressFill = document.getElementById('batch-progress-fill');
      const progressText = document.getElementById('batch-progress-text');
      
      if (progressFill) {
        progressFill.style.width = `${progressPercent}%`;
      }
      
      if (progressText) {
        progressText.textContent = `Đang xử lý ${chunkIndex + 1}/${totalChunks} (${progressPercent}%)...`;
      }
      
      // Get current chunk of comments
      const startIndex = chunkIndex * CHUNK_SIZE;
      const endIndex = Math.min(startIndex + CHUNK_SIZE, allComments.length);
      const currentChunk = allComments.slice(startIndex, endIndex);
      
      // Prepare items for batch processing
      const items = currentChunk.map(text => ({
        text: text,
        platform: platform
      }));
      
      try {
        // Make API request for this chunk
        const response = await fetch(`${API_ENDPOINT}/extension/batch-detect`, {
          method: 'POST',
          headers: headers,
          body: JSON.stringify({
            items: items,
            save_to_db: isLoggedIn, // Only save if logged in
            store_clean: false // Don't store clean comments
          })
        });
        
        if (!response.ok) {
          // Handle potential 401 if token is required but missing/invalid
          if (response.status === 401) {
            clearAuthData(); // Clear invalid token data
            checkAuthStatus(); // Re-run auth check to update UI
            throw new Error(`Yêu cầu xác thực. Vui lòng đăng nhập lại.`);
          }
          throw new Error(`Lỗi API: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Add the chunk results to our combined results
        if (result.results && Array.isArray(result.results)) {
          allResults = allResults.concat(result.results);
          
          // Update local stats for this chunk's results
          result.results.forEach(item => {
            if (item.prediction !== undefined) {
              updateLocalStats(item.prediction);
            }
          });
        }
      } catch (error) {
        console.error(`Error processing chunk ${chunkIndex + 1}:`, error);
        // Continue with other chunks even if one fails
      }
      
      // Add a small delay between chunks to prevent overwhelming the server
      if (chunkIndex < totalChunks - 1) {
        await new Promise(resolve => setTimeout(resolve, 300));
      }
    }
    
    // Create a combined result object
    const combinedResult = {
      results: allResults,
      count: allResults.length
    };
    
    // Display all results together
    displayBatchResults(combinedResult);
    
    // Show message about completion
    if (needsChunking) {
      showNotification(`Đã hoàn thành phân tích ${allResults.length} dòng theo ${totalChunks} đợt.`, 'success');
    }
    
  } catch (error) {
    console.error('Batch processing error:', error);
    showNotification(error.message, 'error');
  } finally {
    // Reset button state - restore original styling and text
    batchDetectBtn.disabled = false;
    batchDetectBtn.innerHTML = '<span class="icon">⚡</span> Phân tích hàng loạt';
  }
}

/**
 * Display batch results in the UI
 */
function displayBatchResults(batchData) {
  if (!batchResultContainer || !batchResultList) return;
  
  // Clear previous results
  batchResultList.innerHTML = '';
  
  // Show the results container
  batchResultContainer.classList.remove('hidden');
  
  // Get the results array
  const results = batchData.results || [];
  
  // Update summary counts
  const batchTotal = document.getElementById('batch-total');
  const batchCleanCount = document.getElementById('batch-clean-count');
  const batchOffensiveCount = document.getElementById('batch-offensive-count');
  const batchHateCount = document.getElementById('batch-hate-count');
  const batchSpamCount = document.getElementById('batch-spam-count');
  
  // Count by category
  const counts = {
    clean: 0,
    offensive: 0,
    hate: 0,
    spam: 0
  };
  
  // Generate result items
  results.forEach(result => {
    // Update counts
    switch(result.prediction) {
      case 0: counts.clean++; break;
      case 1: counts.offensive++; break;
      case 2: counts.hate++; break;
      case 3: counts.spam++; break;
    }
    
    // Create result item element
    const resultItem = document.createElement('div');
    resultItem.className = 'batch-result-item';
    
    // Get category text
    const categoryText = result.prediction_text || 
      ['clean', 'offensive', 'hate', 'spam'][result.prediction] || 'unknown';
    
    // Create HTML for the item
    resultItem.innerHTML = `
      <div class="result-text">${result.text}</div>
      <div class="result-meta">
        <span class="label ${categoryText}">${categoryText}</span>
      </div>
    `;
    
    // Add to the list
    batchResultList.appendChild(resultItem);
  });
  
  // Update summary counts in UI
  if (batchCleanCount) batchCleanCount.textContent = counts.clean;
  if (batchOffensiveCount) batchOffensiveCount.textContent = counts.offensive;
  if (batchHateCount) batchHateCount.textContent = counts.hate;
  if (batchSpamCount) batchSpamCount.textContent = counts.spam;
  
  // Scroll to results
  batchResultContainer.scrollIntoView({ behavior: 'smooth' });
}

/**
 * Load statistics from storage
 */
function loadStats() {
  // Load local stats from chrome.storage
  chrome.storage.sync.get('stats', function(data) {
    if (data.stats) {
      // Update common stats
      if (statScanned) statScanned.textContent = data.stats.scanned || 0;
      if (statClean) statClean.textContent = data.stats.clean || 0;
      if (statOffensive) statOffensive.textContent = data.stats.offensive || 0;
      if (statHate) statHate.textContent = data.stats.hate || 0;
      if (statSpam) statSpam.textContent = data.stats.spam || 0;
      
      // Update tab stats
      const tabStatScanned = document.getElementById('tab-stat-scanned');
      const tabStatClean = document.getElementById('tab-stat-clean');
      const tabStatOffensive = document.getElementById('tab-stat-offensive');
      const tabStatHate = document.getElementById('tab-stat-hate');
      const tabStatSpam = document.getElementById('tab-stat-spam');
      
      if (tabStatScanned) tabStatScanned.textContent = data.stats.scanned || 0;
      if (tabStatClean) tabStatClean.textContent = data.stats.clean || 0;
      if (tabStatOffensive) tabStatOffensive.textContent = data.stats.offensive || 0;
      if (tabStatHate) tabStatHate.textContent = data.stats.hate || 0;
      if (tabStatSpam) tabStatSpam.textContent = data.stats.spam || 0;
    } else {
      // Initialize with zeros if no local stats found
      if (statScanned) statScanned.textContent = 0;
      if (statClean) statClean.textContent = 0;
      if (statOffensive) statOffensive.textContent = 0;
      if (statHate) statHate.textContent = 0;
      if (statSpam) statSpam.textContent = 0;
       
      // Update tab stats
      const tabStatScanned = document.getElementById('tab-stat-scanned');
      const tabStatClean = document.getElementById('tab-stat-clean');
      const tabStatOffensive = document.getElementById('tab-stat-offensive');
      const tabStatHate = document.getElementById('tab-stat-hate');
      const tabStatSpam = document.getElementById('tab-stat-spam');
       
      if (tabStatScanned) tabStatScanned.textContent = 0;
      if (tabStatClean) tabStatClean.textContent = 0;
      if (tabStatOffensive) tabStatOffensive.textContent = 0;
      if (tabStatHate) tabStatHate.textContent = 0;
      if (tabStatSpam) tabStatSpam.textContent = 0;
    }
  });
}

/**
 * Initialize statistics without charts - simplified version
 */
function initStatsCharts(stats) {
  // Just update the statistics values without creating charts
  console.log("Statistics updated without using charts:", stats);
  
  // Update the stats in the UI - using the same interface but no charts
  const tabStatScanned = document.getElementById('tab-stat-scanned');
  const tabStatClean = document.getElementById('tab-stat-clean');
  const tabStatOffensive = document.getElementById('tab-stat-offensive');
  const tabStatHate = document.getElementById('tab-stat-hate');
  const tabStatSpam = document.getElementById('tab-stat-spam');
  
  if (tabStatScanned) tabStatScanned.textContent = stats.scanned || stats.total || 0;
  if (tabStatClean) tabStatClean.textContent = stats.clean || 0;
  if (tabStatOffensive) tabStatOffensive.textContent = stats.offensive || 0;
  if (tabStatHate) tabStatHate.textContent = stats.hate || 0;
  if (tabStatSpam) tabStatSpam.textContent = stats.spam || 0;
}

/**
 * Placeholder function for updateTrendChart - for compatibility with existing code
 */
function updateTrendChart(trendData, period) {
  console.log('Chart functionality disabled - trend data will not be visualized');
  // No implementation - charts have been disabled
}

/**
 * Fetch settings from the server API
 */
async function fetchServerSettings() {
  try {
    const authData = await getAuthData();
    if (!authData || !authData.access_token) {
      return null;
    }
    
    const response = await fetch(`${API_ENDPOINT}/extension/settings`, {
      headers: {
        'Authorization': `Bearer ${authData.access_token}`
      }
    });
    
    if (!response.ok) {
      console.error('Failed to fetch settings from server:', response.status);
      return null;
    }
    
    const data = await response.json();
    return data.settings || null;
  } catch (error) {
    console.error('Error fetching settings from server:', error);
    return null;
  }
}

/**
 * Save settings to the server API
 */
async function saveServerSettings(settings) {
  try {
    const authData = await getAuthData();
    if (!authData || !authData.access_token) {
      return false;
    }
    
    const response = await fetch(`${API_ENDPOINT}/extension/settings`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${authData.access_token}`
      },
      body: JSON.stringify(settings)
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error saving settings to server:', error);
    return false;
  }
}