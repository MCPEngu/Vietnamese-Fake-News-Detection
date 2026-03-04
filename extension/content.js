/**
 * Content Script - Inject vào Facebook
 * Chức năng: Đọc DOM → Lấy thông tin bài đăng → Gửi cho background.js
 */

// ============================================================================
// CONFIG & STATE
// ============================================================================
let extensionEnabled = true;
let currentMode = "feed"; // "feed" hoặc "group"
let currentGroupId = null;
const processedPosts = new Set(); // Tránh xử lý trùng

// ============================================================================
// INITIALIZATION
// ============================================================================
function initialize() {
  console.log("[Content] Initializing...");
  
  // Load settings
  loadSettings();
  
  // Detect if we're in a group
  detectGroupContext();
  
  // Start observing DOM changes
  startObserver();
  
  // Process existing posts
  processExistingPosts();
  
  console.log("[Content] Initialized successfully");
}

/**
 * Load settings from storage
 */
function loadSettings() {
  chrome.storage.sync.get(["enabled", "mode", "groupId"], (data) => {
    extensionEnabled = data.enabled !== false;
    currentMode = data.mode || "feed";
    currentGroupId = data.groupId;
    console.log("[Content] Settings loaded:", { extensionEnabled, currentMode, currentGroupId });
  });
}

/**
 * Detect if current page is a Facebook group
 */
function detectGroupContext() {
  const url = window.location.href;
  
  // Check if URL contains /groups/
  const groupMatch = url.match(/facebook\.com\/groups\/([^\/\?]+)/);
  
  if (groupMatch) {
    currentGroupId = groupMatch[1];
    currentMode = "group";
    
    // Save to storage
    chrome.runtime.sendMessage({
      action: "saveSettings",
      data: { mode: "group", groupId: currentGroupId }
    });
    
    console.log("[Content] Detected group:", currentGroupId);
  } else {
    currentMode = "feed";
    console.log("[Content] Feed mode");
  }
}

// ============================================================================
// DOM OBSERVER
// ============================================================================
let observer = null;

function startObserver() {
  observer = new MutationObserver((mutations) => {
    if (!extensionEnabled) return;
    
    for (const mutation of mutations) {
      if (mutation.type === "childList" && mutation.addedNodes.length > 0) {
        for (const node of mutation.addedNodes) {
          if (node.nodeType === Node.ELEMENT_NODE) {
            // Tìm các bài đăng mới
            const posts = findPosts(node);
            posts.forEach(processPost);
          }
        }
      }
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
  
  console.log("[Content] Observer started");
}

function stopObserver() {
  if (observer) {
    observer.disconnect();
    observer = null;
  }
}

// ============================================================================
// POST EXTRACTION
// ============================================================================

/**
 * Find all post elements within a node
 */
function findPosts(node) {
  // Facebook post selectors (có thể cần cập nhật khi FB thay đổi DOM)
  const selectors = [
    '[data-ad-preview="message"]', // Post content
    'div[data-ad-comet-preview="message"]',
    'div[data-ad-rendering-role="story_message"]',
    // Fallback selectors
    'div.x1iorvi4[dir="auto"]', // Common content wrapper
  ];
  
  const posts = [];
  
  for (const selector of selectors) {
    if (node.matches && node.matches(selector)) {
      posts.push(node);
    }
    const found = node.querySelectorAll ? node.querySelectorAll(selector) : [];
    posts.push(...found);
  }
  
  return posts;
}

/**
 * Process existing posts on page load
 */
function processExistingPosts() {
  const posts = findPosts(document.body);
  console.log(`[Content] Found ${posts.length} existing posts`);
  posts.forEach(processPost);
}

/**
 * Process a single post element
 */
function processPost(postElement) {
  // Generate unique ID for post
  const postId = generatePostId(postElement);
  
  if (!postId || processedPosts.has(postId)) {
    return; // Already processed
  }
  
  processedPosts.add(postId);
  
  // Extract post data
  const postData = extractPostData(postElement);
  
  if (!postData || !postData.content_text || postData.content_text.length < 20) {
    return; // Skip short or empty posts
  }
  
  console.log("[Content] Processing post:", postData.content_text.substring(0, 50) + "...");
  
  // Send to background for analysis
  chrome.runtime.sendMessage(
    {
      action: "analyzePost",
      data: postData
    },
    (response) => {
      if (response && response.success) {
        displayResult(postElement, response.data);
      } else {
        console.error("[Content] Analysis failed:", response?.error);
      }
    }
  );
}

/**
 * Extract data from post element
 */
function extractPostData(postElement) {
  const data = {
    content_text: null,
    timestamp: null,
    user_id: null
  };
  
  // 1. Extract content text
  data.content_text = postElement.textContent?.trim() || "";
  
  // 2. Extract timestamp
  // Facebook hiển thị timestamp trong nhiều format khác nhau
  const postContainer = postElement.closest('[data-ad-rendering-role]') 
                       || postElement.closest('div[role="article"]')
                       || postElement.parentElement?.parentElement;
  
  if (postContainer) {
    // Tìm timestamp element
    const timeElement = postContainer.querySelector('a[href*="/posts/"] span')
                       || postContainer.querySelector('span[id*="jsc"]');
    
    if (timeElement) {
      data.timestamp = parseTimestamp(timeElement.textContent);
    }
  }
  
  // Fallback: use current time if timestamp not found
  if (!data.timestamp) {
    data.timestamp = new Date().toISOString();
  }
  
  // 3. Extract user_id (only in group mode)
  if (currentMode === "group") {
    const userLink = postContainer?.querySelector('a[href*="user/"]')
                    || postContainer?.querySelector('a[href*="/profile.php"]')
                    || postContainer?.querySelector('h2 a, h3 a, h4 a');
    
    if (userLink) {
      const href = userLink.getAttribute('href');
      data.user_id = extractUserIdFromHref(href);
    }
  }
  
  return data;
}

/**
 * Parse Facebook timestamp to ISO format
 */
function parseTimestamp(timeText) {
  if (!timeText) return new Date().toISOString();
  
  const now = new Date();
  
  // Common patterns: "2 giờ", "3 phút", "Hôm qua", "15 tháng 2"
  if (timeText.includes("phút")) {
    const mins = parseInt(timeText) || 0;
    return new Date(now - mins * 60000).toISOString();
  }
  
  if (timeText.includes("giờ")) {
    const hours = parseInt(timeText) || 0;
    return new Date(now - hours * 3600000).toISOString();
  }
  
  if (timeText.includes("Hôm qua") || timeText.includes("hôm qua")) {
    return new Date(now - 86400000).toISOString();
  }
  
  // Default: return current time
  return now.toISOString();
}

/**
 * Extract user ID from profile URL
 */
function extractUserIdFromHref(href) {
  if (!href) return null;
  
  // Pattern 1: /user/123456789/
  const userMatch = href.match(/user\/(\d+)/);
  if (userMatch) return userMatch[1];
  
  // Pattern 2: /profile.php?id=123456789
  const idMatch = href.match(/id=(\d+)/);
  if (idMatch) return idMatch[1];
  
  // Pattern 3: /username (vanity URL)
  const vanityMatch = href.match(/facebook\.com\/([^\/\?]+)/);
  if (vanityMatch && !vanityMatch[1].includes('.')) {
    return vanityMatch[1]; // Return username as ID
  }
  
  return null;
}

/**
 * Generate unique ID for a post
 */
function generatePostId(postElement) {
  const text = postElement.textContent?.substring(0, 100) || "";
  return hashString(text);
}

function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash.toString(16);
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================

/**
 * Display analysis result on the post
 */
function displayResult(postElement, result) {
  // Remove existing indicator if any
  const existingIndicator = postElement.parentElement?.querySelector('.fake-news-indicator');
  if (existingIndicator) {
    existingIndicator.remove();
  }
  
  // Create indicator element
  const indicator = document.createElement('div');
  indicator.className = 'fake-news-indicator';
  
  // Set style based on result
  if (result.label === 0) {
    // Real news
    indicator.classList.add('real');
    indicator.innerHTML = `
      <span class="icon">✓</span>
      <span class="text">Tin thật</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
  } else {
    // Fake news
    indicator.classList.add('fake');
    indicator.innerHTML = `
      <span class="icon">⚠</span>
      <span class="text">Nghi ngờ tin giả</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
  }
  
  // Insert indicator
  postElement.style.position = 'relative';
  postElement.parentElement.insertBefore(indicator, postElement);
  
  // Add border to post
  if (result.label === 0) {
    postElement.style.borderLeft = '3px solid #4CAF50';
  } else {
    postElement.style.borderLeft = '3px solid #f44336';
  }
  postElement.style.paddingLeft = '10px';
  
  console.log(`[Content] Displayed result: ${result.label === 0 ? 'REAL' : 'FAKE'}`);
}

// ============================================================================
// MESSAGE LISTENER (from popup)
// ============================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case "toggleEnabled":
      extensionEnabled = request.enabled;
      console.log("[Content] Extension", extensionEnabled ? "enabled" : "disabled");
      sendResponse({ success: true });
      break;
      
    case "reprocessPosts":
      processedPosts.clear();
      processExistingPosts();
      sendResponse({ success: true });
      break;
  }
});

// ============================================================================
// START
// ============================================================================
// Wait for DOM to be ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
