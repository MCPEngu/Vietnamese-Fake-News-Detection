/**
 * Content Script - Vietnamese Fake News Detector
 * Inject vào Facebook để phát hiện tin giả
 */

// ============================================================================
// CONFIG & STATE
// ============================================================================
let extensionEnabled = true;
let currentMode = "feed";       // "feed" hoặc "group"
let currentGroupId = null;      // ID của group hiện tại (nếu có)
let lastUrl = window.location.href;

// ============================================================================
// INITIALIZATION
// ============================================================================
function initialize() {
  loadSettings();
  detectContext();      // Detect context ban đầu
  startObserver();
  startUrlWatcher();    // Watch URL changes (SPA navigation)
  setTimeout(processExistingPosts, 2000);
}

function loadSettings() {
  chrome.storage.sync.get(["enabled"], (data) => {
    extensionEnabled = data.enabled !== false;
  });
}

/**
 * Detect context từ URL hiện tại
 * Gọi khi: init, URL change
 */
function detectContext() {
  const url = window.location.href;
  const groupMatch = url.match(/facebook\.com\/groups\/([^\/\?#]+)/);
  
  if (groupMatch) {
    const newGroupId = groupMatch[1];
    
    // Nếu vào group MỚI (khác group cũ hoặc từ feed)
    if (currentMode !== "group" || currentGroupId !== newGroupId) {
      currentMode = "group";
      currentGroupId = newGroupId;
      
      // Thông báo server: đã vào group
      notifyGroupEnter(newGroupId);
      console.log(`[FakeNews] Entered group: ${newGroupId}`);
    }
  } else {
    // Không phải group → feed mode
    if (currentMode === "group") {
      // Rời group → thông báo server
      notifyGroupLeave(currentGroupId);
      console.log(`[FakeNews] Left group: ${currentGroupId}`);
    }
    
    currentMode = "feed";
    currentGroupId = null;
  }
}

/**
 * Watch URL changes - Override History API + fallback polling
 * Facebook dùng pushState/replaceState khi navigate
 */
function startUrlWatcher() {
  // Override pushState
  const originalPushState = history.pushState;
  history.pushState = function(...args) {
    originalPushState.apply(this, args);
    onUrlChange();
  };
  
  // Override replaceState  
  const originalReplaceState = history.replaceState;
  history.replaceState = function(...args) {
    originalReplaceState.apply(this, args);
    onUrlChange();
  };
  
  // Listen popstate (back/forward button)
  window.addEventListener('popstate', onUrlChange);
  
  // FALLBACK: Poll URL mỗi 1s (Facebook đôi khi không trigger History API)
  setInterval(() => {
    if (window.location.href !== lastUrl) {
      console.log(`[FakeNews] URL changed (poll): ${lastUrl} → ${window.location.href}`);
      onUrlChange();
    }
  }, 1000);
}

function onUrlChange() {
  if (window.location.href !== lastUrl) {
    console.log(`[FakeNews] onUrlChange: ${lastUrl} → ${window.location.href}`);
    lastUrl = window.location.href;
    detectContext();
  }
}

/**
 * Gửi thông báo đến server khi vào group
 */
function notifyGroupEnter(groupId) {
  if (!chrome.runtime?.id) return;
  
  chrome.runtime.sendMessage({
    action: "enterGroup",
    groupId: groupId
  });
}

/**
 * Gửi thông báo đến server khi rời group
 */
function notifyGroupLeave(groupId) {
  if (!chrome.runtime?.id || !groupId) return;
  
  chrome.runtime.sendMessage({
    action: "leaveGroup", 
    groupId: groupId
  });
}

// ============================================================================
// DOM OBSERVER
// ============================================================================
function startObserver() {
  // Scan định kỳ mỗi 3 giây
  setInterval(() => {
    if (extensionEnabled) processExistingPosts();
  }, 3000);
  
  // Scan khi scroll (debounced)
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      if (extensionEnabled) processExistingPosts();
    }, 500);
  }, { passive: true });
}

// ============================================================================
// POST EXTRACTION
// ============================================================================
function findArticles() {
  const storyMessages = document.querySelectorAll('[data-ad-rendering-role="story_message"]');
  const posts = [];
  
  storyMessages.forEach((storyMsg, index) => {
    // Tìm container cha
    const container = storyMsg.closest('[data-pagelet]') || 
                      storyMsg.closest('.x1yztbdb') ||
                      storyMsg.parentElement?.parentElement?.parentElement?.parentElement ||
                      storyMsg;
    
    posts.push({ element: container, storyMessage: storyMsg, index });
  });
  
  // Fallback: tìm div[dir="auto"] có text dài
  if (posts.length === 0) {
    const dirAutoDivs = document.querySelectorAll('div[dir="auto"]');
    dirAutoDivs.forEach((div, index) => {
      const text = div.textContent?.trim() || "";
      if (text.length > 30 && !isMetadata(text)) {
        const container = div.closest('[data-pagelet]') || div.parentElement?.parentElement;
        if (container && !posts.some(p => p.element === container)) {
          posts.push({ element: container || div, storyMessage: div, index });
        }
      }
    });
  }
  
  return posts;
}

function isMetadata(text) {
  const first50 = text.substring(0, 50);
  return /^\d+\s*(giờ|phút|ngày|tuần|tháng|giây)/i.test(first50) ||
         /^(Like|Comment|Share|Thích|Bình luận|Chia sẻ)/i.test(first50) ||
         /^\d+[KMk]?\s*(likes?|comments?|shares?|lượt)/i.test(first50) ||
         /^(Sponsored|Được tài trợ|See more|Xem thêm)/i.test(first50);
}

// ============================================================================
// PROCESS POSTS
// ============================================================================
function processExistingPosts() {
  const posts = findArticles();
  
  for (const post of posts) {
    const { element: container, storyMessage } = post;
    const text = storyMessage.textContent?.trim() || "";
    
    // Skip conditions
    if (container.querySelector('.fake-news-indicator')) continue;
    if (storyMessage.parentElement?.querySelector('.fake-news-indicator')) continue;
    if (text.length < 5) continue;
    if (container.dataset.fakeNewsChecked === 'pending') continue;
    
    // Process
    container.dataset.fakeNewsChecked = 'pending';
    processPost(container, storyMessage, text);
  }
}

function processPost(container, storyMessage, text) {
  if (!chrome.runtime?.id) {
    container.dataset.fakeNewsChecked = '';
    return;
  }
  
  // Tạo postData - khác nhau tùy mode
  const postData = {
    content_text: text,
    timestamp: new Date().toISOString(),
    mode: currentMode
  };

  const engagement = extractEngagement(container);
  postData.num_like = engagement.num_like;
  postData.num_cmt = engagement.num_cmt;
  postData.num_share = engagement.num_share;
  
  // Group mode: extract user_id từ post
  if (currentMode === "group") {
    postData.user_id = extractUserId(container);
  }
  
  chrome.runtime.sendMessage(
    { action: "analyzePost", data: postData },
    (response) => {
      if (chrome.runtime.lastError) {
        container.dataset.fakeNewsChecked = '';
        return;
      }
      
      if (response?.success) {
        displayResult(container, storyMessage, response.data);
        container.dataset.fakeNewsChecked = 'done';
      } else {
        container.dataset.fakeNewsChecked = '';
      }
    }
  );
}

function parseCompactNumber(text) {
  if (!text) return 0;
  const raw = String(text).trim().toLowerCase();
  const normalized = raw.replace(/,/g, '.').replace(/\s+/g, '');
  const match = normalized.match(/(\d+(?:\.\d+)?)([kmb]|nghìn|ngan|triệu|trieu|tỷ|ty)?/i);
  if (!match) return 0;

  const value = parseFloat(match[1]);
  const suffix = (match[2] || '').toLowerCase();
  if (Number.isNaN(value)) return 0;

  if (suffix === 'k' || suffix === 'nghìn' || suffix === 'ngan') return Math.round(value * 1_000);
  if (suffix === 'm' || suffix === 'triệu' || suffix === 'trieu') return Math.round(value * 1_000_000);
  if (suffix === 'b' || suffix === 'tỷ' || suffix === 'ty') return Math.round(value * 1_000_000_000);
  return Math.round(value);
}

function extractEngagement(container) {
  const text = (container?.innerText || '').replace(/\s+/g, ' ');
  if (!text) {
    return { num_like: 0, num_cmt: 0, num_share: 0 };
  }

  let numLike = 0;
  let numCmt = 0;
  let numShare = 0;

  const likePatterns = [
    /(\d+(?:[\.,]\d+)?\s*(?:k|m|b|nghìn|ngan|triệu|trieu|tỷ|ty)?)\s*(?:lượt\s*)?(?:thích|like)s?/i,
  ];
  const cmtPatterns = [
    /(\d+(?:[\.,]\d+)?\s*(?:k|m|b|nghìn|ngan|triệu|trieu|tỷ|ty)?)\s*(?:bình\s*luận|comment)s?/i,
  ];
  const sharePatterns = [
    /(\d+(?:[\.,]\d+)?\s*(?:k|m|b|nghìn|ngan|triệu|trieu|tỷ|ty)?)\s*(?:lượt\s*)?(?:chia\s*sẻ|share)s?/i,
  ];

  for (const pattern of likePatterns) {
    const m = text.match(pattern);
    if (m) {
      numLike = parseCompactNumber(m[1]);
      break;
    }
  }
  for (const pattern of cmtPatterns) {
    const m = text.match(pattern);
    if (m) {
      numCmt = parseCompactNumber(m[1]);
      break;
    }
  }
  for (const pattern of sharePatterns) {
    const m = text.match(pattern);
    if (m) {
      numShare = parseCompactNumber(m[1]);
      break;
    }
  }

  return {
    num_like: numLike,
    num_cmt: numCmt,
    num_share: numShare,
  };
}

/**
 * Extract user ID từ post container (dùng cho group mode)
 */
function extractUserId(container) {
  // Tìm link profile của người đăng
  const userLinks = container.querySelectorAll('a[href*="/user/"], a[href*="profile.php"], a[role="link"][href*="facebook.com/"]');
  
  for (const link of userLinks) {
    const href = link.getAttribute('href') || '';
    
    // Pattern 1: /user/123456789
    const userIdMatch = href.match(/\/user\/(\d+)/);
    if (userIdMatch) return userIdMatch[1];
    
    // Pattern 2: profile.php?id=123456789
    const profileMatch = href.match(/profile\.php\?id=(\d+)/);
    if (profileMatch) return profileMatch[1];
    
    // Pattern 3: facebook.com/username (không phải groups, posts, etc)
    if (!href.includes('/groups/') && !href.includes('/posts/') && !href.includes('/photos/')) {
      const usernameMatch = href.match(/facebook\.com\/([a-zA-Z0-9.]+)(?:\/|\?|$)/);
      if (usernameMatch && usernameMatch[1] !== 'groups') {
        return usernameMatch[1]; // Return username nếu không có numeric ID
      }
    }
  }
  
  return null;
}

// ============================================================================
// DISPLAY RESULTS
// ============================================================================
function displayResult(container, storyMessage, result) {
  if (!container || !document.body.contains(container)) return;
  if (container.querySelector('.fake-news-indicator')) return;
  
  const indicator = document.createElement('div');
  indicator.className = 'fake-news-indicator';
  
  if (result.label === 0) {
    indicator.classList.add('real');
    indicator.innerHTML = `
      <span class="icon">✓</span>
      <span class="text">Tin thật</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
    container.style.borderLeft = '4px solid #4CAF50';
  } else {
    indicator.classList.add('fake');
    indicator.innerHTML = `
      <span class="icon">⚠</span>
      <span class="text">Nghi ngờ tin giả</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
    container.style.borderLeft = '4px solid #f44336';
  }
  
  // Chèn indicator
  if (storyMessage?.parentElement && document.body.contains(storyMessage)) {
    storyMessage.parentElement.insertBefore(indicator, storyMessage);
  } else {
    container.insertBefore(indicator, container.firstChild);
  }
  
  container.style.paddingLeft = '8px';
}

// ============================================================================
// MESSAGE LISTENER
// ============================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case "toggleEnabled":
      extensionEnabled = request.enabled;
      sendResponse({ success: true });
      break;
      
    case "reprocessPosts":
      document.querySelectorAll('.fake-news-indicator').forEach(el => el.remove());
      document.querySelectorAll('[data-fake-news-checked]').forEach(el => {
        delete el.dataset.fakeNewsChecked;
        el.style.borderLeft = '';
        el.style.paddingLeft = '';
      });
      processExistingPosts();
      sendResponse({ success: true });
      break;
  }
});

// ============================================================================
// START
// ============================================================================
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initialize);
} else {
  initialize();
}
