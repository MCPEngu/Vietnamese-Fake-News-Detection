/**
 * Content Script - Inject vào Facebook
 * Chức năng: Đọc DOM → Lấy thông tin bài đăng → Gửi cho background.js
 * 
 * QUAN TRỌNG: Mỗi bài đăng = 1 ARTICLE container (div[role="article"])
 * Không tìm từng text element nhỏ → tránh 1 bài bị chia nhiều phần
 */

// ============================================================================
// CONFIG & STATE
// ============================================================================
let extensionEnabled = true;
let currentMode = "feed";
let currentGroupId = null;
const processedArticles = new Set(); // Track bằng article ID, không phải text

// ============================================================================
// INITIALIZATION
// ============================================================================
function initialize() {
  console.log("[Content] Initializing...");
  
  loadSettings();
  detectGroupContext();
  startObserver();
  
  // Delay để đợi FB render xong
  setTimeout(() => {
    processExistingPosts();
  }, 1500);
  
  console.log("[Content] Initialized successfully");
}

function loadSettings() {
  chrome.storage.sync.get(["enabled", "mode", "groupId"], (data) => {
    extensionEnabled = data.enabled !== false;
    currentMode = data.mode || "feed";
    currentGroupId = data.groupId;
  });
}

function detectGroupContext() {
  const url = window.location.href;
  const groupMatch = url.match(/facebook\.com\/groups\/([^\/\?]+)/);
  
  if (groupMatch) {
    currentGroupId = groupMatch[1];
    currentMode = "group";
    chrome.runtime.sendMessage({
      action: "saveSettings",
      data: { mode: "group", groupId: currentGroupId }
    });
    console.log("[Content] Group mode:", currentGroupId);
  } else {
    currentMode = "feed";
    console.log("[Content] Feed mode");
  }
}

// ============================================================================
// DOM OBSERVER - Đơn giản hóa, dùng interval scan
// ============================================================================

function startObserver() {
  // Scan định kỳ mỗi 2 giây (đơn giản và hiệu quả hơn MutationObserver cho FB)
  setInterval(() => {
    if (extensionEnabled) {
      processExistingPosts();
    }
  }, 2000);
  
  // Scan khi scroll
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      if (extensionEnabled) {
        processExistingPosts();
      }
    }, 500);
  });
  
  console.log("[Content] Observer started (interval mode)");
}

// ============================================================================
// POST EXTRACTION - TÌM ARTICLE CONTAINER (KHÔNG PHẢI TEXT ELEMENTS)
// ============================================================================

/**
 * Tìm tất cả ARTICLE containers - mỗi article = 1 bài đăng duy nhất
 */
function findArticles() {
  // Facebook dùng div[role="article"] cho mỗi bài đăng
  return Array.from(document.querySelectorAll('div[role="article"]'));
}

/**
 * Tạo ID duy nhất cho article
 */
function getArticleId(article) {
  // Cách 1: Dùng aria-describedby (unique ID từ Facebook)
  const ariaDescribedBy = article.getAttribute('aria-describedby');
  if (ariaDescribedBy) {
    return `fb-${ariaDescribedBy}`;
  }
  
  // Cách 2: Tìm link đến post
  const postLink = article.querySelector('a[href*="/posts/"], a[href*="permalink"], a[href*="story_fbid"]');
  if (postLink) {
    const href = postLink.getAttribute('href');
    const match = href.match(/(\d{10,})/); // Tìm số dài (post ID)
    if (match) {
      return `fb-post-${match[1]}`;
    }
  }
  
  // Cách 3: Hash từ vị trí + text ngắn (fallback)
  const rect = article.getBoundingClientRect();
  const text = getArticleText(article);
  if (text && text.length > 30) {
    return `fb-hash-${hashString(text.substring(0, 100) + rect.top)}`;
  }
  
  return null;
}

/**
 * Lấy TEXT CONTENT chính của bài đăng (không lấy comments)
 */
function getArticleText(article) {
  // Đánh dấu đã check article này
  if (article.dataset.fakeNewsChecked === 'pending') {
    return null; // Đang xử lý
  }
  
  // Cách 1: Tìm div có data attribute đặc biệt (ưu tiên cao)
  const messageDiv = article.querySelector('[data-ad-preview="message"]') ||
                     article.querySelector('[data-ad-comet-preview="message"]') ||
                     article.querySelector('[data-ad-rendering-role="story_message"]');
  
  if (messageDiv) {
    return messageDiv.textContent?.trim() || "";
  }
  
  // Cách 2: Tìm div[dir="auto"] có text dài nhất trong phần trên của article
  // (phần dưới thường là comments)
  const allDirAuto = article.querySelectorAll('div[dir="auto"]');
  
  let bestText = "";
  let bestLength = 0;
  
  for (const div of allDirAuto) {
    // Bỏ qua nếu nằm trong comment section
    if (div.closest('[aria-label*="comment"], [aria-label*="bình luận"]')) {
      continue;
    }
    
    const text = div.textContent?.trim() || "";
    
    // Bỏ qua text quá ngắn hoặc là metadata
    if (text.length < 30 || isMetadata(text)) {
      continue;
    }
    
    // Lấy text dài nhất (thường là nội dung chính)
    if (text.length > bestLength) {
      bestText = text;
      bestLength = text.length;
    }
  }
  
  return bestText;
}

/**
 * Kiểm tra text có phải metadata (time, reactions, etc.)
 */
function isMetadata(text) {
  const first50 = text.substring(0, 50);
  const metaPatterns = [
    /^\d+\s*(giờ|phút|ngày|tuần|tháng|giây)/i,
    /^(Like|Comment|Share|Thích|Bình luận|Chia sẻ)/i,
    /^\d+[KMk]?\s*(likes?|comments?|shares?|lượt)/i,
    /^(Sponsored|Được tài trợ)/i,
    /^(See more|Xem thêm|See less|Thu gọn)/i,
    /^(Write a comment|Viết bình luận)/i,
    /^All reactions/i,
  ];
  
  return metaPatterns.some(pattern => pattern.test(first50));
}

/**
 * Process existing posts on page
 */
function processExistingPosts() {
  const articles = findArticles();
  let processedCount = 0;
  
  for (const article of articles) {
    // Bỏ qua nếu đã có indicator
    if (article.querySelector('.fake-news-indicator')) {
      continue;
    }
    
    const articleId = getArticleId(article);
    
    if (!articleId || processedArticles.has(articleId)) {
      continue;
    }
    
    // Kiểm tra article có đang hiển thị trên màn hình không
    const rect = article.getBoundingClientRect();
    const isVisible = rect.top < window.innerHeight + 200 && rect.bottom > -200;
    
    if (!isVisible) {
      continue; // Chỉ process bài gần viewport
    }
    
    // Đánh dấu đang xử lý
    article.dataset.fakeNewsChecked = 'pending';
    processedArticles.add(articleId);
    
    processArticle(article, articleId);
    processedCount++;
  }
  
  if (processedCount > 0) {
    console.log(`[Content] Processed ${processedCount} new articles`);
  }
}

/**
 * Process một article
 */
function processArticle(article, articleId) {
  const text = getArticleText(article);
  
  if (!text || text.length < 30) {
    console.log(`[Content] Skip article: text too short or empty`);
    article.dataset.fakeNewsChecked = 'skipped';
    return;
  }
  
  console.log(`[Content] Processing: "${text.substring(0, 50)}..."`);
  
  // Tạo post data
  const postData = {
    content_text: text,
    timestamp: extractTimestamp(article),
    user_id: currentMode === "group" ? extractUserId(article) : null
  };
  
  // Gửi đến background
  if (!chrome.runtime?.id) {
    console.warn("[Content] Extension context invalidated");
    return;
  }
  
  try {
    chrome.runtime.sendMessage(
      { action: "analyzePost", data: postData },
      (response) => {
        if (chrome.runtime.lastError) {
          console.error("[Content] Runtime error:", chrome.runtime.lastError.message);
          article.dataset.fakeNewsChecked = 'error';
          return;
        }
        
        if (response?.success) {
          displayResult(article, response.data);
          article.dataset.fakeNewsChecked = 'done';
        } else {
          console.error("[Content] Analysis failed:", response?.error);
          article.dataset.fakeNewsChecked = 'error';
        }
      }
    );
  } catch (e) {
    console.error("[Content] sendMessage error:", e);
    article.dataset.fakeNewsChecked = 'error';
  }
}

/**
 * Extract timestamp từ article
 */
function extractTimestamp(article) {
  // Tìm link chứa thời gian (thường là link to post)
  const timeLinks = article.querySelectorAll('a[href*="/posts/"], a[href*="permalink"], a[href*="story_fbid"]');
  for (const link of timeLinks) {
    const spans = link.querySelectorAll('span');
    for (const span of spans) {
      const text = span.textContent?.trim();
      if (text && /^\d+\s*(giờ|phút|ngày|h|m|d)/i.test(text)) {
        return parseTimestamp(text);
      }
    }
  }
  return new Date().toISOString();
}

/**
 * Extract user ID từ article
 */
function extractUserId(article) {
  const userLinks = article.querySelectorAll('a[href*="/user/"], a[href*="profile.php"], a[href*="facebook.com/"]');
  for (const link of userLinks) {
    const href = link.getAttribute('href');
    const match = href.match(/user\/(\d+)|id=(\d+)/);
    if (match) return match[1] || match[2];
  }
  return null;
}

/**
 * Parse timestamp text thành ISO format
 */
function parseTimestamp(timeText) {
  if (!timeText) return new Date().toISOString();
  
  const now = new Date();
  
  if (/phút|m\b/i.test(timeText)) {
    const mins = parseInt(timeText) || 0;
    return new Date(now - mins * 60000).toISOString();
  }
  
  if (/giờ|h\b/i.test(timeText)) {
    const hours = parseInt(timeText) || 0;
    return new Date(now - hours * 3600000).toISOString();
  }
  
  if (/ngày|d\b/i.test(timeText)) {
    const days = parseInt(timeText) || 0;
    return new Date(now - days * 86400000).toISOString();
  }
  
  if (/hôm qua|yesterday/i.test(timeText)) {
    return new Date(now - 86400000).toISOString();
  }
  
  return now.toISOString();
}

/**
 * Hash string to create ID
 */
function hashString(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16);
}

// ============================================================================
// DISPLAY RESULTS - Hiển thị KẾT QUẢ trên ARTICLE CONTAINER
// ============================================================================

/**
 * Hiển thị kết quả phân tích trên article
 */
function displayResult(article, result) {
  if (!article || !document.body.contains(article)) {
    console.warn("[Content] Article no longer in DOM");
    return;
  }
  
  // Kiểm tra đã có indicator chưa (tránh duplicate)
  if (article.querySelector('.fake-news-indicator')) {
    return;
  }
  
  // Tạo indicator element
  const indicator = document.createElement('div');
  indicator.className = 'fake-news-indicator';
  
  if (result.label === 0) {
    indicator.classList.add('real');
    indicator.innerHTML = `
      <span class="icon">✓</span>
      <span class="text">Tin thật</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
  } else {
    indicator.classList.add('fake');
    indicator.innerHTML = `
      <span class="icon">⚠</span>
      <span class="text">Nghi ngờ tin giả</span>
      <span class="confidence">${(result.confidence * 100).toFixed(0)}%</span>
    `;
  }
  
  // Chèn indicator vào ĐẦU article
  article.style.position = 'relative';
  article.insertBefore(indicator, article.firstChild);
  
  // Thêm border cho article 
  if (result.label === 0) {
    article.style.borderLeft = '4px solid #4CAF50';
  } else {
    article.style.borderLeft = '4px solid #f44336';
  }
  article.style.paddingLeft = '8px';
  
  console.log(`[Content] Displayed: ${result.label === 0 ? 'REAL' : 'FAKE'} (${(result.confidence * 100).toFixed(0)}%)`);
}

// ============================================================================
// MESSAGE LISTENER
// ============================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case "toggleEnabled":
      extensionEnabled = request.enabled;
      console.log("[Content] Extension", extensionEnabled ? "enabled" : "disabled");
      sendResponse({ success: true });
      break;
      
    case "reprocessPosts":
      processedArticles.clear();
      // Xóa tất cả indicators hiện có
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
