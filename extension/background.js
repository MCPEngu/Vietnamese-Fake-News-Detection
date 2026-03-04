/**
 * Background Service Worker
 * Chức năng: Nhận dữ liệu từ content.js → Gửi API → Trả kết quả
 */

// ============================================================================
// CONFIG
// ============================================================================
const API_ENDPOINT = "http://localhost:8000";

// ============================================================================
// MESSAGE LISTENER
// ============================================================================
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("[Background] Received message:", request.action);
  
  switch (request.action) {
    case "analyzePost":
      handleAnalyzePost(request.data)
        .then(result => sendResponse({ success: true, data: result }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true; // Keep channel open for async response
      
    case "getSettings":
      chrome.storage.sync.get(["mode", "groupId"], (data) => {
        sendResponse({ success: true, data: data });
      });
      return true;
      
    case "saveSettings":
      chrome.storage.sync.set(request.data, () => {
        sendResponse({ success: true });
      });
      return true;
      
    default:
      sendResponse({ success: false, error: "Unknown action" });
  }
});

// ============================================================================
// API HANDLERS
// ============================================================================

/**
 * Gửi dữ liệu bài đăng đến server để phân tích
 * @param {Object} postData - Dữ liệu bài đăng
 * @returns {Promise<Object>} - Kết quả phân tích
 */
async function handleAnalyzePost(postData) {
  try {
    // Lấy settings hiện tại
    const settings = await getSettings();
    
    // Chuẩn bị payload
    const payload = {
      content_text: postData.content_text,
      timestamp: postData.timestamp,
      mode: settings.mode || "feed", // "feed" hoặc "group"
    };
    
    // Nếu ở chế độ group, thêm user_id và group_id
    if (settings.mode === "group") {
      payload.user_id = postData.user_id;
      payload.group_id = settings.groupId;
    }
    
    console.log("[Background] Sending to API:", payload);
    
    // Gọi API
    const response = await fetch(`${API_ENDPOINT}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status}`);
    }
    
    const result = await response.json();
    console.log("[Background] API Response:", result);
    
    // Cập nhật stats
    updateStats(result.label);
    
    return result;
    
  } catch (error) {
    console.error("[Background] API Error:", error);
    throw error;
  }
}

/**
 * Lấy settings từ storage
 */
function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["mode", "groupId"], (data) => {
      resolve(data);
    });
  });
}

/**
 * Cập nhật thống kê
 */
function updateStats(label) {
  chrome.storage.sync.get(["stats"], (data) => {
    const stats = data.stats || { total: 0, real: 0, fake: 0 };
    stats.total += 1;
    if (label === 0) {
      stats.real += 1;
    } else {
      stats.fake += 1;
    }
    chrome.storage.sync.set({ stats });
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================
console.log("[Background] Service worker started");

// Set default settings on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({
    mode: "feed",
    groupId: null,
    enabled: true,
    stats: { total: 0, real: 0, fake: 0 }
  });
  console.log("[Background] Extension installed, default settings applied");
});
