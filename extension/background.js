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
  switch (request.action) {
    case "analyzePost":
      handleAnalyzePost(request.data)
        .then(result => sendResponse({ success: true, data: result }))
        .catch(error => sendResponse({ success: false, error: error.message }));
      return true;
      
    case "enterGroup":
      // Lưu vào storage (để persist qua service worker restarts)
      console.log(`[BG] enterGroup: ${request.groupId}`);
      chrome.storage.local.set({ 
        currentContext: { mode: "group", groupId: request.groupId }
      });
      notifyServerGroupEnter(request.groupId);
      sendResponse({ success: true });
      return true;
      
    case "leaveGroup":
      console.log(`[BG] leaveGroup: ${request.groupId}`);
      chrome.storage.local.set({ 
        currentContext: { mode: "feed", groupId: null }
      });
      notifyServerGroupLeave(request.groupId);
      sendResponse({ success: true });
      return true;
      
    case "getContext":
      chrome.storage.local.get(['currentContext'], (data) => {
        sendResponse({ 
          success: true, 
          data: data.currentContext || { mode: "feed", groupId: null }
        });
      });
      return true;
      
    default:
      sendResponse({ success: false, error: "Unknown action" });
  }
});

/**
 * Notify server khi user vào group (tạo file thống kê nếu chưa có)
 */
async function notifyServerGroupEnter(groupId) {
  try {
    await fetch(`${API_ENDPOINT}/group/enter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group_id: groupId })
    });
  } catch (e) {
    console.error("[BG] notifyServerGroupEnter failed:", e);
  }
}

/**
 * Notify server khi user rời group
 */
async function notifyServerGroupLeave(groupId) {
  try {
    await fetch(`${API_ENDPOINT}/group/leave`, {
      method: "POST", 
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ group_id: groupId })
    });
  } catch (e) {
    console.error("[BG] notifyServerGroupLeave failed:", e);
  }
}

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
    // Lấy context từ storage
    const storageData = await chrome.storage.local.get(['currentContext']);
    const currentContext = storageData.currentContext || { mode: "feed", groupId: null };
    
    // Chuẩn bị payload
    const payload = {
      content_text: postData.content_text,
      timestamp: postData.timestamp,
      mode: postData.mode || currentContext.mode,
      num_like: Number(postData.num_like || 0),
      num_cmt: Number(postData.num_cmt || 0),
      num_share: Number(postData.num_share || 0)
    };
    
    // Nếu ở chế độ group, thêm user_id và group_id
    if (payload.mode === "group" && currentContext.groupId) {
      payload.user_id = postData.user_id;
      payload.group_id = currentContext.groupId;
    }
    
    // Gọi API
    const response = await fetch(`${API_ENDPOINT}/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`API Error ${response.status}: ${errText}`);
    }
    
    const result = await response.json();

    // Cập nhật stats
    updateStats(result.label);
    
    return result;
    
  } catch (error) {
    console.error("[BG] handleAnalyzePost failed:", error, "postData:", postData);
    throw error;
  }
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

// Set default settings on install
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({
    mode: "feed",
    groupId: null,
    enabled: true,
    stats: { total: 0, real: 0, fake: 0 }
  });
});
