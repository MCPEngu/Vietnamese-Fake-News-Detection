/**
 * Background script for the Toxic Language Detector Extension
 * Handles API communication and browser events
 */

// Configuration
const API_ENDPOINT = "http://localhost:7860";

// Authentication credentials - in production use a more secure storage method
const AUTH_CREDENTIALS = {
  username: "admin",
  password: "password"
};

// Create a Base64 encoded Basic Auth token
const BASIC_AUTH_TOKEN = btoa(`${AUTH_CREDENTIALS.username}:${AUTH_CREDENTIALS.password}`);

// Buffer để lưu trữ comments chờ xử lý batch
let commentsBuffer = [];
const BATCH_SIZE = 100; // Kích thước batch, xử lý 100 comments một lần
let batchProcessingTimeout = null;

// Initialize extension state
chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({
    enabled: true,
    threshold: 0.7,
    highlightToxic: true,
    platforms: {
      facebook: true,
      youtube: true,
      twitter: true
    },
    stats: {
      scanned: 0,
      clean: 0,
      offensive: 0,
      hate: 0,
      spam: 0
    },
    displayOptions: {
      showClean: true,
      showOffensive: true,
      showHate: true,
      showSpam: true
    },
    useBatchProcessing: true // Mặc định bật xử lý batch
  });
});

// Listen for messages from content script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "analyzeText") {
    // Kiểm tra xem có sử dụng batch processing không
    chrome.storage.sync.get("useBatchProcessing", (data) => {
      const useBatchProcessing = data.useBatchProcessing !== undefined ? data.useBatchProcessing : true;
      
      if (useBatchProcessing) {
        // Thêm vào buffer và gửi kết quả promise
        addToBuffer(message.text, message.platform, message.commentId, sender.tab?.url)
          .then(result => sendResponse(result))
          .catch(error => sendResponse({ error: error.message }));
      } else {
        // Phân tích ngay
        analyzeText(message.text, message.platform, message.commentId, sender.tab?.url)
          .then(result => sendResponse(result))
          .catch(error => sendResponse({ error: error.message }));
      }
    });
    return true; // Required for async sendResponse
  }
  
  if (message.action === "getSettings") {
    chrome.storage.sync.get(null, (data) => {
      sendResponse(data);
    });
    return true;
  }
  
  if (message.action === "updateSettings") {
    chrome.storage.sync.set(message.settings, () => {
      sendResponse({ success: true });
    });
    return true;
  }
  
  if (message.action === "resetStats") {
    chrome.storage.sync.set({
      stats: {
        scanned: 0,
        clean: 0,
        offensive: 0,
        hate: 0,
        spam: 0
      }
    }, () => {
      sendResponse({ success: true });
    });
    return true;
  }
  
  if (message.action === "flushBatchBuffer") {
    // Xử lý ngay batch hiện tại nếu có
    if (commentsBuffer.length > 0) {
      processCommentsBatch();
      sendResponse({ success: true, message: `Đã xử lý ${commentsBuffer.length} comments` });
    } else {
      sendResponse({ success: true, message: "Không có comments trong buffer" });
    }
    return true;
  }
});

/**
 * Thêm comment vào buffer và xử lý khi đạt kích thước batch
 * @param {string} text - The text to analyze
 * @param {string} platform - The platform (facebook, youtube, twitter)
 * @param {string} commentId - The ID of the comment
 * @param {string} sourceUrl - URL of the page
 * @returns {Promise} - Promise with result
 */
function addToBuffer(text, platform, commentId, sourceUrl) {
  return new Promise((resolve, reject) => {
    // Tạo một identifier duy nhất cho comment này
    const commentIdentifier = `${platform}_${commentId}`;
    
    // Kiểm tra xem comment đã tồn tại trong buffer chưa
    const existingIndex = commentsBuffer.findIndex(item => 
      item.identifier === commentIdentifier);
    
    if (existingIndex >= 0) {
      // Comment đã tồn tại, trả về kết quả resolved
      resolve({ 
        text: text,
        prediction: 0, // Giá trị mặc định, sẽ được cập nhật sau
        confidence: 0,
        prediction_text: "pending",
        category: "pending",
        status: "buffered",
        message: "Comment added to batch processing queue"
      });
      return;
    }
    
    // Thêm callback để trả kết quả sau khi xử lý
    const newItem = {
      text: text,
      platform: platform,
      platform_id: commentId,
      source_url: sourceUrl,
      identifier: commentIdentifier,
      resolve: resolve,
      reject: reject,
      metadata: {
        source: "extension",
        browser: navigator.userAgent,
        timestamp: new Date().toISOString()
      }
    };
    
    // Thêm vào buffer
    commentsBuffer.push(newItem);
    
    // Nếu đạt kích thước batch thì xử lý luôn
    if (commentsBuffer.length >= BATCH_SIZE) {
      // Clear timeout nếu đang chờ
      if (batchProcessingTimeout) {
        clearTimeout(batchProcessingTimeout);
        batchProcessingTimeout = null;
      }
      
      // Xử lý batch ngay lập tức
      processCommentsBatch();
    } else if (!batchProcessingTimeout) {
      // Nếu chưa đủ và chưa có timer thì set timeout
      batchProcessingTimeout = setTimeout(() => {
        processCommentsBatch();
        batchProcessingTimeout = null;
      }, 2000); // Xử lý sau 2 giây nếu không đủ kích thước batch
    }
  });
}

/**
 * Hàm chuyển đổi đối tượng JSON thành định dạng form URL encoded
 * @param {Object} data - Dữ liệu JSON cần chuyển đổi
 * @returns {string} - Chuỗi form URL encoded
 */
function jsonToFormData(data) {
  return Object.keys(data)
    .map(key => {
      if (typeof data[key] === 'object' && data[key] !== null) {
        return `${encodeURIComponent(key)}=${encodeURIComponent(JSON.stringify(data[key]))}`;
      }
      return `${encodeURIComponent(key)}=${encodeURIComponent(data[key])}`;
    })
    .join('&');
}

/**
 * Xử lý batch comments
 */
async function processCommentsBatch() {
  if (commentsBuffer.length === 0) return;
  
  // Lấy ra các comments cần xử lý
  const batchToProcess = [...commentsBuffer];
  commentsBuffer = []; // Reset buffer
  
  console.log(`Xử lý batch với ${batchToProcess.length} comments`);
  
  try {
    // Chuẩn bị request
    const batchItems = batchToProcess.map(item => ({
      text: item.text,
      platform: item.platform,
      platform_id: item.platform_id,
      source_url: item.source_url,
      metadata: item.metadata
    }));
    
    // Gọi API batch với JSON
    const response = await fetch(`${API_ENDPOINT}/extension/batch-detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Basic ${BASIC_AUTH_TOKEN}`
      },
      body: JSON.stringify({
        items: batchItems,          // Trường items được mong đợi bởi backend
        store_clean: false,         // Không lưu comments sạch
        save_to_db: false           // Không lưu vào database, chỉ trả về kết quả phân loại
      })
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        throw new Error("Authentication failed. Please check your credentials.");
      }
      if (response.status === 422) {
        // Lấy thêm thông tin chi tiết từ response để debug
        try {
          const errorData = await response.json();
          console.error("Lỗi 422: Dữ liệu không đúng định dạng", errorData);
        } catch (parseError) {
          console.error("Lỗi 422: Không thể parse error response");
        }
        
        console.error("Request body:", JSON.stringify({
          items: batchItems,
          store_clean: false,
          save_to_db: false
        }));
        throw new Error("API error: Unprocessable Entity (422)");
      }
      throw new Error(`API error: ${response.status}`);
    }
    
    const batchResult = await response.json();
    
    // Map kết quả với các item ban đầu
    const resultsMap = {};
    batchResult.results.forEach(result => {
      // Tìm item dựa trên text để map kết quả
      const matchingItem = batchToProcess.find(item => item.text === result.text);
      if (matchingItem) {
        resultsMap[matchingItem.identifier] = result;
      }
    });
    
    // Cập nhật stats
    updateStatsFromBatch(batchResult.results);
    
    // Resolve tất cả promise
    batchToProcess.forEach(item => {
      const result = resultsMap[item.identifier];
      
      if (result) {
        // Nếu có kết quả
        const categoryNames = ["clean", "offensive", "hate", "spam"];
        result.category = categoryNames[result.prediction] || "unknown";
        
        item.resolve(result);
      } else {
        // Nếu không có kết quả (có thể là comment sạch không được trả về)
        item.resolve({
          text: item.text,
          prediction: 0,
          confidence: 0.9,
          prediction_text: "clean",
          category: "clean"
        });
      }
    });
    
    console.log(`Đã xử lý batch thành công: ${batchResult.count} kết quả`);
  } catch (error) {
    console.error("Error in batch processing:", error);
    
    // Resolve tất cả với lỗi
    batchToProcess.forEach(item => {
      // Fallback to individual analysis
      analyzeText(item.text, item.platform, item.platform_id, item.source_url)
        .then(result => item.resolve(result))
        .catch(err => item.reject(err));
    });
  }
}

/**
 * Cập nhật thống kê từ kết quả batch
 * @param {Array} results - Kết quả phân tích batch
 */
function updateStatsFromBatch(results) {
  chrome.storage.sync.get("stats", (data) => {
    const stats = data.stats || { 
      scanned: 0, 
      clean: 0, 
      offensive: 0, 
      hate: 0, 
      spam: 0 
    };
    
    // Tăng số lượng đã quét
    stats.scanned += results.length;
    
    // Cập nhật thống kê theo loại
    results.forEach(result => {
      switch(result.prediction) {
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
    });
    
    chrome.storage.sync.set({ stats });
  });
}

/**
 * Analyze text using the API
 * @param {string} text - The text to analyze
 * @param {string} platform - The platform (facebook, youtube, twitter)
 * @param {string} commentId - The ID of the comment
 * @param {string} sourceUrl - URL of the page
 * @returns {Promise} - API response
 */
async function analyzeText(text, platform, commentId, sourceUrl) {
  try {
    const response = await fetch(`${API_ENDPOINT}/extension/detect`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Basic ${BASIC_AUTH_TOKEN}`
      },
      body: JSON.stringify({
        text: text,
        platform: platform,
        platform_id: commentId,
        source_url: sourceUrl,
        save_to_db: false,  // Không lưu vào database, chỉ trả về kết quả phân loại
        metadata: {
          source: "extension",
          browser: navigator.userAgent
        }
      })
    });
    
    if (!response.ok) {
      if (response.status === 401) {
        throw new Error("Authentication failed. Please check your credentials.");
      }
      if (response.status === 422) {
        console.error("Lỗi 422: Dữ liệu không đúng định dạng", text, platform);
        throw new Error("API error: Unprocessable Entity (422)");
      }
      throw new Error(`API error: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Map prediction values to category names
    const categoryNames = ["clean", "offensive", "hate", "spam"];
    result.category = categoryNames[result.prediction] || "unknown";
    
    // Update stats
    chrome.storage.sync.get("stats", (data) => {
      const stats = data.stats || { 
        scanned: 0, 
        clean: 0, 
        offensive: 0, 
        hate: 0, 
        spam: 0 
      };
      
      stats.scanned += 1;
      
      // Update specific category count
      switch(result.prediction) {
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
        default:
          // Unknown category
          break;
      }
      
      chrome.storage.sync.set({ stats });
    });
    
    return result;
  } catch (error) {
    console.error("Error analyzing text:", error);
    throw error;
  }
}