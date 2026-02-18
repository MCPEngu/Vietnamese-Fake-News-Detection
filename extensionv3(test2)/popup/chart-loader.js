/**
 * Chart.js loader for Toxic Language Detector Extension
 * Uses local Chart.js library instead of CDN to avoid Content Security Policy issues
 */

// Check if Chart.js is already loaded
if (typeof Chart === 'undefined') {
  // Create script element to load Chart.js from local file
  const script = document.createElement('script');
  script.src = '../libs/chart.min.js'; // Path to local Chart.js file
  script.async = true;
  
  // Add to document
  document.head.appendChild(script);
  
  // Log that we're loading Chart.js
  console.log('Loading Chart.js from local file');
  
  // You could add a listener for load events if needed
  script.onload = function() {
    console.log('Chart.js loaded successfully');
    // You could trigger chart initialization here if needed
    if (typeof initializeCharts === 'function') {
      initializeCharts();
    }
  };
  
  script.onerror = function() {
    console.error('Failed to load Chart.js from local file');
  };
} else {
  console.log('Chart.js already loaded');
} 