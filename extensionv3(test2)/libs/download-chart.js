// This file can be used to download Chart.js from CDN
// Run it with Node.js: node download-chart.js

const https = require('https');
const fs = require('fs');
const path = require('path');

// URL to Chart.js minified version
const url = 'https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.7.1/chart.min.js';
const outputFile = path.join(__dirname, 'chart.min.js');

console.log(`Downloading Chart.js from ${url}...`);

// Create a file stream to save the file
const fileStream = fs.createWriteStream(outputFile);

// Download the file
https.get(url, (response) => {
  // Check if the request was successful
  if (response.statusCode !== 200) {
    console.error(`Failed to download Chart.js: ${response.statusCode} ${response.statusMessage}`);
    fileStream.close();
    fs.unlinkSync(outputFile); // Delete the file if download failed
    return;
  }

  // Pipe the response to the file
  response.pipe(fileStream);

  // Handle errors during download
  fileStream.on('error', (err) => {
    console.error(`Error writing to file: ${err.message}`);
    fileStream.close();
    fs.unlinkSync(outputFile); // Delete the file if there was an error
  });

  // Handle successful download
  fileStream.on('finish', () => {
    fileStream.close();
    console.log(`Chart.js has been downloaded and saved to: ${outputFile}`);
  });
}).on('error', (err) => {
  console.error(`Error downloading Chart.js: ${err.message}`);
  fileStream.close();
  fs.unlinkSync(outputFile); // Delete the file if there was an error
}); 