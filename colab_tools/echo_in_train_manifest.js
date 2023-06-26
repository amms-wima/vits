const fs = require('fs');       // import fs from 'fs';
const path = require('path');   // import path from 'path';

// Get command line arguments
const args = process.argv.slice(2);
const filePath = args[0];

if (!filePath) {
  console.error('Please provide the path to the JSON file.');
  process.exit(1);
}

// Read the JSON file
const resolvedPath = path.resolve(filePath);
let jsonData;
try {
  jsonData = fs.readFileSync(resolvedPath, 'utf8');
} catch (err) {
  console.error('Failed to read the JSON file:', err.message);
  process.exit(1);
}

// Parse JSON
let manifest;
try {
  manifest = JSON.parse(jsonData);
} catch (err) {
  console.error('Failed to parse JSON:', err.message);
  process.exit(1);
}

// Extract required properties
const { last_updated: lastUpdated, latest_step: latestStep, summed_losses: summedLosses, best_model: { step: bestModelStep, summed_losses: bestModelSummedLosses } = {} } = manifest;

// Format the output line
const outputLine = `${lastUpdated}> curr: [${latestStep}, ${summedLosses.toFixed(3)}], best: [${bestModelStep || ''}, ${bestModelSummedLosses ? bestModelSummedLosses.toFixed(2) : ''}]`;

// Log the output line
console.log(outputLine);
