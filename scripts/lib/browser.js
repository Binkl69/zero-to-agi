/* Find a Chromium to drive. A hard-coded path here once meant the real-browser
   checks only ran on the one machine they were written on. Order: an explicit
   CHROME_PATH, then whatever Playwright has installed, then the usual system
   locations. Returns launch options for chromium.launch(). */
const fs = require('fs');

module.exports = function launchOptions(extraArgs) {
  const args = ['--no-sandbox'].concat(extraArgs || []);
  const candidates = [
    process.env.CHROME_PATH,
    '/opt/pw-browsers/chromium',
    '/usr/bin/chromium',
    '/usr/bin/chromium-browser',
    '/usr/bin/google-chrome',
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  ].filter(Boolean);
  for (const p of candidates) {
    try { if (fs.statSync(p).isFile()) return { executablePath: p, args }; } catch (e) {}
  }
  // no known binary: let Playwright use its own download (npx playwright install chromium)
  return { args };
};
