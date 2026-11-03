// Screenshots of Grafana panels for fixed time windows (run by tools/evidence.py).
// node tools/grafana_shots.js BASE_URL USER PASSWORD SHOTS.json
// SHOTS.json: [{"path": "...png", "url": "/d/<uid>/?viewPanel=..&from=..&to=.."}]
"use strict";
const fs = require("fs");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const [base, user, password, list] = process.argv.slice(2);
const shots = JSON.parse(fs.readFileSync(list, "utf8"));

(async () => {
  const browser = await chromium.launch();
  const auth = Buffer.from(`${user}:${password}`).toString("base64");
  const context = await browser.newContext({
    viewport: { width: 1100, height: 520 },
    colorScheme: "light",
    extraHTTPHeaders: { Authorization: `Basic ${auth}` },
  });
  const page = await context.newPage();
  for (const s of shots) {
    await page.goto(base + s.url + "&kiosk&theme=light", { waitUntil: "networkidle" });
    await page.waitForTimeout(2500);
    await page.screenshot({ path: s.path });
    console.log("shot", s.path);
  }
  await browser.close();
})().catch((e) => { console.error("FAILED", e); process.exit(1); });
