// Screenshots of every page of the web app (run by slurm/ui_preview.sh).
// node tools/ui_shots.js BASE_URL OUT_DIR PASSWORD
"use strict";
const path = require("path");
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright-core");

const [base, out, password] = process.argv.slice(2);
const HDR = { "X-AskIISc": "1" };

async function login(context, user) {
  const r = await context.request.post(`${base}/api/auth/login`, { data: { user_id: user, password }, headers: HDR });
  if (!r.ok()) throw new Error(`login ${user}: ${r.status()}`);
}

async function shot(page, name, opts) {
  await page.waitForTimeout(400);
  await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: !!(opts && opts.full) });
  console.log("shot", name);
}

async function run(browser, theme, viewport) {
  const suffix = `${theme}${viewport.width < 600 ? "-mobile" : ""}`;
  const context = await browser.newContext({ viewport, colorScheme: theme, deviceScaleFactor: 1 });
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(`SCRIPT ERROR ${page.url()}: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") errors.push(`${page.url()}: ${m.text()}`); });

  await page.goto(base + "/");
  await shot(page, `01-login-${suffix}`);
  if (viewport.width >= 600) {
    await page.click("#to-register"); await page.fill("#reg-user", "nobody@iisc.example");
    await page.fill("#reg-password", "long enough phrase"); await page.fill("#reg-again", "long enough phrase");
    await page.click("#reg-send"); await page.waitForTimeout(500);
    await shot(page, `02-register-${suffix}`);
    await page.click("#register-form [data-to=login]");
    await page.click("#to-forgot"); await page.waitForTimeout(300);
    await shot(page, `02-forgot-password-${suffix}`);
    await page.goto(base + "/#set-password=" + "x".repeat(43)); await page.reload(); await page.waitForTimeout(300);
    await shot(page, `03-set-password-${suffix}`);
  }

  await login(context, "u_cds_stu");
  await page.goto(base + "/");
  await shot(page, `04-chat-welcome-${suffix}`);
  if (viewport.width < 600) await page.click("#menu");
  await page.locator(".chat-item").nth(1).click();
  await page.waitForTimeout(300);
  await shot(page, `05-chat-answer-${suffix}`);
  if (viewport.width < 600) {
    await page.click("#menu"); await shot(page, `06-chat-sidebar-${suffix}`);
    await page.click("#hide-side"); await shot(page, `07-chat-after-${suffix}`);
  } else {
    await page.click("#me-btn"); await shot(page, `06-chat-menu-${suffix}`);
    await page.click("#settings"); await shot(page, `07-settings-${suffix}`);
    await page.keyboard.press("Escape");
    await page.click("#temp-chat"); await shot(page, `08-temporary-${suffix}`);
    await page.goto(base + "/admin"); await shot(page, `09-admin-denied-${suffix}`);
  }

  if (viewport.width >= 600) {
    const admin = await browser.newContext({ viewport, colorScheme: theme });
    await login(admin, "u_admin");
    const a = await admin.newPage();
    a.on("pageerror", (e) => errors.push(`SCRIPT ERROR ${a.url()}: ${e.message}`));
    for (const t of ["overview", "people", "requests", "documents", "grafana", "requirements", "models"]) {
      await a.goto(`${base}/admin#${t}`);
      await a.waitForTimeout(700);
      if (t === "documents") { await a.click('[data-doc="d02"]'); await a.waitForTimeout(800); }
      if (t === "documents") { await shot(a, `10-admin-documents-open-${suffix}`); await a.click("#d-close"); await a.waitForTimeout(200); }
      await shot(a, `10-admin-${t}-${suffix}`, { full: true });
    }
    await a.goto(`${base}/admin#documents`); await a.waitForTimeout(700);
    await a.click("#u-open"); await a.fill("#u-doc", "d02"); await a.waitForTimeout(800);
    await shot(a, `14-admin-upload-taken-${suffix}`); await a.keyboard.press("Escape");
    await a.goto(`${base}/admin#people`); await a.waitForTimeout(700);
    await a.click("[data-menu]"); await shot(a, `13-admin-people-menu-${suffix}`); await a.keyboard.press("Escape");
    await a.click("#add-open"); await a.fill("#pf-year", "2025"); await shot(a, `11-admin-add-person-${suffix}`);
    await a.selectOption("#pf-role", "faculty"); await a.selectOption("#pf-kind", "visiting");
    await shot(a, `11-admin-add-faculty-${suffix}`);
    await a.keyboard.press("Escape");
    await a.click("#bulk-open");
    await a.fill("#b-text", "user_id,role,program,joined,kind\nb1@iisc.example,student,phd,2025,\nb2@iisc.example,faculty,,,visiting\n");
    await a.click("#b-check"); await shot(a, `12-admin-bulk-${suffix}`);
    await admin.close();
  }
  await context.close();
  return errors;
}

(async () => {
  const browser = await chromium.launch();
  const errors = [];
  errors.push(...await run(browser, "light", { width: 1440, height: 900 }));
  errors.push(...await run(browser, "dark", { width: 1440, height: 900 }));
  errors.push(...await run(browser, "light", { width: 390, height: 844 }));
  await browser.close();
  console.log(errors.length ? `PAGE ERRORS:\n${errors.join("\n")}` : "no page errors");
  // A script error breaks the page; failed requests (401 before sign-in) do not.
  if (errors.some((e) => e.startsWith("SCRIPT ERROR"))) { console.error("FAILED: script errors"); process.exitCode = 1; }
})().catch((e) => { console.error("FAILED", e); process.exit(1); });
