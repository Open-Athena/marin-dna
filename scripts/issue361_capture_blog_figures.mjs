#!/usr/bin/env node

import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const WebSocket = require("ws");

const FIGURE_IDS = [
  "fig-training-datasets",
  "fig-evaluation-datasets",
  "fig-evaluation-readouts",
  "fig-upstream-cds-specialists",
  "fig-upstream-cds-balance",
  "fig-annotation-derived-training-pool",
  "fig-hyperparameter-transfer-methodology",
  "fig-learning-rate-transfer",
  "fig-adam-transfer",
  "fig-region-hyperparameter-transfer",
  "fig-loss-scaling",
  "fig-parameters-vs-vep",
  "fig-loss-vs-vep",
  "fig-missense-readout-scaling",
  "fig-five-region-lineage",
  "fig-mixture-lineage-trajectories",
  "fig-mixture-lineage-probe",
  "fig-mendelian-leaderboard",
  "fig-mendelian-leaderboard-probe",
];

const [url, outputDirectory, rawWidth = "1440", rawHeight = "1100"] =
  process.argv.slice(2);
assert(url && outputDirectory, "usage: capture-blog-figures URL OUTPUT_DIR [WIDTH HEIGHT]");
const width = Number(rawWidth);
const height = Number(rawHeight);
assert(Number.isInteger(width) && width > 0, rawWidth);
assert(Number.isInteger(height) && height > 0, rawHeight);

const chrome = process.env.CHROME_BIN;
assert(chrome && fs.existsSync(chrome), "set CHROME_BIN to a Chromium executable");
fs.mkdirSync(outputDirectory, { recursive: true });
const auditResults = [];

const debuggingPort = 9333;
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "issue361-chrome-"));
const browser = spawn(
  chrome,
  [
    "--headless=new",
    "--no-sandbox",
    "--disable-gpu",
    "--disable-breakpad",
    "--disable-component-update",
    "--disable-crash-reporter",
    "--hide-scrollbars",
    "--no-first-run",
    `--remote-debugging-port=${debuggingPort}`,
    `--user-data-dir=${profile}`,
    "about:blank",
  ],
  { detached: true, stdio: "ignore" },
);

function killBrowserGroup(signal) {
  try {
    process.kill(-browser.pid, signal);
  } catch (error) {
    if (error.code !== "ESRCH") throw error;
  }
}

async function pageTarget() {
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const targets = await fetch(`http://127.0.0.1:${debuggingPort}/json/list`).then(
        (response) => response.json(),
      );
      const page = targets.find((target) => target.type === "page");
      if (page) return page;
    } catch (_) {
      // Chrome is still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Chrome debugging endpoint did not start");
}

const target = await pageTarget();
const socket = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((resolve, reject) => {
  socket.once("open", resolve);
  socket.once("error", reject);
});

let nextId = 1;
const pending = new Map();
socket.on("message", (rawMessage) => {
  const message = JSON.parse(rawMessage.toString());
  if (!message.id || !pending.has(message.id)) return;
  const { resolve, reject } = pending.get(message.id);
  pending.delete(message.id);
  if (message.error) reject(new Error(JSON.stringify(message.error)));
  else resolve(message.result);
});

function command(method, params = {}) {
  const id = nextId;
  nextId += 1;
  socket.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

try {
  await command("Page.enable");
  await command("Runtime.enable");
  await command("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: width <= 600,
  });
  await command("Page.navigate", { url });
  await new Promise((resolve) => setTimeout(resolve, 3500));
  await command("Runtime.evaluate", {
    expression: "document.fonts.ready",
    awaitPromise: true,
  });

  for (const [index, figureId] of FIGURE_IDS.entries()) {
    const evaluated = await command("Runtime.evaluate", {
      expression: `(() => {
        const figure = document.getElementById(${JSON.stringify(figureId)});
        if (!figure) return null;
        const originalRect = figure.getBoundingClientRect();
        const placeholder = document.createComment('issue361-figure-placeholder');
        figure.replaceWith(placeholder);
        const overlay = document.createElement('div');
        overlay.className = 'blog-post-content';
        overlay.style.cssText = [
          'position: fixed',
          'inset: 0',
          'z-index: 2147483647',
          'box-sizing: border-box',
          'padding: 16px',
          'max-width: none',
          'overflow: hidden',
          'background: var(--bg)',
        ].join(';');
        figure.style.width = originalRect.width + 'px';
        overlay.appendChild(figure);
        document.body.appendChild(overlay);
        const rect = figure.getBoundingClientRect();
        const svg = figure.querySelector('svg.figure-svg');
        const text = svg?.querySelector('text');
        const svgRect = svg?.getBoundingClientRect();
        const screenBox = (element) => {
          const box = element.getBBox();
          const matrix = element.getScreenCTM();
          const points = [
            new DOMPoint(box.x, box.y),
            new DOMPoint(box.x + box.width, box.y),
            new DOMPoint(box.x, box.y + box.height),
            new DOMPoint(box.x + box.width, box.y + box.height),
          ].map((point) => point.matrixTransform(matrix));
          const xs = points.map((point) => point.x);
          const ys = points.map((point) => point.y);
          return {
            left: Math.min(...xs),
            right: Math.max(...xs),
            top: Math.min(...ys),
            bottom: Math.max(...ys),
            width: Math.max(...xs) - Math.min(...xs),
            height: Math.max(...ys) - Math.min(...ys),
          };
        };
        const textBox = (element) => {
          const spans = [...element.querySelectorAll('tspan')]
            .filter((span) => span.textContent.trim());
          if (spans.length === 0) return screenBox(element);
          const boxes = spans.map(screenBox);
          return {
            left: Math.min(...boxes.map((box) => box.left)),
            right: Math.max(...boxes.map((box) => box.right)),
            top: Math.min(...boxes.map((box) => box.top)),
            bottom: Math.max(...boxes.map((box) => box.bottom)),
            width: Math.max(...boxes.map((box) => box.right)) -
              Math.min(...boxes.map((box) => box.left)),
            height: Math.max(...boxes.map((box) => box.bottom)) -
              Math.min(...boxes.map((box) => box.top)),
          };
        };
        const visibleText = [...(svg?.querySelectorAll('text') ?? [])]
          .filter((element) => {
            const style = getComputedStyle(element);
            const box = textBox(element);
            return style.display !== 'none' && style.visibility !== 'hidden' &&
              Number(style.opacity) !== 0 && box.width > 0 && box.height > 0 &&
              element.textContent.trim();
          })
          .map((element, index) => {
            const box = textBox(element);
            return {
              element,
              index,
              label: element.textContent.replace(/\s+/g, ' ').trim().slice(0, 80),
              box,
              rotated: element.getAttribute('transform')?.includes('rotate(') ||
                Boolean(element.closest('g[transform*="rotate("]')),
            };
          });
        const textOverlaps = [];
        for (let left = 0; left < visibleText.length; left += 1) {
          for (let right = left + 1; right < visibleText.length; right += 1) {
            const first = visibleText[left];
            const second = visibleText[right];
            if (first.rotated || second.rotated) continue;
            const overlapWidth = Math.min(first.box.right, second.box.right) -
              Math.max(first.box.left, second.box.left);
            const overlapHeight = Math.min(first.box.bottom, second.box.bottom) -
              Math.max(first.box.top, second.box.top);
            if (overlapWidth > 4 && overlapHeight > 4) {
              textOverlaps.push({
                first: first.label,
                second: second.label,
                overlapWidth: Number(overlapWidth.toFixed(1)),
                overlapHeight: Number(overlapHeight.toFixed(1)),
              });
            }
          }
        }
        const pillOverflows = [];
        for (const pill of svg?.querySelectorAll('rect[rx]') ?? []) {
          const height = Number(pill.getAttribute('height'));
          if (!(height >= 18 && height <= 50)) continue;
          const pillBox = screenBox(pill);
          for (const item of visibleText) {
            const centerX = (item.box.left + item.box.right) / 2;
            const centerY = (item.box.top + item.box.bottom) / 2;
            if (centerX < pillBox.left || centerX > pillBox.right ||
                centerY < pillBox.top || centerY > pillBox.bottom) continue;
            const overflow = {
              left: pillBox.left - item.box.left,
              right: item.box.right - pillBox.right,
              top: pillBox.top - item.box.top,
              bottom: item.box.bottom - pillBox.bottom,
            };
            if (Math.max(...Object.values(overflow)) > 3.5) {
              pillOverflows.push({
                label: item.label,
                overflow: Object.fromEntries(
                  Object.entries(overflow).map(([key, value]) => [
                    key,
                    Number(Math.max(0, value).toFixed(1)),
                  ]),
                ),
              });
            }
          }
        }
        const svgOverflows = svgRect ? visibleText
          .filter((item) => item.box.left < svgRect.left - 1 ||
            item.box.right > svgRect.right + 1 ||
            item.box.top < svgRect.top - 1 ||
            item.box.bottom > svgRect.bottom + 1)
          .map((item) => item.label) : [];
        const frameBox = figure.querySelector('.figure-frame')?.getBoundingClientRect();
        const captionBox = figure.querySelector('figcaption')?.getBoundingClientRect();
        const captionFrameOverlap = frameBox && captionBox
          ? Math.max(0, Math.min(frameBox.bottom, captionBox.bottom) -
              Math.max(frameBox.top, captionBox.top))
          : 0;
        window.__issue361FigureCapture = { figure, placeholder, overlay };
        return {
          x: rect.left,
          y: rect.top,
          width: rect.width,
          height: rect.height,
          inlined: Boolean(figure.querySelector('svg.figure-svg')),
          family: text ? getComputedStyle(text).fontFamily : null,
          textOverlaps,
          pillOverflows,
          svgOverflows,
          captionFrameOverlap: Number(captionFrameOverlap.toFixed(1)),
        };
      })()`,
      returnByValue: true,
    });
    const figure = evaluated.result.value;
    assert(figure, `missing #${figureId}`);
    assert(figure.inlined, `SVG was not inlined for #${figureId}`);
    assert(figure.family.toLowerCase().includes("lato"), figure.family);
    auditResults.push({
      figureId,
      width: Number(figure.width.toFixed(1)),
      textOverlaps: figure.textOverlaps,
      pillOverflows: figure.pillOverflows,
      svgOverflows: figure.svgOverflows,
      captionFrameOverlap: figure.captionFrameOverlap,
    });

    const padding = 16;
    const screenshot = await command("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
      fromSurface: true,
      clip: {
        x: Math.max(0, figure.x - padding),
        y: Math.max(0, figure.y - padding),
        width: Math.min(width, figure.width + 2 * padding),
        height: figure.height + 2 * padding,
        scale: 1,
      },
    });
    const filename = `${String(index + 1).padStart(2, "0")}-${figureId}.png`;
    fs.writeFileSync(path.join(outputDirectory, filename), screenshot.data, "base64");
    console.log(
      `${filename}\t${figure.family}\t` +
        `text=${figure.textOverlaps.length} pills=${figure.pillOverflows.length} ` +
        `bounds=${figure.svgOverflows.length} caption=${figure.captionFrameOverlap}`,
    );
    await command("Runtime.evaluate", {
      expression: `(() => {
        const state = window.__issue361FigureCapture;
        state.placeholder.replaceWith(state.figure);
        state.figure.style.removeProperty('width');
        state.overlay.remove();
        delete window.__issue361FigureCapture;
      })()`,
    });
  }
  fs.writeFileSync(
    path.join(outputDirectory, "layout-audit.json"),
    `${JSON.stringify(auditResults, null, 2)}\n`,
  );
  const failures = auditResults.filter(
    (result) =>
      result.textOverlaps.length > 0 ||
      result.pillOverflows.length > 0 ||
      result.svgOverflows.length > 0 ||
      result.captionFrameOverlap > 0,
  );
  assert.equal(
    failures.length,
    0,
    `layout collisions detected in ${failures.map((result) => result.figureId).join(", ")}`,
  );
} finally {
  socket.close();
  const exited = new Promise((resolve) => browser.once("exit", resolve));
  killBrowserGroup("SIGTERM");
  const stopped = await Promise.race([
    exited.then(() => true),
    new Promise((resolve) => setTimeout(() => resolve(false), 2000)),
  ]);
  if (!stopped) {
    killBrowserGroup("SIGKILL");
    await Promise.race([
      exited,
      new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
  }
  try {
    fs.rmSync(profile, {
      recursive: true,
      force: true,
      maxRetries: 10,
      retryDelay: 100,
    });
  } catch (error) {
    console.warn(`Could not remove temporary Chrome profile ${profile}: ${error}`);
  }
}
