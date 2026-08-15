#!/usr/bin/env python3
"""Exercise the browser product through Chrome DevTools Protocol."""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageStat
from websockets.sync.client import connect


class CDP:
    def __init__(self, url: str):
        self.ws = connect(url, origin="http://127.0.0.1:9223",
                          max_size=None, open_timeout=10)
        self.seq = 0
        self.events: list[dict] = []

    def call(self, method: str, params: dict | None = None):
        self.seq += 1
        wanted = self.seq
        self.ws.send(json.dumps({"id": wanted, "method": method,
                                 "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv(timeout=180))
            if msg.get("id") == wanted:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})
            self.events.append(msg)

    def evaluate(self, expression: str, await_promise: bool = False):
        out = self.call("Runtime.evaluate", {
            "expression": expression,
            "awaitPromise": await_promise,
            "returnByValue": True,
        })
        result = out.get("result", {})
        if result.get("subtype") == "error" or "exceptionDetails" in out:
            raise RuntimeError(out.get("exceptionDetails") or result)
        return result.get("value")

    def close(self):
        self.ws.close()


def devtools_page(deadline: float):
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:9223/json",
                                        timeout=1) as response:
                pages = json.load(response)
            page = next(p for p in pages if p.get("type") == "page")
            return page["webSocketDebuggerUrl"]
        except Exception:  # Chrome is still booting
            time.sleep(0.1)
    raise TimeoutError("Chrome DevTools did not start")


LAYOUT = r"""
(() => {
  const f=document.querySelector('#viserFrame');
  return {
    title: document.title,
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    overflowPx: document.documentElement.scrollWidth-innerWidth,
    iframeWidth: f?.contentWindow?.innerWidth ?? null,
    iframeScrollWidth: f?.contentDocument?.documentElement?.scrollWidth ?? null,
    iframeRect: f?.getBoundingClientRect().toJSON() ?? null,
    libraryCards: document.querySelectorAll('#libList .card').length,
    bodies: document.querySelectorAll('#body option').length,
    viewportMessage: document.querySelector('#vpMsg')?.textContent || '',
  };
})()
"""


INTERACTION = r"""
(async () => {
  const sleep=ms=>new Promise(r=>setTimeout(r,ms));
  const wait=async(fn,ms=120000)=>{const end=Date.now()+ms;while(Date.now()<end){const v=fn();if(v)return v;await sleep(100)}throw new Error('browser acceptance timeout')};
  const input=(el,v)=>{el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}))};
  await wait(()=>document.querySelectorAll('#libList .card').length>=2 && document.querySelectorAll('#body option').length>0,15000);
  const cards=[...document.querySelectorAll('#libList .card')];
  cards[0].click();
  cards[1].click();
  document.querySelectorAll('#tl .blk')[0].click();
  document.querySelector('#addGap').click();
  await sleep(50);
  input(document.querySelector('#segN'),'18');
  input(document.querySelector('#segSeed'),'7');
  input(document.querySelector('#segTemp'),'0.8');
  input(document.querySelector('#segPins'),'4:120');
  document.querySelector('#addSe3').click();
  await sleep(50);
  const joint=document.querySelector('.constraint select[data-k="joint"]');
  const jointLabel=joint.options[joint.selectedIndex].textContent;
  if (/pelvis/i.test(jointLabel)) throw new Error('SE3 defaulted to pelvis');
  input(document.querySelector('.constraint input[data-k="frame_start"]'),'65');
  input(document.querySelector('.constraint input[data-k="frame_end"]'),'75');
  input(document.querySelector('.constraint input[data-k="delta_m"][data-a="0"]'),'0.01');
  document.querySelector('#renderMp4').checked=false;
  document.querySelector('#compile').click();
  await wait(()=>document.querySelector('#compile').disabled,5000);
    await wait(()=>!document.querySelector('#compile').disabled && /done/.test(document.querySelector('#jobLog').textContent),120000);
  const resultText=document.querySelector('#result').innerText;
  if (!/continuity\s+PASS/i.test(resultText)) throw new Error('continuity did not pass: '+resultText);

  document.querySelector('[data-v="atlas"]').click();
  await wait(()=>document.querySelector('#atlasMotions .card'),10000);
  let bridge=null;
  for (const card of [...document.querySelectorAll('#atlasMotions .card')]) {
    document.querySelector('#portals').innerHTML='';
    card.click();
    await wait(()=>document.querySelector('#portals').children.length || /no portals/.test(document.querySelector('#portals').textContent),10000);
    bridge=document.querySelector('#portals .portal button');
    if (bridge) break;
  }
  const playable=!!bridge;
  if (bridge) {
    bridge.click();
    await wait(()=>/^(done|QC FLAG)/.test(document.querySelector('#atlasJob').textContent),120000);
  }

  document.querySelector('[data-v="bodies"]').click();
  await wait(()=>document.querySelector('#bodyList .card'),10000);
  document.querySelector('#bodyList .card').click();
  await wait(()=>document.querySelector('#bodyDetail').textContent.trim(),10000);
  await wait(()=>document.querySelectorAll('#semantics > div').length>0,10000);
  await wait(()=>document.querySelector('#bodyVpMsg').style.display==='none',10000);
  return {
    jointLabel,
    timelineFrames: 138,
    resultText,
    playablePortal: playable,
    portalStatus: document.querySelector('#atlasJob').textContent,
    bodyDetail: document.querySelector('#bodyDetail').textContent,
    semanticRows: document.querySelectorAll('#semantics > div').length/2,
    budgetSlider: !!document.querySelector('#segRange'),
    overflowPx: document.documentElement.scrollWidth-innerWidth,
  };
})()
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:7860")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--layout-only", action="store_true")
    parser.add_argument("--out", default="artifacts/browser_acceptance.json")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    chrome = (shutil.which("google-chrome")
              or "/opt/google/chrome/google-chrome")
    # Chrome helper processes can release files a few milliseconds after the
    # parent exits. Cleanup is best-effort test hygiene, not an acceptance
    # result; ignore that platform race after the browser has been reaped.
    with tempfile.TemporaryDirectory(
            prefix="alphamotion-chrome-", ignore_cleanup_errors=True) as profile:
        proc = subprocess.Popen([
            chrome, "--headless=new", "--no-sandbox",
            "--disable-dev-shm-usage", "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader", "--remote-debugging-port=9223",
            "--remote-allow-origins=*", f"--user-data-dir={profile}",
            f"--window-size={args.width},{args.height}", args.url,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cdp = None
        try:
            cdp = CDP(devtools_page(time.monotonic() + 15))
            cdp.call("Runtime.enable")
            if args.width < 500:
                cdp.call("Emulation.setDeviceMetricsOverride", {
                    "width": args.width, "height": args.height,
                    "deviceScaleFactor": 1, "mobile": True,
                })
                cdp.call("Page.reload", {"ignoreCache": True})
            time.sleep(4)
            layout = cdp.evaluate(LAYOUT)
            assert layout["libraryCards"] > 0 and layout["bodies"] > 0, layout
            assert layout["overflowPx"] <= 1, layout
            shot = cdp.call("Page.captureScreenshot", {
                "format": "png", "fromSurface": True})["data"]
            image = Image.open(io.BytesIO(base64.b64decode(shot))).convert("RGB")
            image.save(out.with_name(out.stem + "_full.png"))
            rect = layout["iframeRect"]
            box = (max(0, round(rect["x"])), max(0, round(rect["y"])),
                   min(image.width, round(rect["right"])),
                   min(image.height, round(rect["bottom"])))
            viewport = image.crop(box)
            viewport_path = out.with_name(out.stem + "_viewport.png")
            viewport.save(viewport_path)
            deviation = max(ImageStat.Stat(viewport).stddev)
            # This catches the previous product regression where the scene
            # tree and APIs were alive but the camera looked away from every
            # robot, leaving a uniformly black 3-D canvas.
            assert deviation > 5.0, {
                "viewport_stddev": deviation, "viewport": str(viewport_path)}
            layout["viewportPixelStddev"] = round(deviation, 3)
            # Dynamic camera/mesh stability gate. Capture a short burst while
            # playback is active; full-canvas flashes or camera jumps produce
            # large luminance/black-area discontinuities, unlike articulated
            # motion confined to the robot silhouette.
            burst = []
            for _ in range(12):
                raw = cdp.call("Page.captureScreenshot", {
                    "format": "png", "fromSurface": True})["data"]
                frame = Image.open(io.BytesIO(base64.b64decode(raw))).convert(
                    "RGB").crop(box)
                burst.append(frame)
                time.sleep(0.10)
            means = [sum(ImageStat.Stat(im).mean) / 3 for im in burst]
            black = [float((np.asarray(im.resize((160, 90))).max(axis=2)
                            < 8).mean()) for im in burst]
            diffs = [sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3
                     for a, b in zip(burst, burst[1:])]
            max_mean_delta = max(abs(b - a)
                                 for a, b in zip(means, means[1:]))
            max_black_delta = max(abs(b - a)
                                  for a, b in zip(black, black[1:]))
            assert max_mean_delta < 28.0, {
                "camera_flash_mean_delta": max_mean_delta}
            assert max_black_delta < 0.18, {
                "camera_flash_black_delta": max_black_delta}
            contact = Image.new("RGB", (burst[0].width * 4,
                                         burst[0].height * 3))
            for i, frame in enumerate(burst):
                contact.paste(frame, ((i % 4) * frame.width,
                                      (i // 4) * frame.height))
            contact.save(out.with_name(out.stem + "_stability.png"))
            layout["stability"] = {
                "frames": len(burst),
                "maxMeanDelta": round(max_mean_delta, 3),
                "maxBlackAreaDelta": round(max_black_delta, 4),
                "maxFrameDiff": round(max(diffs), 3),
            }
            result = None if args.layout_only else cdp.evaluate(
                INTERACTION, await_promise=True)
            if result is not None:
                cdp.evaluate(r"""
                    (async () => {
                      const frame=document.querySelector('#bodyFrame');
                      frame.scrollIntoView({block:'center'});
                      await new Promise(r=>setTimeout(r,300));
                      return true;
                    })()
                """, await_promise=True)
                body_shot = cdp.call("Page.captureScreenshot", {
                    "format": "png", "fromSurface": True})["data"]
                body_image = Image.open(io.BytesIO(
                    base64.b64decode(body_shot))).convert("RGB")
                body_image.save(out.with_name(out.stem + "_body.png"))
                body_rect = cdp.evaluate(
                    "document.querySelector('#bodyFrame').getBoundingClientRect().toJSON()")
                body_box = (max(0, round(body_rect["x"])),
                            max(0, round(body_rect["y"])),
                            min(body_image.width, round(body_rect["right"])),
                            min(body_image.height, round(body_rect["bottom"])))
                body_crop = body_image.crop(body_box)
                body_std = max(ImageStat.Stat(body_crop).stddev)
                assert body_std > 5.0, {"body_preview_stddev": body_std}
                result["bodyPreviewPixelStddev"] = round(body_std, 3)
                cdp.evaluate(r"""
                    (async () => {
                      document.querySelector('[data-v="atlas"]').click();
                      const sleep=ms=>new Promise(r=>setTimeout(r,ms));
                      const end=Date.now()+10000;
                      while(Date.now()<end){
                        const c=document.querySelector('#atlasGraph');
                        if(c && c.offsetWidth>0 && c.offsetHeight>0){
                          c.scrollIntoView({block:'center'});
                          await sleep(300);
                          return true;
                        }
                        await sleep(100);
                      }
                      throw new Error('Atlas graph did not become visible');
                    })()
                """, await_promise=True)
                atlas_shot = cdp.call("Page.captureScreenshot", {
                    "format": "png", "fromSurface": True})["data"]
                atlas_image = Image.open(io.BytesIO(
                    base64.b64decode(atlas_shot))).convert("RGB")
                atlas_image.save(out.with_name(out.stem + "_atlas.png"))
                atlas_rect = cdp.evaluate(
                    "document.querySelector('#atlasGraph').getBoundingClientRect().toJSON()")
                atlas_box = (max(0, round(atlas_rect["x"])),
                             max(0, round(atlas_rect["y"])),
                             min(atlas_image.width, round(atlas_rect["right"])),
                             min(atlas_image.height, round(atlas_rect["bottom"])))
                atlas_crop = atlas_image.crop(atlas_box)
                atlas_std = max(ImageStat.Stat(atlas_crop).stddev)
                assert atlas_std > 5.0, {"atlas_graph_stddev": atlas_std}
                result["atlasGraphPixelStddev"] = round(atlas_std, 3)
            errors = [e for e in cdp.events
                      if e.get("method") == "Runtime.exceptionThrown"]
            assert not errors, errors
            report = {"layout": layout, "interaction": result,
                      "runtime_exceptions": errors}
        finally:
            if cdp is not None:
                cdp.close()
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {out}")


if __name__ == "__main__":
    main()
