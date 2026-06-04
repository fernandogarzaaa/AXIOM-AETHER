#!/usr/bin/env python3
"""Axiom Training Dashboard — a tiny, dependency-free live view of training.

Run:  python scripts/dashboard.py
Then open the URL it prints (default http://127.0.0.1:7070).

It reads logs/train_d384.log and shows, in plain language, how training is going:
a single "smartness" curve, how far along it is, and a rough time remaining.
No installs, no internet — Python standard library only.
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys

PORT = int(os.environ.get("AXIOM_DASHBOARD_PORT", "7070"))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_LOG = os.path.join(REPO, "logs", "train_d384.log")
PROMOTE_LOG = os.path.join(REPO, "logs", "promote_d384.log")

STEP_CAP = 4000           # the run's hard step ceiling (upper bound for ETA)
D256_TARGET = 3.2         # the score d384 must beat (d256 reference)
START_LOSS = 9.68         # theoretical "fully confused" for a 16k-vocab model

STEP_RE = re.compile(r"step\s+(\d+)\s+loss~([\d.]+)\s+\((\d+)s\)")
EPOCH_RE = re.compile(r"epoch\s+(\d+)\s+train_loss=([\d.]+)\s+val_ce=([\d.]+)")


def read_points():
    """Parse the training log into a list of {step, loss, secs} points."""
    pts, epochs = [], []
    try:
        with open(TRAIN_LOG, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = STEP_RE.search(line)
                if m:
                    pts.append({"step": int(m.group(1)),
                                "loss": float(m.group(2)),
                                "secs": int(m.group(3))})
                e = EPOCH_RE.search(line)
                if e:
                    epochs.append({"epoch": int(e.group(1)),
                                   "val_ce": float(e.group(3))})
    except FileNotFoundError:
        pass
    return pts, epochs


def is_training():
    """True if the train_semantic process is currently running."""
    try:
        out = subprocess.run(["tasklist"], capture_output=True, text=True, timeout=5).stdout.lower()
        return "train_semantic" in out
    except Exception:
        return False


def gpu_stats():
    """Live GPU utilization / VRAM / temp via nvidia-smi. Changes every poll, so
    it gives real-time proof the GPU is working even between score updates."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        util, used, total, temp = [x.strip() for x in out.split(",")]
        return {"util": int(util), "vram_used": int(used),
                "vram_total": int(total), "temp": int(temp)}
    except Exception:
        return None


def promotion_state():
    """Read the promote watcher log for a final verdict, if any."""
    try:
        with open(PROMOTE_LOG, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except FileNotFoundError:
        return None
    if "d384 is live in production" in txt:
        return "promoted"
    if "FAILED acceptance" in txt or "was never written" in txt:
        return "failed"
    return None


def build_status():
    pts, epochs = read_points()
    running = is_training()
    promo = promotion_state()

    cur = pts[-1] if pts else None
    cur_loss = cur["loss"] if cur else START_LOSS
    cur_step = cur["step"] if cur else 0
    elapsed_s = cur["secs"] if cur else 0

    # Progress: how far from "fully confused" to "beat d256", clamped 0..100.
    span = max(START_LOSS - D256_TARGET, 1e-6)
    progress = max(0.0, min(1.0, (START_LOSS - cur_loss) / span))

    # Rough ETA to the step cap, from the average seconds-per-step so far.
    eta_text = "estimating…"
    if cur and cur_step > 0 and elapsed_s > 0:
        sps = elapsed_s / cur_step
        remaining = max(0, STEP_CAP - cur_step)
        eta_h = (remaining * sps) / 3600.0
        if eta_h >= 1:
            eta_text = f"up to ~{eta_h:.1f} hours"
        else:
            eta_text = f"up to ~{eta_h * 60:.0f} minutes"

    # Friendly headline.
    if promo == "promoted":
        headline, sub, mood = "🎉 New model is live!", \
            "Axiom passed its test and the upgraded brain is now in production.", "done-good"
    elif promo == "failed":
        headline, sub, mood = "Kept the proven model", \
            "The new version didn't beat the current one, so nothing changed. You lost nothing.", "done-neutral"
    elif not running and pts:
        headline, sub, mood = "Training finished", \
            "Checking the result now…", "done-neutral"
    elif cur_loss <= D256_TARGET:
        headline, sub, mood = "🔥 Beating the target!", \
            "Axiom is now scoring better than the current model.", "running"
    elif cur_step < 100:
        headline, sub, mood = "Warming up…", \
            "Axiom is just getting started.", "running"
    else:
        headline, sub, mood = "🧠 Axiom is learning", \
            "It's steadily getting smarter. Lower score = smarter.", "running"

    return {
        "running": running,
        "promo": promo,
        "headline": headline,
        "sub": sub,
        "mood": mood,
        "cur_loss": round(cur_loss, 3),
        "cur_step": cur_step,
        "step_cap": STEP_CAP,
        "elapsed_min": round(elapsed_s / 60.0, 1),
        "progress_pct": round(progress * 100, 1),
        "eta": eta_text,
        "target": D256_TARGET,
        "start": START_LOSS,
        "points": pts[-200:],   # cap payload
        "epochs": epochs,
        "gpu": gpu_stats(),
    }


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Axiom — Training</title>
<style>
:root{--bg:#0b0e14;--panel:#141925;--ink:#e8edf6;--dim:#8a93a6;--good:#4ade80;--accent:#6ea8ff;--warn:#fbbf24}
*{box-sizing:border-box}
body{margin:0;font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:
radial-gradient(1200px 700px at 70% -10%,#16203a 0%,var(--bg) 55%);color:var(--ink);min-height:100vh}
.wrap{max-width:760px;margin:0 auto;padding:32px 20px 64px}
.head{display:flex;align-items:center;gap:12px;margin-bottom:4px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--good);box-shadow:0 0 0 0 rgba(74,222,128,.6);animation:pulse 2s infinite}
.dot.idle{background:var(--dim);animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 12px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
h1{font-size:30px;margin:6px 0 2px;letter-spacing:-.02em}
.sub{color:var(--dim);font-size:17px;margin-bottom:24px}
.card{background:linear-gradient(180deg,var(--panel),#10141f);border:1px solid #1f2940;border-radius:18px;padding:22px;margin-bottom:18px}
.score{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.score .big{font-size:54px;font-weight:700;letter-spacing:-.03em}
.score .lbl{color:var(--dim)}
.bar{height:14px;background:#0c111b;border-radius:99px;overflow:hidden;margin:18px 0 8px;border:1px solid #1f2940}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--good));border-radius:99px;transition:width .6s ease}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:8px}
.stat{background:#0c111b;border:1px solid #1f2940;border-radius:14px;padding:14px}
.stat b{display:block;font-size:22px}
.stat span{color:var(--dim);font-size:13px}
svg{width:100%;height:200px;display:block}
.legend{display:flex;gap:18px;color:var(--dim);font-size:13px;margin-top:8px;flex-wrap:wrap}
.legend i{display:inline-block;width:18px;height:3px;vertical-align:middle;margin-right:6px;border-radius:2px}
.foot{color:var(--dim);font-size:13px;text-align:center;margin-top:18px}
</style></head><body><div class=wrap>
<div class=head><span id=dot class=dot></span><span id=state style="color:var(--dim);font-size:14px">live</span></div>
<h1 id=headline>Loading…</h1>
<div class=sub id=subtext></div>

<div class=card>
  <div class=score><span class=big id=loss>—</span><span class=lbl>confusion score &nbsp;<small>(lower = smarter)</small></span></div>
  <div class=bar><i id=fill style="width:0%"></i></div>
  <div style="color:var(--dim);font-size:13px">Goal: get below <b id=target>3.2</b> to beat the current model</div>
  <div class=grid>
    <div class=stat><b id=prog>0%</b><span>of the way to the goal</span></div>
    <div class=stat><b id=elapsed>0m</b><span>time training</span></div>
    <div class=stat><b id=eta>—</b><span>time remaining</span></div>
  </div>
</div>

<div class=card id=gpucard>
  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
    <span style="font-weight:600">Engine activity <small style="color:var(--dim)">— live, right now</small></span>
    <span id=gpuwork style="color:var(--good);font-size:13px">working</span>
  </div>
  <div class=bar><i id=gpufill style="width:0%;background:linear-gradient(90deg,#4ade80,#fbbf24)"></i></div>
  <div class=grid style="margin-top:14px">
    <div class=stat><b id=gpuutil>—</b><span>GPU effort</span></div>
    <div class=stat><b id=gpumem>—</b><span>memory in use</span></div>
    <div class=stat><b id=gputemp>—</b><span>temperature</span></div>
  </div>
</div>

<div class=card>
  <div id=chart></div>
  <div class=legend><span><i style="background:#6ea8ff"></i>Axiom's score over time</span>
  <span><i style="background:#fbbf24"></i>target to beat</span></div>
</div>
<div class=foot>Updates automatically every few seconds · safe to close anytime — training keeps running</div>
</div>
<script>
function fmt(n){return Math.round(n*10)/10}
function draw(p,target,start){
  if(!p.length){return '<div style="color:#8a93a6;padding:40px;text-align:center">waiting for the first data point…</div>'}
  const W=700,H=200,pad=28
  const steps=p.map(d=>d.step),loss=p.map(d=>d.loss)
  const xmin=Math.min(...steps),xmax=Math.max(...steps,xmin+1)
  const ymin=Math.min(target-0.3,...loss),ymax=Math.max(start,...loss)
  const X=s=>pad+(s-xmin)/(xmax-xmin)*(W-2*pad)
  const Y=l=>pad+(1-(l-ymin)/(ymax-ymin))*(H-2*pad)
  let d='';p.forEach((pt,i)=>{d+=(i?'L':'M')+X(pt.step).toFixed(1)+' '+Y(pt.loss).toFixed(1)+' '})
  const ty=Y(target).toFixed(1)
  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio=none>
   <line x1=${pad} y1=${ty} x2=${W-pad} y2=${ty} stroke=#fbbf24 stroke-width=1.5 stroke-dasharray="5 5" opacity=.8/>
   <path d="${d}" fill=none stroke=#6ea8ff stroke-width=2.5 stroke-linejoin=round stroke-linecap=round/>
   <circle cx=${X(p[p.length-1].step).toFixed(1)} cy=${Y(p[p.length-1].loss).toFixed(1)} r=4 fill=#6ea8ff/>
  </svg>`
}
async function tick(){
 try{
  const s=await (await fetch('/data',{cache:'no-store'})).json()
  document.getElementById('headline').textContent=s.headline
  document.getElementById('subtext').textContent=s.sub
  document.getElementById('loss').textContent=fmt(s.cur_loss)
  document.getElementById('target').textContent=s.target
  document.getElementById('fill').style.width=s.progress_pct+'%'
  document.getElementById('prog').textContent=s.progress_pct+'%'
  document.getElementById('elapsed').textContent=s.elapsed_min>=60?(s.elapsed_min/60).toFixed(1)+'h':Math.round(s.elapsed_min)+'m'
  document.getElementById('eta').textContent=s.running?s.eta:'done'
  document.getElementById('chart').innerHTML=draw(s.points,s.target,s.start)
  const dot=document.getElementById('dot'),st=document.getElementById('state')
  if(s.running){dot.className='dot';st.textContent='training now'}else{dot.className='dot idle';st.textContent='not running'}
  const g=s.gpu,card=document.getElementById('gpucard')
  if(g){card.style.display='block'
   document.getElementById('gpuutil').textContent=g.util+'%'
   document.getElementById('gpumem').textContent=(g.vram_used/1024).toFixed(1)+' GB'
   document.getElementById('gputemp').textContent=g.temp+'°C'
   document.getElementById('gpufill').style.width=g.util+'%'
   const w=document.getElementById('gpuwork')
   if(g.util>=40){w.textContent='working hard';w.style.color='#4ade80'}
   else if(g.util>0){w.textContent='ticking over';w.style.color='#fbbf24'}
   else{w.textContent='idle';w.style.color='#8a93a6'}
  } else {card.style.display='none'}
 }catch(e){}
}
tick();setInterval(tick,3000)
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path.startswith("/data"):
            body = json.dumps(build_status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


def main():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        url = f"http://127.0.0.1:{PORT}"
        print(f"\n  Axiom dashboard running at  {url}\n  (Ctrl+C to stop. Training is unaffected.)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  dashboard stopped. training continues.\n")


if __name__ == "__main__":
    main()
