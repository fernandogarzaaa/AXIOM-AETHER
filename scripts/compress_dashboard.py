#!/usr/bin/env python3
"""Axiom Compression Dashboard — live control + monitor for the proxy.

Run:  python scripts/compress_dashboard.py
Then open the URL it prints (default http://127.0.0.1:7071).

Set compression Off / Low / Medium / High with one click and watch — live — how
much it's saving. Talks to the running proxy's /v1/config (no restart needed).
Python standard library only; no installs, no internet.
"""
import http.server
import json
import os
import socketserver
import urllib.request

PORT = int(os.environ.get("AXIOM_DASH_PORT", "7071"))
PROXY = os.environ.get("AXIOM_PROXY_URL", "http://127.0.0.1:3000").rstrip("/")


def proxy_get_config():
    try:
        with urllib.request.urlopen(PROXY + "/v1/config", timeout=4) as r:
            return json.load(r), None
    except Exception as e:
        return None, str(e)


def proxy_post_config(payload):
    try:
        req = urllib.request.Request(
            PROXY + "/v1/config", data=json.dumps(payload).encode(),
            method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=4) as r:
            return json.load(r), None
    except Exception as e:
        return None, str(e)


PAGE = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Axiom — Compression</title>
<style>
:root{--bg:#0b0e14;--panel:#141925;--ink:#e8edf6;--dim:#8a93a6;--good:#4ade80;--accent:#6ea8ff;--warn:#fbbf24;--off:#64748b}
*{box-sizing:border-box}
body{margin:0;font:16px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);min-height:100vh;
background:radial-gradient(1200px 700px at 70% -10%,#16203a 0%,var(--bg) 55%)}
.wrap{max-width:780px;margin:0 auto;padding:32px 20px 64px}
.head{display:flex;align-items:center;gap:12px}
.dot{width:11px;height:11px;border-radius:50%;background:var(--good);box-shadow:0 0 0 0 rgba(74,222,128,.6);animation:pulse 2s infinite}
.dot.off{background:var(--off);animation:none}
@keyframes pulse{0%{box-shadow:0 0 0 0 rgba(74,222,128,.5)}70%{box-shadow:0 0 0 12px rgba(74,222,128,0)}100%{box-shadow:0 0 0 0 rgba(74,222,128,0)}}
h1{font-size:30px;margin:8px 0 2px;letter-spacing:-.02em}
.sub{color:var(--dim);margin-bottom:24px}
.card{background:linear-gradient(180deg,var(--panel),#10141f);border:1px solid #1f2940;border-radius:18px;padding:22px;margin-bottom:18px}
.lbl{color:var(--dim);font-size:13px;text-transform:uppercase;letter-spacing:.05em;margin-bottom:12px}
.levels{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
.btn{padding:16px 8px;border-radius:14px;border:1px solid #28344f;background:#0c111b;color:var(--ink);
font-size:15px;font-weight:600;cursor:pointer;transition:all .15s;text-align:center}
.btn small{display:block;color:var(--dim);font-weight:400;font-size:11px;margin-top:3px}
.btn:hover{border-color:var(--accent);transform:translateY(-1px)}
.btn.active{background:linear-gradient(180deg,#1c2c4d,#16233d);border-color:var(--accent);box-shadow:0 0 0 1px var(--accent) inset}
.btn.active.off{border-color:var(--off);box-shadow:0 0 0 1px var(--off) inset;background:#161b26}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.stat{background:#0c111b;border:1px solid #1f2940;border-radius:14px;padding:16px}
.stat b{display:block;font-size:26px;letter-spacing:-.02em}
.stat span{color:var(--dim);font-size:12px}
.bar{height:16px;background:#0c111b;border-radius:99px;overflow:hidden;margin:6px 0 4px;border:1px solid #1f2940}
.bar>i{display:block;height:100%;background:linear-gradient(90deg,var(--accent),var(--good));border-radius:99px;transition:width .5s}
.row{display:flex;justify-content:space-between;align-items:baseline;margin-top:4px}
.big{font-size:46px;font-weight:700;letter-spacing:-.03em}
.foot{color:var(--dim);font-size:13px;text-align:center;margin-top:18px}
.warnbox{background:#2a2410;border:1px solid #5c4a16;color:#fde68a;border-radius:12px;padding:12px 14px;margin-bottom:18px;font-size:14px;display:none}
</style></head><body><div class=wrap>
<div class=head><span id=dot class=dot></span><span id=state style="color:var(--dim);font-size:14px">connecting…</span></div>
<h1>Axiom Compression</h1>
<div class=sub>Set how aggressively Axiom shrinks big context — and watch it work, live.</div>

<div class=warnbox id=warn></div>

<div class=card>
  <div class=lbl>Compression level</div>
  <div class=levels>
    <button class=btn data-level=off  onclick="setLevel('off')">Off<small>passthrough</small></button>
    <button class=btn data-level=low  onclick="setLevel('low')">Low<small>huge only</small></button>
    <button class=btn data-level=medium onclick="setLevel('medium')">Medium<small>large pastes</small></button>
    <button class=btn data-level=high onclick="setLevel('high')">High<small>aggressive</small></button>
  </div>
  <div class=row style="margin-top:14px;color:var(--dim);font-size:13px">
    <span>Compresses messages over <b id=thresh>—</b> words</span>
    <span id=activeflag></span>
  </div>
</div>

<div class=card>
  <div class=lbl>Savings so far</div>
  <div class=row><span class=big id=savings>—</span><span style="color:var(--dim)">smaller on the wire</span></div>
  <div class=bar><i id=savefill style="width:0%"></i></div>
  <div class=grid style="margin-top:14px">
    <div class=stat><b id=reqs>0</b><span>requests compressed</span></div>
    <div class=stat><b id=msgs>0</b><span>messages absorbed</span></div>
    <div class=stat><b id=saved>0</b><span>KB saved total</span></div>
  </div>
</div>
<div class=foot>Updates live every 2s · changes apply instantly, no restart · safe to close anytime</div>
</div>
<script>
async function setLevel(lvl){
  try{ await fetch('/set',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({level:lvl})}) }catch(e){}
  tick()
}
function fmtPct(n){return (Math.round(n*10)/10)+'%'}
async function tick(){
 try{
  const r=await fetch('/data',{cache:'no-store'}); const s=await r.json()
  const warn=document.getElementById('warn')
  if(s.error){ document.getElementById('state').textContent='proxy not reachable';
    document.getElementById('dot').className='dot off'; warn.style.display='block';
    warn.textContent='Cannot reach the Axiom proxy at '+s.proxy+'. Is it running?'; return }
  warn.style.display = s.forwarder_ready ? 'none':'block'
  if(!s.forwarder_ready){ warn.textContent='Proxy is running but compression forwarder is off (started with compression disabled). Restart the proxy with compression enabled to use levels.' }
  const on = s.enabled
  document.getElementById('dot').className = on?'dot':'dot off'
  document.getElementById('state').textContent = on?('compressing — '+s.level+' level'):'off (passthrough)'
  document.getElementById('thresh').textContent = s.threshold_tokens
  document.getElementById('activeflag').textContent = s.compression_active?'● live':'○ idle'
  document.querySelectorAll('.btn').forEach(b=>{
    const active = (s.enabled? b.dataset.level===s.level : b.dataset.level==='off')
    b.className = 'btn'+(active?' active':'')+(b.dataset.level==='off'&&active?' off':'')
  })
  const c=s.counters||{}
  document.getElementById('savings').textContent = (c.bytes_in>0)?fmtPct(c.savings_pct):'—'
  document.getElementById('savefill').style.width = Math.max(0,Math.min(100,c.savings_pct||0))+'%'
  document.getElementById('reqs').textContent = c.requests||0
  document.getElementById('msgs').textContent = c.messages_compressed||0
  document.getElementById('saved').textContent = Math.round(((c.bytes_in||0)-(c.bytes_out||0))/1024)
 }catch(e){ document.getElementById('state').textContent='dashboard error' }
}
tick(); setInterval(tick,2000)
</script></body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/data"):
            cfg, err = proxy_get_config()
            if err:
                self._send({"error": err, "proxy": PROXY})
            else:
                self._send(cfg)
        else:
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        if self.path.startswith("/set"):
            n = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            res, err = proxy_post_config(payload)
            self._send(res if not err else {"error": err}, 200 if not err else 502)
        else:
            self._send({"error": "not found"}, 404)


def main():
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        print(f"\n  Axiom compression dashboard at  http://127.0.0.1:{PORT}")
        print(f"  (controls the proxy at {PROXY}; Ctrl+C to stop)\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  dashboard stopped. proxy unaffected.\n")


if __name__ == "__main__":
    main()
