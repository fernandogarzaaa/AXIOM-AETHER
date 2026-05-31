#!/usr/bin/env node
// axiom-scrape.js — headless search-scraping worker for Axiom's JIT search node.
//
// Accepts a query, pulls raw text from a search engine (DuckDuckGo HTML
// endpoint — no API key), strips HTML/markdown clutter, and prints clean text
// to stdout. The Rust engine (src/search_ingest.rs) streams this into the BPE
// tokenizer + online TTT.
//
// Usage:   node axiom-scrape.js "your search query"
// Env:     AXIOM_SCRAPE_MAXCHARS (default 20000)  cap clean-text output
//          AXIOM_SCRAPE_RESULTS  (default 12)      max result snippets
'use strict';

const MAXCHARS = parseInt(process.env.AXIOM_SCRAPE_MAXCHARS || '20000', 10);
const MAXRES = parseInt(process.env.AXIOM_SCRAPE_RESULTS || '12', 10);

function stripHtml(html) {
  return html
    .replace(/<script[\s\S]*?<\/script>/gi, ' ')
    .replace(/<style[\s\S]*?<\/style>/gi, ' ')
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'").replace(/&nbsp;/g, ' ')
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(parseInt(n, 10)))
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/\s*\n\s*/g, '\n')
    .trim();
}

async function scrape(query) {
  const url = 'https://html.duckduckgo.com/html/?q=' + encodeURIComponent(query);
  const res = await fetch(url, {
    headers: {
      'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36',
      'Accept': 'text/html',
    },
  });
  if (!res.ok) throw new Error('search HTTP ' + res.status);
  const html = await res.text();

  // Pull title + snippet blocks from the results page.
  const pieces = [];
  const re = /class="result__(?:title|snippet)"[^>]*>([\s\S]*?)<\/a>/gi;
  let m;
  while ((m = re.exec(html)) && pieces.length < MAXRES * 2) {
    const t = stripHtml(m[1]);
    if (t.length > 2) pieces.push(t);
  }
  let text = pieces.join('\n');
  // Fallback: if the structured extract found little, strip the whole body.
  if (text.length < 200) text = stripHtml(html);
  if (text.length > MAXCHARS) text = text.slice(0, MAXCHARS);
  return text;
}

(async () => {
  const query = process.argv.slice(2).join(' ').trim();
  if (!query) {
    process.stderr.write('usage: node axiom-scrape.js "<query>"\n');
    process.exit(2);
  }
  try {
    const text = await scrape(query);
    process.stderr.write(`[scrape] query=${JSON.stringify(query)} clean_chars=${text.length}\n`);
    process.stdout.write(text);
  } catch (e) {
    process.stderr.write('[scrape] ERROR: ' + (e && e.message ? e.message : e) + '\n');
    process.exit(1);
  }
})();
