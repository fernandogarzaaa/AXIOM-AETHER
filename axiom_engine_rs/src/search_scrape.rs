//! Native Rust web scraping for the live search ingestion node.

use std::collections::HashSet;
use std::time::Duration;

use reqwest::blocking::Client;
use scraper::{Html, Selector};

const CONTENT_SELECTORS: [&str; 9] = [
    "main", "article", "section", "h1", "h2", "h3", "p", "li", "pre",
];
const DEFAULT_MAX_PAGES: usize = 5;
const MAX_TEXT_CHARS: usize = 80_000;

pub fn scrape_query_text(query: &str, max_pages: usize) -> Result<String, reqwest::Error> {
    let max_pages = max_pages.max(1);
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .user_agent("AxiomSearchNode/0.1 (+local TTT ingestion)")
        .build()?;

    let urls = if is_http_url(query) {
        vec![query.to_string()]
    } else {
        let search_url = format!(
            "https://duckduckgo.com/html/?q={}",
            urlencoding::encode(query)
        );
        let search_html = fetch_text(&client, &search_url)?;
        extract_search_result_urls(&search_html, max_pages)
    };

    let mut chunks = Vec::new();
    for url in urls.into_iter().take(max_pages) {
        match fetch_text(&client, &url) {
            Ok(html) => {
                let text = clean_html_text(&html);
                if !text.trim().is_empty() {
                    chunks.push(format!("# Source: {url}\n{text}"));
                }
            }
            Err(e) => eprintln!("[search-scrape] skipped {url}: {e}"),
        }
        if chunks.join("\n\n").len() >= MAX_TEXT_CHARS {
            break;
        }
    }

    Ok(truncate_chars(&chunks.join("\n\n"), MAX_TEXT_CHARS))
}

pub fn default_max_pages() -> usize {
    DEFAULT_MAX_PAGES
}

pub fn clean_html_text(html: &str) -> String {
    let document = Html::parse_document(html);
    let mut chunks: Vec<String> = Vec::new();

    for selector in CONTENT_SELECTORS {
        let selector = Selector::parse(selector).expect("static selector parses");
        for element in document.select(&selector) {
            let text = normalize_ws(&element.text().collect::<Vec<_>>().join(" "));
            if text.len() >= 3 && !chunks.iter().any(|existing| existing == &text) {
                chunks.push(text);
            }
        }
    }

    if chunks.is_empty() {
        return normalize_ws(&document.root_element().text().collect::<Vec<_>>().join(" "));
    }

    chunks.join("\n")
}

pub fn extract_search_result_urls(html: &str, limit: usize) -> Vec<String> {
    let document = Html::parse_document(html);
    let selector = Selector::parse("a.result__a, a[href]").expect("static selector parses");
    let mut seen = HashSet::new();
    let mut urls = Vec::new();

    for element in document.select(&selector) {
        let Some(href) = element.value().attr("href") else {
            continue;
        };
        let Some(url) = normalize_result_url(href) else {
            continue;
        };
        if seen.insert(url.clone()) {
            urls.push(url);
            if urls.len() >= limit {
                break;
            }
        }
    }

    urls
}

fn normalize_ws(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_result_url(href: &str) -> Option<String> {
    let href = href.trim();
    if href.starts_with("http://") || href.starts_with("https://") {
        return Some(href.to_string());
    }

    let marker = "uddg=";
    if let Some(start) = href.find(marker) {
        let encoded = &href[start + marker.len()..];
        let encoded = encoded.split('&').next().unwrap_or(encoded);
        if let Ok(decoded) = urlencoding::decode(encoded) {
            let decoded = decoded.into_owned();
            if decoded.starts_with("http://") || decoded.starts_with("https://") {
                return Some(decoded);
            }
        }
    }

    None
}

fn fetch_text(client: &Client, url: &str) -> Result<String, reqwest::Error> {
    client.get(url).send()?.error_for_status()?.text()
}

fn is_http_url(value: &str) -> bool {
    let value = value.trim();
    value.starts_with("http://") || value.starts_with("https://")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut end = max_chars;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_html_text_keeps_article_content_and_drops_chrome() {
        let html = r#"
            <html>
              <head>
                <style>.hidden { color: red; }</style>
                <script>window.secret = "drop me";</script>
                <title>Ignored browser title</title>
              </head>
              <body>
                <nav>Navigation noise</nav>
                <main>
                  <h1>Rust ingestion node</h1>
                  <p>Axiom absorbs live search results into fast weights.</p>
                  <pre><code>fn scrape() -> Result&lt;()&gt;</code></pre>
                </main>
                <footer>Footer noise</footer>
              </body>
            </html>
        "#;

        let text = clean_html_text(html);

        assert!(text.contains("Rust ingestion node"));
        assert!(text.contains("Axiom absorbs live search results into fast weights."));
        assert!(text.contains("fn scrape() -> Result"));
        assert!(!text.contains("Navigation noise"));
        assert!(!text.contains("Footer noise"));
        assert!(!text.contains("drop me"));
        assert!(!text.contains(".hidden"));
        assert!(!text.contains("  "));
    }

    #[test]
    fn extract_search_result_urls_decodes_duckduckgo_redirects() {
        let html = r#"
            <html><body>
              <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fa%3D1&amp;rut=abc">Example</a>
              <a class="result__a" href="https://second.example/path">Second</a>
              <a class="result__a" href="/local/path">Ignore local</a>
            </body></html>
        "#;

        let urls = extract_search_result_urls(html, 8);

        assert_eq!(
            urls,
            vec![
                "https://example.com/docs?a=1".to_string(),
                "https://second.example/path".to_string()
            ]
        );
    }
}
