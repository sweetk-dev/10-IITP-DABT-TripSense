# collect_access_list_until_empty.py
# Crawl main list pages until an empty page is encountered
import re
import csv
import time
import argparse
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE = "https://access.visitkorea.or.kr"
LIST_TYPE = ["ms", "food", "acm"]
USER_AGENT = "Mozilla/5.0 (compatible; CotIdCollector/1.3; +https://example.com)"

UUID_RE = re.compile(r"detail\.do\?cotId=([0-9a-fA-F-]{36})")

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def parse_list_items(html_text, page_no=None):
    """
    목록 카드에서 cotId, name(strong), area(span.area) 추출
    """
    soup = BeautifulSoup(html_text, "html.parser")
    rows = []
    for a in soup.select('a[href*="detail.do?cotId="]'):
        href = a.get("href") or ""
        m = UUID_RE.search(href)
        if not m:
            continue
        cotid = m.group(1)

        name, area = None, None
        cont = a.select_one(".cont")
        if cont:
            strong = cont.find("strong")
            if strong:
                name = strong.get_text(strip=True)
            area_el = cont.select_one(".area")
            if area_el:
                area = area_el.get_text(strip=True)

        detail_url = urljoin(BASE, href)
        rows.append({
            "cotId": cotid,
            "name": name,
            "area": area,
            "detail_url": detail_url,
            "page_no": page_no
        })
    return rows

def iterate_until_empty(session, list_url, start_page, delay, empty_patience=1, max_pages=100000):
    """
    빈 페이지(카드 0개)가 연속 empty_patience번 나오면 조기 종료.
    max_pages는 안전 상한선(무한루프 방지).
    """
    all_rows = []
    empty_streak = 0
    page = start_page
    while page < start_page + max_pages:
        params = {"page": page}
        print(f"[INFO] Fetching list page: {page} params={params} url={list_url}")
        try:
            r = session.get(list_url, params=params, timeout=20)
        except Exception as e:
            print(f"[WARN] Request error on page {page}: {e}. Stop.")
            break

        if r.status_code >= 400:
            print(f"[WARN] HTTP {r.status_code} on page {page}. Treat as empty.")
            empty_streak += 1
        else:
            rows = parse_list_items(r.text, page_no=page)
            if len(rows) == 0:
                empty_streak += 1
                print(f"[INFO] Page {page} is empty (streak={empty_streak}).")
            else:
                empty_streak = 0
                print(f"[INFO] Page {page}: found {len(rows)} rows")
                all_rows.extend(rows)

        if empty_streak >= empty_patience:
            print(f"[STOP] Reached {empty_streak} consecutive empty pages. Early stop at page {page}.")
            break

        page += 1
        time.sleep(delay)

    return all_rows

def save_outputs(rows, out_base="access_ms_list"):
    # CSV: cotId, name, area, detail_url, page_no
    with open(f"{out_base}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cotId", "name", "area", "detail_url", "page_no"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[DONE] Saved {len(rows)} rows.")
    print(f" - CSV: {out_base}.csv")

def main():
    ap = argparse.ArgumentParser(description="Collect cotIds + name + area until an empty page is encountered.")
    ap.add_argument("--type", type=str, default="ms", help="Accessible tourism type (ms, food, acm)")
    ap.add_argument("--start-page", type=int, default=1, help="Start page (default: 1)")
    ap.add_argument("--delay", type=float, default=0.7, help="Delay seconds between requests (default: 0.7)")
    ap.add_argument("--out", type=str, default="access_ms_list", help="Output base filename (default: access_ms_list)")
    ap.add_argument("--empty-patience", type=int, default=1, help="Stop after N consecutive empty pages (default: 1)")
    ap.add_argument("--max-pages", type=int, default=100000, help="Safety cap for max pages to visit")
    args = ap.parse_args()

    # 타입 유효성 검사
    if args.type not in LIST_TYPE:
        raise ValueError(f"Invalid --type '{args.type}'. Choose one of: {LIST_TYPE}")

    list_url = f"{BASE}/{args.type}/list.do"
    print(f"[INFO] Using list URL: {list_url}")

    session = make_session()
    rows = iterate_until_empty(
        session=session,
        list_url=list_url,
        start_page=args.start_page,
        delay=args.delay,
        empty_patience=args.empty_patience,
        max_pages=args.max_pages,
    )

    # cotId 기준 중복 제거(마지막 항목 우선)
    uniq = {}
    for r in rows:
        uniq[r["cotId"]] = r
    rows_dedup = list(uniq.values())

    print(f"[SUMMARY] Total unique items: {len(rows_dedup)}")
    save_outputs(rows_dedup, out_base=args.out)

if __name__ == "__main__":
    main()
