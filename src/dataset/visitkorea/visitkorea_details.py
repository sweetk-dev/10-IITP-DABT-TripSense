# visitkorea_details.py
# -*- coding: utf-8 -*-
import re
import csv
import json
import time
import argparse
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

BASE = "https://access.visitkorea.or.kr"
DETAIL_PATH_TMPL = "/{type}/detail.do"
USER_AGENT = "Mozilla/5.0 (compatible; VisitKoreaDetailCrawler/3.0; +https://example.com)"

TAB_TITLES = {"사진", "기본 정보", "무장애 편의정보", "지도"}

# --------------------------- HTTP Session -----------------------------------
def make_session():
    s = requests.Session()
    retries = Retry(
        total=6,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"])
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def build_detail_url(cotid: str, type_: str) -> str:
    return f"{BASE}{DETAIL_PATH_TMPL.format(type=type_)}?cotId={cotid}"

# --------------------------- Helpers ----------------------------------------
def extract_text(el):
    return el.get_text(" ", strip=True) if el else None

def _urls_from_style(style_value: str):
    if not style_value:
        return []
    urls = []
    for m in re.finditer(r'url\((["\']?)(.*?)\1\)', style_value):
        u = m.group(2).strip()
        if u:
            urls.append(u)
    return urls

_KV_SEPS = [":", "：", "\t"]
_KV_RE_SPACE = re.compile(r"\s{2,}")

def split_kv(text):
    t = (text or "").strip()
    if not t:
        return None
    for sep in _KV_SEPS:
        if sep in t:
            k, v = t.split(sep, 1)
            return k.strip(), v.strip()
    parts = _KV_RE_SPACE.split(t, maxsplit=1)
    if len(parts) == 2:
        k, v = parts[0].strip(), parts[1].strip()
        if 1 <= len(k) <= 60 and v:
            return k, v
    return None

def normalize_homepage_value(val):
    urls = []
    if isinstance(val, dict):
        urls = list(val.get("links") or [])
        if not urls and val.get("text"):
            urls = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", val["text"])
        base_text = val.get("text") or ""
    elif isinstance(val, str):
        urls = re.findall(r"(https?://[^\s]+|www\.[^\s]+)", val)
        base_text = val
    else:
        base_text = ""
    out = []
    for u in urls:
        if u.startswith(("http://", "https://")):
            out.append(u)
        elif u.startswith("www."):
            out.append("https://" + u)
    # Force https scheme when possible
    if out and out[0].startswith("http://"):
        out[0] = "https://" + out[0][7:]
    return out[0] if out else (base_text or None)

# --------------------------- Header parser (Title/Area/Dates) ---------------
def parse_header_fields(soup):
    """
    Returns: {"title": str|None, "area": str|None, "create_date": str|None, "update_date": str|None}
    """
    out = {"title": None, "area": None, "create_date": None, "update_date": None}

    # Title is <h2> inside .db_datail
    h2 = soup.select_one(".db_datail > h2") or soup.select_one("h2")
    if h2 and h2.get_text(strip=True):
        out["title"] = h2.get_text(strip=True)

    # Area & dates from .area_adrs > span
    spans = soup.select(".db_datail .area_adrs span")
    for sp in spans:
        t = sp.get_text(" ", strip=True)
        if not t:
            continue
        if "등록일" in t:
            out["create_date"] = t.split(":", 1)[-1].strip()
        elif "수정일" in t:
            out["update_date"] = t.split(":", 1)[-1].strip()
        else:
            if out["area"] is None:
                out["area"] = t

    return out

# --------------------------- Section collectors -----------------------------
def collect_gallery_images(soup, base_url, page_url):
    urls = []
    galleries = soup.select(".gallery-thumbs, .gallery-top, .gallery, .swiper-container")
    for g in galleries:
        for img in g.find_all("img"):
            srcs = []
            if img.get("src"): srcs.append(img["src"])
            for k in ("data-src", "data-original", "data-lazy", "data-echo"):
                if img.get(k): srcs.append(img[k])
            if img.get("srcset"):
                parts = [p.strip().split()[0] for p in img["srcset"].split(",") if p.strip()]
                srcs.extend(parts)
            for s in srcs:
                if not s or s.startswith("data:"): continue
                if s.startswith("//"): s = "https:" + s
                elif s.startswith("/"): s = urljoin(base_url, s)
                elif not s.startswith("http"): s = urljoin(page_url, s)
                urls.append(s)
        for el in g.select(".swiper-slide, [style*='background']"):
            style = el.get("style") or ""
            for s in _urls_from_style(style):
                if s.startswith("//"): s = "https:" + s
                elif s.startswith("/"): s = urljoin(base_url, s)
                elif not s.startswith("http"): s = urljoin(page_url, s)
                urls.append(s)
    seen, out = set(), []
    for u in urls:
        if u not in seen:
            out.append(u); seen.add(u)
    return out

def collect_tags(soup):
    tags = []
    tag_root = soup.select_one(".tag_area")
    if tag_root:
        for el in tag_root.select("a, span, li"):
            t = (el.get_text(" ", strip=True) or "").strip("#").strip()
            if t:
                tags.append(t)
    seen, out = set(), []
    for t in tags:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def parse_dl_table(root):
    data = {}
    if not root:
        return data
    for dt in root.find_all("dt"):
        key = dt.get_text(strip=True)
        dd = dt.find_next_sibling("dd")
        if not key or not dd:
            continue
        links = [a.get("href") for a in dd.find_all("a", href=True)]
        txt = dd.get_text(" ", strip=True)
        data[key] = {"text": txt, "links": links} if links else txt
    return data

def parse_basic_info_block(soup):
    intro = None
    kv = {}
    root = soup.select_one(".basic_info.target") or soup.select_one(".basic_info")
    if not root:
        return {"소개": None, "kv": {}}  # 소개 키를 "소개"로 고정
    for sel in ["p", ".con", ".desc"]:
        el = root.select_one(sel)
        if el and el.get_text(strip=True):
            intro = el.get_text(" ", strip=True)
            break
    for dl in root.find_all("dl"):
        kv.update(parse_dl_table(dl))
    ICON_MAP = {
        "addr": "주소", "address": "주소", "loc": "주소",
        "tel": "문의", "phone": "문의", "call": "문의",
        "home": "홈페이지", "url": "홈페이지", "link": "홈페이지",
        "time": "이용시간", "clock": "이용시간", "hour": "이용시간",
        "close": "휴무일", "holiday": "휴무일",
        "fee": "이용요금", "price": "이용요금",
    }
    li_nodes = root.select("ul > li")
    auto_idx = 1
    for li in li_nodes:
        em = li.find("em")
        span = li.find("span")
        value_text = extract_text(span) or extract_text(li)
        value_text = value_text.strip() if value_text else None
        if not value_text:
            continue
        label = extract_text(em) if em else None
        if (not label or len(label) == 0) and em:
            label = (em.get("aria-label") or em.get("title") or "").strip() or None
        if (not label or len(label) == 0) and em:
            cls = " ".join(em.get("class", []))
            guessed = None
            for key, ko in ICON_MAP.items():
                if key in cls:
                    guessed = ko
                    break
            label = guessed
        if not label:
            label = f"item_{auto_idx}"
            auto_idx += 1
        kv.setdefault(label, value_text)
    for tr in root.select("table tr"):
        cells = [extract_text(td) for td in tr.find_all(["th", "td"])]
        if len(cells) >= 2 and cells[0] and cells[1]:
            kv.setdefault(cells[0], cells[1])
    # 정규화: 홈페이지 값
    for k in list(kv.keys()):
        if any(tok in k for tok in ("홈페이지", "웹사이트", "사이트")):
            kv[k] = normalize_homepage_value(kv[k])

    # 최종 basic_info는 요구 스키마에 맞춰 "소개" + 나머지 주요 필드만 꺼내 정리
    # 기본 kv를 그대로 남기되, 호출부에서 JSON 직렬화
    out = {"소개": intro}
    out.update(kv)
    return out

# --------------------------- Barrier-free (structured) ----------------------
BF_CLASS_TO_LABEL = {
    "physical": "지체장애",
    "visual":   "시각장애",
    "hearing":  "청각장애",
    "infants":  "영유아가족",
    "elderly":  "고령자",
}

def parse_barrierfree_block_structured(soup):
    """
    Returns:
      {
        "지체장애": {"주차여부": "...", ...},
        "고령자": {"휠체어 대여": "..."},
        ...
      }
    Prefers PC layout; falls back to mobile layout if needed.
    """
    root = soup.select_one(".barrierfree_info")
    out = {}
    if not root:
        return out

    # PC layout: tit_area(h4) -> next sibling .list
    pc = root.select_one(".pc")
    if pc:
        for ta in pc.select(".tit_area"):
            h4 = ta.find("h4")
            if not h4:
                continue
            # category name: from class or text
            cls = ""
            for c in h4.get("class", []):
                if c in BF_CLASS_TO_LABEL:
                    cls = c
                    break
            cat_name = h4.get_text(strip=True)
            if cls in BF_CLASS_TO_LABEL:
                cat_name = BF_CLASS_TO_LABEL[cls]
            if not cat_name:
                continue

            # find paired list
            list_div = ta.find_next_sibling(lambda el: getattr(el, "name", None) and "list" in el.get("class", []))
            if not list_div:
                list_div = ta.find_next(lambda el: getattr(el, "name", None) and "list" in el.get("class", []))
            if not list_div:
                continue

            kv = {}
            for li in list_div.select("li"):
                key = extract_text(li.find("em")) or None
                val = extract_text(li.find("span")) or extract_text(li) or None
                if key and val:
                    kv[key] = " ".join(val.split())
            if kv:
                out[cat_name] = kv

    # Fallback: mobile layout
    if not out:
        mo = root.select_one(".mo")
        if mo:
            categories = []
            for li in mo.select(".bfinfo li"):
                txt = extract_text(li)
                if not txt:
                    sp = li.select_one("[class^=icon]")
                    txt = extract_text(sp)
                if txt:
                    categories.append(txt.strip())

            slides = mo.select(".gallery-top .swiper-slide")
            for idx, slide in enumerate(slides):
                if idx >= len(categories):
                    break
                cat = categories[idx]
                kv = {}
                for li in slide.select("li"):
                    key = extract_text(li.find("em")) or None
                    val = extract_text(li.find("span")) or extract_text(li) or None
                    if key and val:
                        kv[key] = " ".join(val.split())
                if kv:
                    out[cat] = kv

    return out

# --------------------------- Detail parser ----------------------------------
def parse_detail_page(html: str, url: str, cotid: str, title_fallback=None):
    soup = BeautifulSoup(html, "html.parser")

    header = parse_header_fields(soup)
    title = header["title"] or title_fallback

    basic_info = parse_basic_info_block(soup)
    barrierfree_dict = parse_barrierfree_block_structured(soup)
    images = collect_gallery_images(soup, BASE, url)
    tags = collect_tags(soup)

    # lat/lng: data-* or JS vars (locationX/locationY)
    lat = lng = None
    map_el = soup.select_one('[data-lat][data-lng]')
    if map_el:
        lat, lng = map_el.get("data-lat"), map_el.get("data-lng")
    if not lat or not lng:
        # site uses "locationX/Y" in scripts
        m = re.search(r"locationX['\"]?\s*[:=]\s*['\"]?(-?\d+(?:\.\d+)?)", html)
        n = re.search(r"locationY['\"]?\s*[:=]\s*['\"]?(-?\d+(?:\.\d+)?)", html)
        if m and n:
            lat, lng = m.group(1), n.group(1)

    # cast to float if possible
    lat_val = float(lat) if lat is not None else None
    lng_val = float(lng) if lng is not None else None

    return {
        "cotId": cotid,
        "detail_url": url,
        "title": title,
        "area": header["area"],
        "create_date": header["create_date"],
        "update_date": header["update_date"],
        "basic_info": basic_info,
        "barrierfree_info": barrierfree_dict,
        "images": images,
        "tags": tags,
        "lat": lat_val,
        "lng": lng_val,
    }

# --------------------------- Fetch One --------------------------------------
def fetch_detail(session, cotid: str, delay: float, type_: str, title_fallback=None):
    url = build_detail_url(cotid, type_)
    r = session.get(url, timeout=25)
    r.raise_for_status()
    data = parse_detail_page(r.text, url, cotid, title_fallback=title_fallback)
    time.sleep(delay)
    return data

# --------------------------- CSV I/O ----------------------------------------
def read_access_list_csv(path: Path, cotid_col="cotId", name_col="name", area_col="area"):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = (row.get(cotid_col) or "").strip()
            if not c:
                continue
            rows.append({
                "cotId": c,
                "name_list": (row.get(name_col) or "").strip() or None,
                "area_list": (row.get(area_col) or "").strip() or None,
                "detail_url_list": (row.get("detail_url") or "").strip() or None,
                "page_no": (row.get("page_no") or "").strip() or None,
            })
    return rows

def save_json(rows, out_json: Path):
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def save_csv(rows, out_csv: Path):
    fields = [
        "cotId", "detail_url", "title",
        "area", "create_date", "update_date",    # UPDATED: include header fields
        "basic_info",
        "barrierfree_info",
        "images", "tags",
        "lat", "lng",
        "name_list", "area_list", "detail_url_list", "page_no",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = dict(r)
            # JSON-encode complex fields
            for key in ["basic_info", "barrierfree_info", "images", "tags"]:
                if isinstance(row.get(key), (list, dict)):
                    row[key] = json.dumps(row[key], ensure_ascii=False)
            w.writerow(row)

# --------------------------- Main -------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Detail crawler for access.visitkorea.or.kr")
    ap.add_argument("--type", type=str, default="ms", choices=["ms", "food", "acm"], help="Detail page type")
    ap.add_argument("--data", type=str, default="ms_main_list.csv", help="Path to input list CSV (e.g., food_main_list.csv)")
    ap.add_argument("--delay", type=float, default=0.7, help="Delay between requests")
    ap.add_argument("--limit", type=int, default=None, help="Limit items for testing")
    ap.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    ap.add_argument("--out", type=str, default="food_details", help="Output base filename (e.g., food_details)")
    # Optional column names if your list CSV schema differs
    ap.add_argument("--cotid-col", type=str, default="cotId")
    ap.add_argument("--name-col", type=str, default="name")
    ap.add_argument("--area-col", type=str, default="area")
    args = ap.parse_args()

    # 1) Load list CSV
    list_rows = read_access_list_csv(
        Path(args.data),
        cotid_col=args.cotid_col,
        name_col=args.name_col,
        area_col=args.area_col,
    )
    if args.limit is not None:
        list_rows = list_rows[args.start:args.start + args.limit]
    else:
        list_rows = list_rows[args.start:]

    print(f"[INFO] Loaded {len(list_rows)} rows from {args.data}")

    # 2) Crawl details
    session = make_session()
    results = []
    for i, row in enumerate(list_rows, 1):
        cotid = row["cotId"]
        try:
            print(f"[{i}/{len(list_rows)}] Fetching {cotid} ...")
            detail = fetch_detail(
                session, cotid, delay=args.delay, type_=args.type,
                title_fallback=row.get("name_list")
            )
            merged = {
                **detail,
                # "name_list": row.get("name_list"),
                # "area_list": row.get("area_list"),
                # "detail_url_list": row.get("detail_url_list"),
                # "page_no": row.get("page_no"),
            }
            results.append(merged)
        except Exception as e:
            print(f"[WARN] Failed {cotid}: {e}")

    # 3) Save (exactly as given by --out)
    out_json = Path(f"{args.out}.json")
    # out_csv  = Path(f"{args.out}.csv")
    print(f"[INFO] Saving {len(results)} items")
    save_json(results, out_json)
    print(f"[DONE] JSON: {out_json}")
    # save_csv(results, out_csv)
    # print(f"[DONE] CSV: {out_csv}")

if __name__ == "__main__":
    main()
