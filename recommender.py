# poi_recommender.py
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

PLACE_MAP_KO2CODE = {
    "관광지": "ms",
    "숙박": "acm",
    "음식점": "food",
}
CODE2PLACE_KO = {v: k for k, v in PLACE_MAP_KO2CODE.items()}

# Only these 3 are filterable per request
VALID_DISABILITIES = {"지체장애", "시각장애", "청각장애"}

DATE_PAT = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")  # e.g., 2025.10.26

def parse_args():
    ap = argparse.ArgumentParser(description="Filter POIs by place type and disability types")
    ap.add_argument("--place", required=True, choices=["관광지", "숙박", "음식점"],
                    help="단일 선택 (관광지|숙박|음식점)")
    ap.add_argument("--disabilities", required=True,
                    help="쉼표로 구분된 장애유형들 (예: 지체장애,시각장애)")
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="details JSON/CSV 파일 경로(여러 개 가능). 예: ms_details.json acm_details.json")
    ap.add_argument("--match", choices=["all", "any"], default="all",
                    help="장애유형 매칭 방식: all(모두 포함, 기본) | any(하나라도 포함)")
    ap.add_argument("--topk", type=int, default=50, help="최대 결과 개수 (기본 50)")
    ap.add_argument("--save-format", choices=["json", "csv", "both", "none"], default="none",
                    help="결과 저장 형식 (기본 저장 안 함)")
    ap.add_argument("--out", type=str, default="poi_results",
                    help="저장 파일 베이스명 (확장자는 자동)")
    return ap.parse_args()

def _infer_code_from_url_or_filename(rec: Dict[str, Any], src_path: Path) -> Optional[str]:
    url = (rec.get("detail_url") or "").lower()
    if "/ms/" in url:
        return "ms"
    if "/acm/" in url:
        return "acm"
    if "/food/" in url:
        return "food"
    name = src_path.name.lower()
    if "ms" in name:
        return "ms"
    if "acm" in name:
        return "acm"
    if "food" in name:
        return "food"
    return None

def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    if DATE_PAT.match(s):
        try:
            return datetime.strptime(s, "%Y.%m.%d")
        except Exception:
            return None
    return None

def _load_one(path: Path) -> List[Dict[str, Any]]:
    data = []
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            if isinstance(obj, list):
                data = obj
            else:
                raise ValueError(f"{path}: JSON must be a list")
    elif path.suffix.lower() == ".csv":
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            data = list(rdr)
            # convert JSON-like fields back from string if present
            for r in data:
                for k in ("basic_info", "barrierfree_info", "images", "tags"):
                    if k in r and isinstance(r[k], str):
                        s = r[k].strip()
                        if s.startswith("{") or s.startswith("["):
                            try:
                                r[k] = json.loads(s)
                            except Exception:
                                pass
                # cast lat/lng if string
                for k in ("lat", "lng"):
                    if k in r and isinstance(r[k], str) and r[k]:
                        try:
                            r[k] = float(r[k])
                        except Exception:
                            pass
    else:
        raise ValueError(f"Unsupported file type: {path}")
    return data

def load_all(paths: List[str]) -> List[Dict[str, Any]]:
    out = []
    for p in paths:
        part = _load_one(Path(p))
        src = Path(p)
        for rec in part:
            rec["_src_path"] = str(src)
            rec["_code"] = _infer_code_from_url_or_filename(rec, src)
            out.append(rec)
    return out

def record_supports_all(rec: Dict[str, Any], target: List[str]) -> bool:
    bf = rec.get("barrierfree_info") or {}
    return all(cat in bf and bf[cat] for cat in target)

def record_supports_any(rec: Dict[str, Any], target: List[str]) -> bool:
    bf = rec.get("barrierfree_info") or {}
    return any(cat in bf and bf[cat] for cat in target)

def intersect_supported(rec: Dict[str, Any], target: List[str]) -> List[str]:
    bf = rec.get("barrierfree_info") or {}
    return [t for t in target if t in bf and bf[t]]

def filter_and_rank(records: List[Dict[str, Any]],
                    place_code: str,
                    disabilities: List[str],
                    match_mode: str = "all") -> List[Dict[str, Any]]:
    # Filter by place code
    recs = [r for r in records if r.get("_code") == place_code]

    # Filter by disabilities
    if disabilities:
        if match_mode == "any":
            recs = [r for r in recs if record_supports_any(r, disabilities)]
        else:
            recs = [r for r in recs if record_supports_all(r, disabilities)]

    # Annotate helpers
    out = []
    for r in recs:
        matched = intersect_supported(r, disabilities)
        upd = r.get("update_date") or r.get("create_date")
        upd_dt = _parse_date(upd)
        out.append({
            **r,
            "_matched_count": len(matched),
            "_matched_list": matched,
            "_upd_dt": upd_dt or datetime.min,
        })

    # Sort: matched_count desc, update_date desc, title asc
    out.sort(key=lambda x: (-x["_matched_count"], -x["_upd_dt"].timestamp(), (x.get("title") or "")))
    return out

def print_table(rows: List[Dict[str, Any]], topk: int):
    from textwrap import shorten
    print("-" * 120)
    print(f"{'순번':<4} {'제목':<30} {'지역':<18} {'매칭':<8} {'URL'}")
    print("-" * 120)
    for i, r in enumerate(rows[:topk], 1):
        title = shorten((r.get("title") or ""), width=30, placeholder="…")
        area = shorten((r.get("area") or ""), width=18, placeholder="…")
        match_str = ",".join(r.get("_matched_list", []))
        url = r.get("detail_url") or ""
        print(f"{i:<4} {title:<30} {area:<18} {match_str:<8} {url}")
    print("-" * 120)
    print(f"총 {min(len(rows), topk)} / {len(rows)}건 표시")

def save_results(rows: List[Dict[str, Any]], base: str, fmt: str, topk: int):
    sel = rows[:topk]
    if fmt in ("json", "both"):
        p = Path(f"{base}.json")
        # Strip helper keys for saving
        cleaned = [{k: v for k, v in r.items() if not k.startswith("_")} for r in sel]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
        print(f"[DONE] Saved JSON: {p}")
    if fmt in ("csv", "both"):
        p = Path(f"{base}.csv")
        fields = ["cotId", "title", "area", "create_date", "update_date",
                  "lat", "lng", "detail_url"]
        # include matched_list for convenience
        fields += ["matched_disabilities"]
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in sel:
                row = {
                    "cotId": r.get("cotId"),
                    "title": r.get("title"),
                    "area": r.get("area"),
                    "create_date": r.get("create_date"),
                    "update_date": r.get("update_date"),
                    "lat": r.get("lat"),
                    "lng": r.get("lng"),
                    "detail_url": r.get("detail_url"),
                    "matched_disabilities": ",".join(r.get("_matched_list", [])),
                }
                w.writerow(row)
        print(f"[DONE] Saved CSV: {p}")

def main():
    args = parse_args()
    place_code = PLACE_MAP_KO2CODE[args.place]
    targets = [t.strip() for t in args.disabilities.split(",") if t.strip()]
    # Validate target disabilities
    invalid = [t for t in targets if t not in VALID_DISABILITIES]
    if invalid:
        raise SystemExit(f"지원하지 않는 장애유형: {invalid}. 선택 가능: {sorted(VALID_DISABILITIES)}")
    if not targets:
        raise SystemExit("장애유형은 최소 1개 이상 필요합니다. 예: --disabilities 지체장애,시각장애")

    # Load records
    records = load_all(args.inputs)
    if not records:
        raise SystemExit("입력 데이터가 비어있습니다.")

    # Filter + rank
    results = filter_and_rank(records, place_code, targets, match_mode=args.match)

    # Print
    print(f"[INFO] 조건: 장소={args.place} / 장애유형={targets} / 매칭={args.match} / 후보={len(results)}")
    print_table(results, args.topk)

    # Save
    if args.save_format != "none":
        base = args.out
        save_results(results, base, args.save_format, args.topk)

if __name__ == "__main__":
    main()
