# recommend_poi.py
# -*- coding: utf-8 -*-
"""
Generate POI recommendation results (per-query + combined)
and save them under ./results/runs/
"""

import os
import json
import argparse
import pandas as pd

DEFAULT_DATA_PATH = "D:/workspace/SW공인시험/Tourism/data/visitkorea"
DETAIL_FILES = {"ms": "ms_details.json", "acm": "acm_details.json", "food": "food_details.json"}
RESULT_ROOT = "results"
RUN_DIR = os.path.join(RESULT_ROOT, "runs")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")


def load_records(data_path):
    records = []
    for code, fname in DETAIL_FILES.items():
        path = os.path.join(data_path, fname)
        if not os.path.exists(path):
            print(f"[WARN] Missing file: {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for r in data:
                r["_code"] = code  # ms / acm / food 구분용
                records.append(r)
    print(f"[INFO] Loaded {len(records)} POIs")
    return records


def load_queries(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_match(bf, wants, mode):
    """
    bf: barrierfree_info (dict or None)
    wants: list of desired disability types (e.g. ["지체장애"])
    mode: "all" or "any"

    return:
        ok: bool (추천에 포함할지 여부)
        num_matched: int (교집합 개수)
    """
    keys = set((bf or {}).keys())
    wants = set(wants or [])
    matched = wants & keys

    if mode == "all":
        ok = wants.issubset(keys)
    else:
        ok = len(matched) > 0

    return ok, len(matched)


def recommend_for_query(records, q, topk, mode):
    # place 코드에 해당하는 POI만 사용
    subset = [r for r in records if r.get("_code") == q["place"]]
    results = []

    wants = q.get("disabilities", [])
    total_wants = len(wants)

    for r in subset:
        bf = r.get("barrierfree_info")
        ok, matched_count = is_match(bf, wants, mode)

        # relevance(0 or 1)
        is_relevant = 1 if matched_count > 0 else 0

        if ok:
            disability_type = list((bf or {}).keys())

            results.append({
                "place": q.get("place", ""),
                "cotId": r.get("cotId"),
                "title": r.get("title"),
                "area": r.get("area"),
                "disability_type": disability_type,
                "is_relevant": is_relevant,       # num_matched 대체
                "matched_count": matched_count     # 정렬용
            })

    if not results:
        return pd.DataFrame(columns=[
            "rank", "place", "cotId", "title", "area",
            "disability_type", "is_relevant", "precision"
        ])

    df = pd.DataFrame(results)

    # 정렬 기준: 실제 매칭 개수 > title
    df = df.sort_values(
        ["matched_count", "title"],
        ascending=[False, True],
        kind="mergesort"
    )

    # rank 부여
    df["rank"] = range(1, len(df) + 1)

    # 누적 relevant 수 계산 → P(k) 계산 기반
    df["cum_relevant"] = df["is_relevant"].cumsum()

    # precision@k 계산: cum_relevant(k) / k → "n/k" 문자열로 저장
    df["precision"] = df["cum_relevant"].astype(str) + "/" + df["rank"].astype(str)

    # top-k만 유지
    df = df.head(topk)

    # 최종 컬럼 반환
    return df[[
        "rank", "place", "cotId", "title", "area",
        "disability_type", "is_relevant", "precision"
    ]]


def main():
    ap = argparse.ArgumentParser(description="Generate POI recommendation results and save under results/runs/")
    ap.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH)
    ap.add_argument("--queries-json", type=str, required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--match-mode", type=str, default="all", choices=["all", "any"])
    args = ap.parse_args()

    ensure_dir(RUN_DIR)
    records = load_records(args.data_path)
    queries = load_queries(args.queries_json)

    # combined = []
    for q in queries:
        qid = q["query_id"]
        df = recommend_for_query(records, q, args.topk, args.match_mode)
        out_q = os.path.join(RUN_DIR, f"run_{qid}.csv")
        df.to_csv(out_q, index=False, encoding="utf-8-sig")
        print(f"[OK] Saved per-query run → {out_q} ({len(df)} rows)")
        # combined.append(df)

    # if combined:
    #     df_all = pd.concat(combined, ignore_index=True)
    #     combined_path = os.path.join(RUN_DIR, "run_results.csv")
    #     df_all.to_csv(combined_path, index=False, encoding="utf-8-sig")
    #     print(f"[OK] Combined run saved → {combined_path}")


if __name__ == "__main__":
    main()
