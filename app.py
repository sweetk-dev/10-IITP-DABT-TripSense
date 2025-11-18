# streamlit run app.py
# -*- coding: utf-8 -*-
import os
import json
import re
from io import StringIO
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st

# ======================= 설정 =======================
st.set_page_config(page_title="무장애 정보 추천", page_icon="🧭", layout="wide")

DATA_PATH = "./data/visitkorea"
PLACE_MAP_KO2CODE = {"관광지": "ms", "숙박": "acm", "음식점": "food"}
DATE_PAT = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
EPOCH = datetime(1970, 1, 1)

# ======================= Helper 함수 =======================
def _infer_code_from_url_or_name(rec: Dict[str, Any], name_hint: str = "") -> Optional[str]:
    url = str(rec.get("detail_url") or "").lower()
    for code in ["ms", "acm", "food"]:
        if f"/{code}/" in url or code in name_hint.lower():
            return code
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

def _safe_ts(dt: Optional[datetime]) -> float:
    if not dt:
        return 0.0
    if dt < EPOCH:
        dt = EPOCH
    try:
        return dt.timestamp()
    except Exception:
        return 0.0

def load_from_path(data_dir: str) -> List[Dict[str, Any]]:
    """폴더 내 CSV/JSON 파일 자동 로드"""
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)
                 if f.lower().endswith((".csv", ".json"))]
    if not all_files:
        st.error(f"'{data_dir}' 경로에 CSV/JSON 파일이 없습니다.")
        st.stop()

    records = []
    for path in all_files:
        name = os.path.basename(path)
        try:
            if path.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
                data = df.to_dict(orient="records")
            for r in data:
                r["_code"] = _infer_code_from_url_or_name(r, name_hint=name)
            records.extend(data)
        except Exception:
            print(f"[WARN] {name} 파일 로드 실패")
    return records

# ======================= 지역 계층 추출 =======================
def extract_region_hierarchy(records: List[Dict[str, Any]]):
    """area 필드에서 시도 / 시군구 목록 추출"""
    areas = [r.get("area", "") for r in records if r.get("area")]
    sido_list, sigungu_map = set(), {}
    for a in areas:
        parts = re.split(r"\s+", a.strip())
        if len(parts) >= 1:
            sido = parts[0]
            sido_list.add(sido)
            sigungu = parts[1] if len(parts) >= 2 else None
            if sido not in sigungu_map:
                sigungu_map[sido] = set()
            if sigungu:
                sigungu_map[sido].add(sigungu)
    return sorted(list(sido_list)), {k: sorted(list(v)) for k, v in sigungu_map.items()}

# ======================= 필터링 & 정렬 =======================
def record_supports(rec, targets, mode="all"):
    bf = rec.get("barrierfree_info") or {}
    if mode == "any":
        return any(t in bf and bf[t] for t in targets)
    return all(t in bf and bf[t] for t in targets)

def matched_list(rec, targets):
    bf = rec.get("barrierfree_info") or {}
    return [t for t in targets if t in bf and bf[t]]

def filter_and_rank(records, place_code, disabilities, mode="all", sido=None, sigungu=None):
    recs = [r for r in records if r.get("_code") == place_code]
    if sido:
        recs = [r for r in recs if (r.get("area") or "").startswith(sido)]
    if sigungu:
        recs = [r for r in recs if sigungu in (r.get("area") or "")]
    recs = [r for r in recs if record_supports(r, disabilities, mode)]

    ranked = []
    for r in recs:
        mlist = matched_list(r, disabilities)
        upd = r.get("update_date") or r.get("create_date")
        upd_dt = _parse_date(upd) or datetime.min
        ranked.append({**r, "_matched_list": mlist, "_count": len(mlist), "_upd_dt": upd_dt})
    ranked.sort(key=lambda x: (-x["_count"], -_safe_ts(x["_upd_dt"]), (x.get("title") or "")))
    return ranked

# ======================= 변환 / 다운로드 =======================
def to_dataframe(rows):
    return pd.DataFrame([{
        "cotId": r.get("cotId"),
        "title": r.get("title"),
        "area": r.get("area"),
        "update_date": r.get("update_date"),
        "matched_disabilities": ",".join(r.get("_matched_list", [])),
        "detail_url": r.get("detail_url"),
        "lat": r.get("lat"),
        "lng": r.get("lng"),
    } for r in rows])

def df_to_csv_bytes(df):
    buf = StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")

def rows_to_json_bytes(rows):
    cleaned = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    return json.dumps(cleaned, ensure_ascii=False, indent=2).encode("utf-8")

# ======================= 네비게이션 상태 =======================
def go_detail(cotid):
    st.session_state["view"] = "detail"
    st.session_state["selected_cotid"] = cotid

def go_main():
    st.session_state["view"] = "main"
    st.session_state["selected_cotid"] = None

if "view" not in st.session_state:
    st.session_state["view"] = "main"
if "selected_cotid" not in st.session_state:
    st.session_state["selected_cotid"] = None

# ======================= 데이터 로드 =======================
ALL_RECORDS = load_from_path(DATA_PATH)
SIDO_LIST, SIGUNGU_MAP = extract_region_hierarchy(ALL_RECORDS)

# ======================= Main View =======================
def render_main():
    st.title("🧭 무장애 정보 추천")
    with st.sidebar:
        st.header("상세 검색")
        place = st.selectbox("관광정보", ["관광지", "숙박", "음식점"], index=0)
        sido = st.selectbox("시도 선택", ["전체"] + SIDO_LIST, index=0)
        sigungu = st.selectbox(
            "시군구 선택",
            ["전체"] + (SIGUNGU_MAP.get(sido, []) if sido != "전체" else []),
            index=0
        )
        dis_sel = st.multiselect("장애유형", ["지체장애", "시각장애", "청각장애"], default=["지체장애"])
        mode = st.radio("매칭 방식", ["all", "any"], index=0)
        topk = st.number_input("표시 개수", min_value=1, max_value=10000, value=50, step=1)
        map_on = st.checkbox("지도 보기", value=False)

    # 지역 필터 적용
    place_code = PLACE_MAP_KO2CODE[place]
    sido_filter = None if sido == "전체" else sido
    sigungu_filter = None if sigungu == "전체" else sigungu
    results = filter_and_rank(ALL_RECORDS, place_code, dis_sel, mode=mode,
                              sido=sido_filter, sigungu=sigungu_filter)

    st.write(f"**조건**: 장소=`{place}`, 지역=`{sido}` {sigungu if sigungu!='전체' else ''}, 장애유형=`{', '.join(dis_sel)}` / 결과={len(results)}")

    top_rows = results[:topk]
    df = to_dataframe(top_rows)

    st.subheader("검색 결과")
    for _, row in df.iterrows():
        with st.container(border=True):
            cols = st.columns([4, 3, 2, 2, 2])
            cols[0].markdown(f"**{row['title']}**")
            cols[1].markdown(row["area"] or "")
            cols[2].markdown(row["update_date"] or "")
            cols[3].markdown(row["matched_disabilities"] or "")
            if cols[4].button("🔎 상세 보기", key=f"detail_{row['cotId']}"):
                go_detail(row["cotId"])
                st.rerun()

    st.markdown("### 다운로드")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("⬇️ CSV 다운로드", df_to_csv_bytes(df), "poi_results.csv", "text/csv")
    with col2:
        st.download_button("⬇️ JSON 다운로드", rows_to_json_bytes(top_rows), "poi_results.json", "application/json")

    if map_on and {"lat", "lng"}.issubset(df.columns):
        map_df = df.dropna(subset=["lat", "lng"]).rename(columns={"lat": "latitude", "lng": "longitude"})
        if not map_df.empty:
            st.map(map_df[["latitude", "longitude"]])
        else:
            st.info("표시 가능한 위경도 데이터가 없습니다.")

# ======================= Detail View =======================
def render_detail():
    cotid = st.session_state.get("selected_cotid")
    rec = next((r for r in ALL_RECORDS if str(r.get("cotId")) == str(cotid)), None)
    if not rec:
        st.warning("상세 데이터를 찾을 수 없습니다.")
        if st.button("⬅️ 메인으로"): go_main(); st.rerun()
        return

    st.title(f"📍 {rec.get('title', '상세정보')}")
    st.markdown(f"**지역:** {rec.get('area', '-')}  ")
    st.markdown(f"**등록일:** {rec.get('create_date', '-')} / **수정일:** {rec.get('update_date', '-')}")

    if rec.get("detail_url"):
        st.markdown(f"[상세 페이지 바로가기]({rec['detail_url']})")

    if st.button("⬅️ 메인으로"): go_main(); st.rerun()

    st.markdown("---")
    st.subheader("기본 정보")
    basic = rec.get("basic_info") or {}
    if basic:
        for k, v in basic.items():
            st.write(f"- **{k}**: {v}")
    else:
        st.info("기본 정보 없음")

    st.subheader("무장애 편의정보")
    bf = rec.get("barrierfree_info") or {}
    if bf:
        for cat, kv in bf.items():
            st.write(f"### {cat}")
            if isinstance(kv, dict):
                for k, v in kv.items():
                    st.write(f"- {k}: {v}")
            else:
                st.write(kv)
    else:
        st.info("무장애 편의정보 없음")

    imgs = rec.get("images") or []
    if imgs:
        st.subheader("이미지")
        for i in range(0, len(imgs), 3):
            cols = st.columns(3)
            for j, url in enumerate(imgs[i:i+3]):
                with cols[j]:
                    st.image(url, use_container_width=True)

    if rec.get("lat") and rec.get("lng"):
        st.subheader("지도")
        map_df = pd.DataFrame([{"latitude": rec["lat"], "longitude": rec["lng"]}])
        st.map(map_df)

# ======================= 실행 =======================
if st.session_state["view"] == "detail":
    render_detail()
else:
    render_main()
