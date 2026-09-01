# POI Recommendation

This repository provides a command-line tool for recommending POIs (Points of Interest) based on a set of input queries. The tool loads query information from a JSON file and returns the top-k recommended POIs according to the selected matching mode.


## Installation

```bash
pip install requests
pip install beautifulsoup4
pip install pandas
```

## Usage

### Example `queries.json`
```json
[
  { "query_id": "Q1",  "place": "ms",   "disabilities": ["지체장애"] },
  { "query_id": "Q2",  "place": "acm",  "disabilities": ["지체장애"] },
  { "query_id": "Q3", "place": "food", "disabilities": ["지체장애"] },
  { "query_id": "Q4",  "place": "ms",   "disabilities": ["지체장애", "시각장애"] },
  { "query_id": "Q5",  "place": "acm",  "disabilities": ["지체장애", "시각장애"] }
]
```


### Basic Command
```bash
python recommend_poi.py --queries-json queries.json --topk 10 --match-mode all
```

### Arguments

| Argument         | Description                                 | Required | Example       |
|------------------|---------------------------------------------|----------|---------------|
| --queries-json   | Path to the JSON file containing query items | Yes      | queries.json  |
| --topk           | Number of POIs to return                     | No       | 10            |
| --match-mode     | Matching strategy used for filtering POIs     | No       | all, any      |

---

### DEMO (Streamlit)
```bash
streamlit run app.py
```

## 라이선스

이 프로젝트는 MIT 라이선스로 배포됩니다. 전문은 [LICENSE](LICENSE) 파일을 참고하십시오.

본 연구는 정부(과학기술정보통신부)의 재원으로 정보통신기획평가원의 지원을 받아 수행된 연구입니다.
(연구개발과제번호 RS-2024-003976, 데이터 기반 장애인 데이터 탐색·활용 해결기술 개발)
