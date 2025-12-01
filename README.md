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
