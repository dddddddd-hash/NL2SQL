import argparse
import csv
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import requests
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HomeRAG 可量化评估脚本")
    parser.add_argument(
        "--dataset",
        default="eval/dataset.sample.jsonl",
        help="评估数据集 JSONL 路径",
    )
    parser.add_argument(
        "--api-url",
        default="http://127.0.0.1:8000/api/v1/query",
        help="同步查询接口地址",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="单条请求超时（秒）",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"数据集不存在: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            row.setdefault("id", f"case_{line_no}")
            row.setdefault("category", "general")
            row.setdefault("expected_sql_keywords", [])
            row.setdefault("expected_answer_keywords", [])
            rows.append(row)
    if not rows:
        raise ValueError("数据集为空，请至少提供1条样本")
    return rows


def keyword_hit_rate(text: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    text_lower = (text or "").lower()
    hit = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hit / len(keywords)


def evaluate_case(
    case: Dict[str, Any], api_url: str, timeout: int
) -> Dict[str, Any]:
    payload = {"query": case["query"]}
    result: Dict[str, Any] = {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
    }

    try:
        resp = requests.post(api_url, json=payload, timeout=timeout)
        result["status_code"] = resp.status_code
        if resp.status_code != 200:
            result["success"] = False
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
            result["elapsed_ms"] = None
            result["sql_keyword_hit_rate"] = 0.0
            result["answer_keyword_hit_rate"] = 0.0
            return result

        data = resp.json()
        sql_text = (data.get("sql") or "").strip()
        answer_text = (data.get("answer") or "").strip()
        elapsed_ms = int(data.get("elapsed_ms") or 0)

        sql_hit = keyword_hit_rate(sql_text, case.get("expected_sql_keywords", []))
        answer_hit = keyword_hit_rate(
            answer_text, case.get("expected_answer_keywords", [])
        )

        success = (
            bool(sql_text)
            and "失败" not in answer_text
            and "抱歉" not in answer_text
            and "遇到问题" not in answer_text
        )

        result.update(
            {
                "success": success,
                "error": "",
                "elapsed_ms": elapsed_ms,
                "sql": sql_text,
                "answer": answer_text,
                "sql_keyword_hit_rate": round(sql_hit, 4),
                "answer_keyword_hit_rate": round(answer_hit, 4),
            }
        )
        return result

    except Exception as e:
        result["status_code"] = None
        result["success"] = False
        result["error"] = str(e)
        result["elapsed_ms"] = None
        result["sql_keyword_hit_rate"] = 0.0
        result["answer_keyword_hit_rate"] = 0.0
        return result


def percentile(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * p)
    return float(sorted_values[idx])


def build_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    success_count = sum(1 for r in results if r["success"])
    latencies = [r["elapsed_ms"] for r in results if isinstance(r["elapsed_ms"], int)]
    sql_hits = [r["sql_keyword_hit_rate"] for r in results]
    answer_hits = [r["answer_keyword_hit_rate"] for r in results]

    category_stats: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r["category"]
        category_stats.setdefault(cat, {"total": 0, "success": 0})
        category_stats[cat]["total"] += 1
        category_stats[cat]["success"] += int(r["success"])

    for cat in category_stats:
        item = category_stats[cat]
        item["success_rate"] = round(
            item["success"] / item["total"] if item["total"] else 0.0, 4
        )

    summary = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "total_cases": total,
        "success_rate": round(success_count / total if total else 0.0, 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "p50_latency_ms": round(percentile(latencies, 0.5), 2),
        "p90_latency_ms": round(percentile(latencies, 0.9), 2),
        "sql_keyword_hit_rate": round(statistics.mean(sql_hits), 4) if sql_hits else 0.0,
        "answer_keyword_hit_rate": round(statistics.mean(answer_hits), 4)
        if answer_hits
        else 0.0,
        "category_stats": category_stats,
    }
    return summary


def plot_charts(results: List[Dict[str, Any]], summary: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_names = list(summary["category_stats"].keys())
    cat_rates = [summary["category_stats"][c]["success_rate"] * 100 for c in cat_names]
    plt.figure(figsize=(8, 4))
    plt.bar(cat_names, cat_rates)
    plt.ylim(0, 100)
    plt.ylabel("成功率 (%)")
    plt.title("各类别任务成功率")
    plt.tight_layout()
    plt.savefig(out_dir / "category_success_rate.png", dpi=160)
    plt.close()

    latencies = [r["elapsed_ms"] for r in results if isinstance(r["elapsed_ms"], int)]
    plt.figure(figsize=(8, 4))
    if latencies:
        plt.hist(latencies, bins=min(10, max(3, len(latencies))))
    plt.xlabel("耗时 (ms)")
    plt.ylabel("样本数")
    plt.title("请求延迟分布")
    plt.tight_layout()
    plt.savefig(out_dir / "latency_hist.png", dpi=160)
    plt.close()

    metrics = [
        summary["success_rate"] * 100,
        summary["sql_keyword_hit_rate"] * 100,
        summary["answer_keyword_hit_rate"] * 100,
    ]
    labels = ["成功率", "SQL关键词命中", "回答关键词命中"]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, metrics)
    plt.ylim(0, 100)
    plt.ylabel("比例 (%)")
    plt.title("核心质量指标")
    plt.tight_layout()
    plt.savefig(out_dir / "quality_metrics.png", dpi=160)
    plt.close()


def write_csv(results: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "category",
        "query",
        "success",
        "status_code",
        "elapsed_ms",
        "sql_keyword_hit_rate",
        "answer_keyword_hit_rate",
        "error",
        "sql",
        "answer",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k) for k in fields})


def write_reports(summary: Dict[str, Any], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_json = report_dir / "summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = f"""# HomeRAG 评估报告

- 生成时间: {summary['generated_at']}
- 样本总数: {summary['total_cases']}
- 成功率: {summary['success_rate'] * 100:.2f}%
- 平均延迟: {summary['avg_latency_ms']} ms
- P50 延迟: {summary['p50_latency_ms']} ms
- P90 延迟: {summary['p90_latency_ms']} ms
- SQL关键词命中率: {summary['sql_keyword_hit_rate'] * 100:.2f}%
- 回答关键词命中率: {summary['answer_keyword_hit_rate'] * 100:.2f}%

## 可视化图表

![分类成功率](category_success_rate.png)
![延迟分布](latency_hist.png)
![核心质量指标](quality_metrics.png)
"""
    (report_dir / "report.md").write_text(md, encoding="utf-8")

    category_rows = ""
    for category, item in summary.get("category_stats", {}).items():
        category_rows += (
            "<tr>"
            f"<td>{category}</td>"
            f"<td>{item.get('total', 0)}</td>"
            f"<td>{item.get('success', 0)}</td>"
            f"<td>{item.get('success_rate', 0.0) * 100:.2f}%</td>"
            "</tr>"
        )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HomeRAG 评估报告</title>
  <style>
    :root {{
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --primary: #4f46e5;
      --border: #e5e7eb;
      --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
      --radius: 14px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #eef2ff 0%, #f8fafc 35%, #f5f7fb 100%);
    }}
    .wrap {{
      max-width: 1100px;
      margin: 24px auto 40px;
      padding: 0 16px;
    }}
    .hero {{
      background: radial-gradient(circle at 20% 10%, #6366f1, #4338ca 70%);
      color: #fff;
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--shadow);
    }}
    .hero h1 {{ margin: 0 0 8px 0; font-size: 28px; }}
    .small {{ margin: 0; color: rgba(255, 255, 255, 0.9); font-size: 14px; }}
    .kpi {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 16px;
    }}
    .metric-title {{ color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .metric-value {{ font-size: 24px; font-weight: 700; color: var(--primary); }}
    .section {{ margin-top: 18px; }}
    .section h2 {{ margin: 0 0 12px 0; font-size: 18px; }}
    .chart {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      padding: 12px;
      margin-bottom: 14px;
    }}
    img {{
      width: 100%;
      margin: 0;
      border-radius: 10px;
      display: block;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: #fff;
    }}
    th, td {{
      text-align: left;
      padding: 12px;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }}
    th {{ background: #f8fafc; color: #374151; }}
    tr:last-child td {{ border-bottom: none; }}
    @media (max-width: 900px) {{
      .kpi {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 600px) {{
      .kpi {{ grid-template-columns: 1fr; }}
      .hero h1 {{ font-size: 24px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>HomeRAG 评估报告</h1>
      <p class="small">生成时间：{summary['generated_at']}</p>
    </div>

    <div class="kpi">
      <div class="card"><div class="metric-title">样本总数</div><div class="metric-value">{summary['total_cases']}</div></div>
      <div class="card"><div class="metric-title">成功率</div><div class="metric-value">{summary['success_rate'] * 100:.2f}%</div></div>
      <div class="card"><div class="metric-title">平均延迟</div><div class="metric-value">{summary['avg_latency_ms']} ms</div></div>
      <div class="card"><div class="metric-title">P50 延迟</div><div class="metric-value">{summary['p50_latency_ms']} ms</div></div>
      <div class="card"><div class="metric-title">P90 延迟</div><div class="metric-value">{summary['p90_latency_ms']} ms</div></div>
      <div class="card"><div class="metric-title">SQL 关键词命中率</div><div class="metric-value">{summary['sql_keyword_hit_rate'] * 100:.2f}%</div></div>
      <div class="card"><div class="metric-title">回答关键词命中率</div><div class="metric-value">{summary['answer_keyword_hit_rate'] * 100:.2f}%</div></div>
    </div>

    <div class="section">
      <h2>分类统计</h2>
      <div class="card">
        <table>
          <thead>
            <tr>
              <th>类别</th>
              <th>样本数</th>
              <th>成功数</th>
              <th>成功率</th>
            </tr>
          </thead>
          <tbody>
            {category_rows}
          </tbody>
        </table>
      </div>
    </div>

    <div class="section">
      <h2>各类别任务成功率</h2>
      <div class="chart">
        <img src="./category_success_rate.png" alt="category success rate" />
      </div>
    </div>
    <div class="section">
      <h2>延迟分布</h2>
      <div class="chart">
        <img src="./latency_hist.png" alt="latency hist" />
      </div>
    </div>
    <div class="section">
      <h2>核心质量指标</h2>
      <div class="chart">
        <img src="./quality_metrics.png" alt="quality metrics" />
      </div>
    </div>
  </div>
</body>
</html>
"""
    (report_dir / "report.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path("reports") / "eval" / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    for case in dataset:
        results.append(evaluate_case(case, args.api_url, args.timeout))

    summary = build_summary(results)

    (output_root / "raw_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(results, output_root / "results.csv")
    plot_charts(results, summary, output_root)
    write_reports(summary, output_root)

    latest_dir = Path("reports") / "eval" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "latest_run.txt").write_text(str(output_root), encoding="utf-8")

    print("评估完成")
    print(f"报告目录: {output_root}")
    print(f"HTML报告: {output_root / 'report.html'}")


if __name__ == "__main__":
    main()
