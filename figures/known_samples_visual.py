import polars as pl
import os
from functions import (
    classify_cancer_samples,
    get_nlp,
)

def build_html_report(tp, fp, tn, fn, accuracy, precision, recall, cancer_acc, non_cancer_acc):
    total = tp + fp + tn + fn
    
    # Calculate intensities for matrix styling
    max_val = max(tp, tn, fp, fn)
    def intensity(val):
        return max(0.1, val / max_val if max_val > 0 else 0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Performance Metrics</title>
    <link href="https://fonts.googleapis.com/css2?family=Liberation+Sans:wght@400;700&family=Arial:wght@400;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Arial', 'Liberation Sans', sans-serif;
            background-color: #ffffff;
            color: #000;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }}
        .container {{
            max-width: 1000px;
            width: 100%;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #555;
            font-size: 16px;
        }}
        
        /* Metric Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 40px;
        }}
        .card {{
            padding: 24px;
            border-radius: 15px;
            text-align: center;
            border: none;
        }}
        .card h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #444; margin-bottom: 15px; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #000; }}
        
        /* Transparent Backgrounds */
        .c-acc {{ background: transparent; }}
        .c-prec {{ background: transparent; }}
        .c-rec {{ background: transparent; }}
        .c-tot {{ background: transparent; }}

        /* Confusion Matrix */
        .matrix-container {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            text-align: center;
            background: #fff;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 2px solid #ccc;
        }}
        th, td {{
            padding: 20px 30px;
            border: 1px solid #ddd;
        }}
        th {{
            background: #f8fafc;
            color: #334155;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .true-class {{ background: #f1f5f9; font-weight: bold; writing-mode: vertical-rl; transform: rotate(180deg); }}
        
        /* Matrix Cells styling based on pastel blue/orange vibe */
        .cell-tp {{ background: rgba(49, 130, 189, {intensity(tp)}); color: { '#fff' if intensity(tp)>0.5 else '#000' }; }}
        .cell-tn {{ background: rgba(49, 130, 189, {intensity(tn)}); color: { '#fff' if intensity(tn)>0.5 else '#000' }; }}
        .cell-fp {{ background: rgba(215, 138, 60, {intensity(fp)}); color: { '#fff' if intensity(fp)>0.5 else '#000' }; }}
        .cell-fn {{ background: rgba(215, 138, 60, {intensity(fn)}); color: { '#fff' if intensity(fn)>0.5 else '#000' }; }}

        .cell-value {{ font-size: 28px; font-weight: bold; margin-bottom: 5px; }}
        .cell-label {{ font-size: 12px; font-weight: bold; opacity: 0.9; }}

        /* Export Button Styling */
        .export-controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            font-family: Arial, sans-serif;
            z-index: 1000;
        }}
        .export-btn {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            color: #334155;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            font-weight: bold;
            transition: all 0.2s;
        }}
        .export-btn:hover {{ background: #e2e8f0; }}

    </style>
</head>
<body>

    <div class="export-controls" id="exportControls">
        <button class="export-btn" onclick="exportSVG()">📥 SVG</button>
        <button class="export-btn" onclick="exportPNG()">📥 PNG (High-Res)</button>
    </div>

    <div class="container">
        <div class="header">
            <h1>Cancer Classification Performance</h1>
            <p>Evaluating automated pipeline accuracy against manually curated ground truth</p>
        </div>

        <div class="metrics-grid">
            <div class="card c-acc">
                <h3>Overall Accuracy</h3>
                <div class="value">{accuracy:.1f}%</div>
            </div>
            <div class="card c-prec">
                <h3>Precision</h3>
                <div class="value">{precision:.1f}%</div>
            </div>
            <div class="card c-rec">
                <h3>Recall</h3>
                <div class="value">{recall:.1f}%</div>
            </div>
            <div class="card c-tot">
                <h3>N Samples</h3>
                <div class="value">{total}</div>
            </div>
        </div>

        <div class="matrix-container">
            <table>
                <tr>
                    <th colspan="2" rowspan="2" style="border: none; background: transparent;"></th>
                    <th colspan="2">Predicted Condition</th>
                </tr>
                <tr>
                    <th>Cancer</th>
                    <th>Non-Cancer</th>
                </tr>
                <tr>
                    <th rowspan="2" class="true-class">Actual<br/>Condition</th>
                    <th>Cancer</th>
                    <td class="cell-tp">
                        <div class="cell-value">{tp}</div>
                        <div class="cell-label">True Positive</div>
                    </td>
                    <td class="cell-fn">
                        <div class="cell-value">{fn}</div>
                        <div class="cell-label">False Negative</div>
                    </td>
                </tr>
                <tr>
                    <th>Non-Cancer</th>
                    <td class="cell-fp">
                        <div class="cell-value">{fp}</div>
                        <div class="cell-label">False Positive</div>
                    </td>
                    <td class="cell-tn">
                        <div class="cell-value">{tn}</div>
                        <div class="cell-label">True Negative</div>
                    </td>
                </tr>
            </table>
        </div>

        <div style="margin-top: 40px; text-align: center; color: #666; font-size: 14px;">
            <p><strong>Cancer Pred Accuracy:</strong> {cancer_acc:.1f}% &nbsp;&nbsp;|&nbsp;&nbsp; <strong>Non-Cancer Pred Accuracy:</strong> {non_cancer_acc:.1f}%</p>
            <p style="margin-top: 10px;">Select &amp; Copy these metrics directly for your poster or presentations.</p>
        </div>
    </div>

    <!-- DOM-to-Image Library for Native Rasterization -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/2.6.0/dom-to-image.min.js"></script>
    <script>
        function exportSVG() {{
            domtoimage.toSvg(document.querySelector('.container'), {{ bgcolor: '#ffffff' }})
                .then(function (dataUrl) {{
                    var link = document.createElement('a');
                    link.download = 'metrics_dashboard.svg';
                    link.href = dataUrl;
                    link.click();
                }});
        }}
        function exportPNG() {{
            var node = document.querySelector('.container');
            var scale = 3; // Sharp high-res for poster
            var param = {{
                height: node.offsetHeight * scale,
                width: node.offsetWidth * scale,
                quality: 1,
                style: {{
                    transform: 'scale(' + scale + ')',
                    transformOrigin: 'top left',
                    width: node.offsetWidth + 'px',
                    height: node.offsetHeight + 'px'
                }},
                bgcolor: '#ffffff'
            }};
            domtoimage.toPng(node, param)
                .then(function (dataUrl) {{
                    var link = document.createElement('a');
                    link.download = 'metrics_dashboard.png';
                    link.href = dataUrl;
                    link.click();
                }});
        }}
    </script>
</body>
</html>
"""
    with open("metrics_dashboard.html", "w") as f:
        f.write(html)
    print("✅ Successfully generated metrics_dashboard.html")

if __name__ == "__main__":
    print("Starting visual metrics script. Initializing NLP and Polars...")
    nlp = get_nlp()
    df = pl.read_excel("data/manual_label_not_mouse.xlsx")

    label_cols = ["run_accession", "is_cancer"]
    if "cancer_type" in df.columns:
        label_cols.append("cancer_type")
    labeled_df = df.select([c for c in label_cols if c in df.columns])

    combined_df = pl.read_csv(
        "data/combined_metadata_noncancer_removed.csv",
        infer_schema_length=10000,
        ignore_errors=True,
    ).unique(subset=["run_accession"], keep="first")
    
    if "cancer_type" in combined_df.columns and "cancer_type" in labeled_df.columns:
        combined_df = combined_df.drop("cancer_type")

    df = labeled_df.join(combined_df, on="run_accession", how="left")
    ground_truth = df.select("run_accession", "is_cancer")
    df = df.drop("is_cancer", "cancer_type")

    predicted_df = classify_cancer_samples(df, fallback_providers=False)
    predicted_df = predicted_df.join(ground_truth, on="run_accession", how="left")
    predicted_df = predicted_df.with_columns(
        pl.col("confidence_category").alias("final_classification")
    )

    cancer_classes = ["confident_cancer", "confirmed_by_medspacy", "likely_cancer"]
    predicted_df = predicted_df.with_columns(
        pl.when(pl.col("final_classification").is_in(cancer_classes))
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("predicted_cancer")
    )
    predicted_df = predicted_df.with_columns(
        pl.when(pl.col("is_cancer") >= 1)
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("actual_cancer")
    )

    tp = predicted_df.filter((pl.col("predicted_cancer") == 1) & (pl.col("actual_cancer") == 1)).height
    fp = predicted_df.filter((pl.col("predicted_cancer") == 1) & (pl.col("actual_cancer") == 0)).height
    tn = predicted_df.filter((pl.col("predicted_cancer") == 0) & (pl.col("actual_cancer") == 0)).height
    fn = predicted_df.filter((pl.col("predicted_cancer") == 0) & (pl.col("actual_cancer") == 1)).height

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    cancer_acc = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    non_cancer_acc = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

    print(f"Metrics Calulcated: Accuracy: {accuracy:.2f}% | Precision: {precision:.2f}% | Recall: {recall:.2f}%")
    
    build_html_report(tp, fp, tn, fn, accuracy, precision, recall, cancer_acc, non_cancer_acc)

    os.makedirs("outputs", exist_ok=True)
    predicted_df.write_csv("outputs/all_predictions.csv")
    print("Done! Open metrics_dashboard.html to view the beautiful interactive metrics.")
