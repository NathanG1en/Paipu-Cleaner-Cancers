import spacy
from spacy import displacy
from functions import get_nlp

# The highly-descriptive sample record we identified
SAMPLE_TEXT = "Primary osteosarcoma of the left humerus; no evidence of metastatic disease in the lungs."

def generate_visual():
    print("Loading MedSpaCy pipeline...")
    nlp = get_nlp()
    
    print(f"Processing text: '{SAMPLE_TEXT}'")
    doc = nlp(SAMPLE_TEXT)
    
    # --- ENHANCEMENT: Manually expose negation logic to displacy ---
    new_ents = []
    # Identify negation triggers first if possible (ConText stores them in doc._.context_graph)
    context_graph = getattr(doc._, "context_graph", None)
    if context_graph:
        for modifier in context_graph.modifiers:
            new_ents.append(spacy.tokens.Span(doc, modifier._start, modifier._end, label="NEGATION"))

    for ent in doc.ents:
        label = ent.label_
        if getattr(ent._, "is_negated", False):
            label = "NEGATED_CANCER"
        new_ents.append(spacy.tokens.Span(doc, ent.start, ent.end, label=label))
    
    # Sort and remove overlaps (negation triggers can sometimes overlap with other things)
    new_ents = spacy.util.filter_spans(new_ents)
    doc.ents = new_ents

    # Define colors matching the research poster aesthetic
    colors = {
        "CANCER": "#cde3f3",       # Pastel Blue
        "NON_CANCER": "#b5dcd0",   # Pastel Green
        "NEGATED_CANCER": "#fecdd3", # Soft Rose/Pink
        "NEGATION": "#e2e8f0"      # Soft Gray for the trigger itself
    }
    options = {"colors": colors, "compact": True}
    
    # Generate the raw DisplaCy HTML
    html_content = displacy.render(doc, style="ent", page=False, options=options)
    
    # Professional Template Wrap
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedSpaCy NER Visualization</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        
        body {{
            font-family: 'Inter', sans-serif;
            background-color: #ffffff;
            margin: 0;
            padding: 60px 40px;
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #1e293b;
        }}
        
        .container {{
            max-width: 900px;
            width: 100%;
            background: #fff;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
            position: relative;
        }}
        
        h1 {{
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 20px;
            text-align: center;
            color: #0f172a;
        }}
        
        .viz-wrapper {{
            font-size: 1.25rem;
            line-height: 2.2;
            padding: 20px 0;
        }}
        
        /* DisplaCy overrides for better poster look */
        .entities {{
            line-height: 2.5 !important;
        }}
        
        [data-label] {{
            font-weight: 600 !important;
            border-radius: 4px !important;
            padding: 0.2em 0.4em !important;
        }}
        
        [data-label]::after {{
            font-size: 0.6em !important;
            margin-left: 0.5rem !important;
            opacity: 0.8;
        }}

        .caption {{
            margin-top: 30px;
            font-size: 14px;
            color: #64748b;
            text-align: center;
            line-height: 1.6;
        }}

        /* Export Controls */
        .export-controls {{
            position: fixed;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 1000;
        }}
        .export-btn {{
            background: #f8fafc;
            border: 1px solid #cbd5e1;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            color: #334155;
            transition: all 0.2s;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}
        .export-btn:hover {{ background: #f1f5f9; border-color: #94a3b8; }}

    </style>
</head>
<body>
    <div class="export-controls">
        <button class="export-btn" onclick="exportSVG()">📥 SVG</button>
        <button class="export-btn" onclick="exportPNG()">📥 PNG (High-Res)</button>
    </div>

    <div class="container" id="viz-container">
        <h1>Context-Aware Cancer Entity Extraction</h1>
        <div class="viz-wrapper">
            {html_content}
        </div>
        <p class="caption">
            <strong>Sample Source:</strong> GSM7807443 Title Metadata<br>
            Showing automatic extraction of clinical conditions and spatial context. 
            Note how the pipeline distinguishes between primary conditions and metastatic signals.
        </p>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/dom-to-image/2.6.0/dom-to-image.min.js"></script>
    <script>
        function exportSVG() {{
            const node = document.getElementById('viz-container');
            domtoimage.toSvg(node, {{ bgcolor: '#ffffff' }})
                .then(function (dataUrl) {{
                    var link = document.createElement('a');
                    link.download = 'ner_visual.svg';
                    link.href = dataUrl;
                    link.click();
                }});
        }}
        function exportPNG() {{
            const node = document.getElementById('viz-container');
            const scale = 3;
            
            // Fix "wonky" width by capturing current actual dimensions
            const originalWidth = node.offsetWidth;
            const originalHeight = node.offsetHeight;

            const param = {{
                height: originalHeight * scale,
                width: originalWidth * scale,
                quality: 1,
                style: {{
                    transform: 'scale(' + scale + ')',
                    transformOrigin: 'top left',
                    width: originalWidth + 'px',
                    height: originalHeight + 'px',
                    margin: '0',
                    boxShadow: 'none'
                }},
                bgcolor: '#ffffff'
            }};
            
            domtoimage.toPng(node, param)
                .then(function (dataUrl) {{
                    var link = document.createElement('a');
                    link.download = 'ner_visual.png';
                    link.href = dataUrl;
                    link.click();
                }})
                .catch(function (error) {{
                    console.error('Export failed', error);
                }});
        }}
    </script>
</body>
</html>
"""
    with open("ner_visual.html", "w") as f:
        f.write(full_html)
    
    print("\n✅ Success! NER visualization saved to: ner_visual.html")
    print("\nIdentified Entities:")
    for ent in doc.ents:
        neg = " [NEGATED]" if getattr(ent._, "is_negated", False) else ""
        print(f" - {ent.text} ({ent.label_}){neg}")

if __name__ == "__main__":
    generate_visual()
