import base64
import zlib
import urllib.request
import os

def encode_kroki(text):
    return base64.urlsafe_b64encode(zlib.compress(text.encode('utf-8'), 9)).decode('ascii')

def download_kroki(text, outfile, outformat='svg'):
    encoded = encode_kroki(text)
    url = f"https://kroki.io/mermaid/{outformat}/{encoded}"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response, open(outfile, 'wb') as out_file:
            out_file.write(response.read())
        print(f"✅ Saved {outfile}")
    except Exception as e:
        print(f"❌ Failed to download {outfile}: {e}")

def main():
    print("Generating diagrams via Kroki API...")
    
    # 1. Poster Diagram
    if os.path.exists("poster_diagram.mmd"):
        with open("poster_diagram.mmd", "r") as f:
            poster_text = f.read()
        download_kroki(poster_text, "poster_diagram.svg", "svg")
        download_kroki(poster_text, "poster_diagram.png", "png")
        
    # 2. High Level Diagram
    if os.path.exists("high_level.mmd"):
        with open("high_level.mmd", "r") as f:
            high_text = f.read()
        download_kroki(high_text, "high_level.svg", "svg")
        download_kroki(high_text, "high_level.png", "png")

if __name__ == "__main__":
    main()
