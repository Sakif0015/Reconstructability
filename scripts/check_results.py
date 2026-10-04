import os, json, glob
for f in sorted(glob.glob("results/*.json")):
    try: json.load(open(f)); print("OK  ", f, os.path.getsize(f))
    except Exception as e: print("BAD ", f, type(e).__name__)