import os
import re

DIR = r"C:\Users\Hp\Downloads\Louzini_Manuscript\Thesis_Final"
PATTERNS = [
    (r"\b93\s*%", "93% success or rate"),
    (r"\b93\b", "93 reference"),
    (r"\b95\s*%", "95% success or rate"),
    (r"four\s+controller\s+variants", "four controller variants"),
    (r"four\s+variants", "four variants"),
    (r"pending\s+campaign", "pending campaign"),
    (r"pending\s+No-Energy", "pending No-Energy"),
]

def audit():
    for f in os.listdir(DIR):
        if not f.endswith(".tex"):
            continue
        path = os.path.join(DIR, f)
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            content = file.read()
            for pattern, desc in PATTERNS:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    print(f"[audit] Found '{desc}' in {f}: {len(matches)} occurrences")
                    # Find context
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        start = max(0, match.start() - 40)
                        end = min(len(content), match.end() + 40)
                        print(f"    Context: ...{content[start:end].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    audit()
