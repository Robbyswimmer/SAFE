import json
import argparse
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    args = parser.parse_args()
    
    subsets = Counter()
    total = 0
    
    print(f"Scanning {args.file}...")
    with open(args.file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    subsets[item.get('subset', 'unknown')] += 1
                    total += 1
                except:
                    pass
                    
    print(f"Total samples: {total}")
    print("Subsets:")
    for k, v in subsets.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
