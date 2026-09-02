import argparse
import json
from src.datasets import build

parser = argparse.ArgumentParser(description='Validate a pinned dataset before atomic publication')
parser.add_argument('--manifest',default='sources/fixture.json')
parser.add_argument('--database',default='data/ashe.db')
args = parser.parse_args()
print(json.dumps(build(args.manifest,args.database),indent=2))
