import re
import sys
from io import StringIO
import pandas as pd

csv_snippet_re = re.compile('<csvsnippet>(.*?)</csvsnippet>', re.DOTALL)
filenames = sys.argv[1:]

csvs = []
for f in filenames:
  with open(f, 'r') as f_csv:
    text = f_csv.read()
    snippets = csv_snippet_re.findall(text)
    if len(snippets) > 1:
        raise ValueError()
    csvs.append(pd.read_csv(StringIO(snippets[0])))

combined_csv = pd.concat(csvs, axis=0, sort=True).copy()
combined_csv_str = combined_csv.to_csv(index=False)
sys.stdout.write(combined_csv_str)
