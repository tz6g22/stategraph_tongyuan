import json,re,string
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_state_transition_2case_test'; IDS=['row4-question52','row4-question81']; METHODS=['stategraph_v4','graphiti','mem0','letta']
def n(s): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in s.lower() if c not in string.punctuation)).split())
def f(a,b):
 x,y=n(a).split(),n(b).split(); o=sum((Counter(x)&Counter(y)).values()); return 0 if not o else 2*(o/len(x))*(o/len(y))/((o/len(x))+(o/len(y)))
rows=pq.read_table('/home/cody/data/memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet',columns=['answers']).to_pylist(); ans=rows[4]['answers']; gold={'row4-question52':ans[52],'row4-question81':ans[81]}; out={}
for m in METHODS:
 ps=[json.loads(x) for x in (OUT/m/'predictions.jsonl').read_text().splitlines() if x.strip()]; rr=[]
 for p in ps:
  g=gold[p['case_id']]; refs=g if isinstance(g,list) else [g]; a=p['final_answer']; rr.append({'case_id':p['case_id'],'prediction':a,'gold':refs,'primary':max(float(n(x) in n(a)) for x in refs),'EM':max(float(n(x)==n(a)) for x in refs),'F1':max(f(a,x) for x in refs)})
 out[m]={'completed_cases':len(rr),'metrics':{k:sum(x[k] for x in rr)/len(rr) for k in ['primary','EM','F1']},'cases':rr}
(OUT/'evaluation.json').write_text(json.dumps({'gold_loaded_after_all_prediction_seals':True,'results':out},ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
