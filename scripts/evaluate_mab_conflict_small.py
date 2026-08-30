import json,re,string
from collections import Counter
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_conflict_small_test'
METHODS=['stategraph_v4','graphiti','mem0','letta']
def norm(s):
    s=s.lower(); s=''.join(c for c in s if c not in string.punctuation); s=re.sub(r'\b(a|an|the)\b',' ',s); return ' '.join(s.split())
def f1(a,b):
    x,y=norm(a).split(),norm(b).split()
    if not x or not y: return float(x==y)
    o=sum((Counter(x)&Counter(y)).values())
    return 0 if not o else 2*(o/len(x))*(o/len(y))/((o/len(x))+(o/len(y)))
def main():
    answers=pq.read_table('/home/cody/data/memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet',columns=['answers']).slice(0,1).to_pylist()[0]['answers']
    gold={f'row0-question{i}':([v] if isinstance(v,str) else [str(x) for x in v]) for i,v in enumerate(answers)}
    out={}; failures=[]
    for m in METHODS:
        preds=[json.loads(x) for x in (OUT/m/'predictions.jsonl').read_text().splitlines() if x.strip()]; rows=[]
        for p in preds:
            refs=gold[p['case_id']]; a=p['final_answer']; sub=max(float(norm(r) in norm(a)) for r in refs); ex=max(float(norm(r)==norm(a)) for r in refs); ff=max(f1(a,r) for r in refs)
            rows.append({'case_id':p['case_id'],'prediction':a,'gold':refs,'substring_exact_match':sub,'exact_match':ex,'token_f1':ff})
            if not sub: failures.append({'method':m,'case_id':p['case_id'],'category':'generation','prediction':a,'gold':refs})
        out[m]={'completed_cases':len(rows),'planned_cases':3,'primary_metric':'substring_exact_match','metrics':{'primary':sum(x['substring_exact_match'] for x in rows)/3,'EM':sum(x['exact_match'] for x in rows)/3,'F1':sum(x['token_f1'] for x in rows)/3},'cases':rows}
    (OUT/'evaluation.json').write_text(json.dumps({'status':'SMALL CONFLICT DIAGNOSTIC','gold_loaded_after_all_prediction_seals':True,'results':out},ensure_ascii=False,indent=2))
    (OUT/'failure_analysis.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in failures))
    print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
