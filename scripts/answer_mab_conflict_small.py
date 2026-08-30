from __future__ import annotations
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_conflict_small_test'; CASES=['row0-question4','row0-question6','row0-question0']; METHODS=['stategraph_v4','graphiti','mem0','letta']
sys.path.insert(0,str(ROOT/'evaluation_protocol'))
from agent_memory_comparison_common import load_answer_config,answer_messages,bounded_context,call_deepseek,canonical_sha256,write_jsonl,sha256_bytes
def read(p):
 text=Path(p).read_text()
 try:
  value=json.loads(text)
  return value if isinstance(value,list) else [value]
 except json.JSONDecodeError:
  return [json.loads(x) for x in text.splitlines() if x.strip()]
def main():
 cfg,cfgbytes,cfgh=load_answer_config(); atomic=[]
 for m in METHODS:
  if (OUT/m/'predictions.seal.json').exists():
   print(m,'already sealed'); continue
  src=OUT/'stategraph_v4/runs/memoryagentbench_conflict/retrieval.jsonl' if m=='stategraph_v4' else OUT/m/'retrieval.jsonl'
  records=read(src); md=OUT/m; md.mkdir(exist_ok=True)
  preds=[]
  for r in records:
   context=r.get('retrieved_context',[]); bounded,budget=bounded_context(context,cfg); msgs=answer_messages(r['question'],bounded,cfg)
   ans,meta=call_deepseek(msgs,cfg)
   preds.append({'dataset':'MemoryAgentBench-Conflict-Resolution','baseline':m,'case_id':r['case_id'],'question':r['question'],'final_answer':ans,'answer':ans,'model':cfg['model']['name'],'answer_config_sha256':cfgh,'prompt_sha256':canonical_sha256(msgs),'retrieval_sha256':canonical_sha256(r),'context_budget':budget,'response_metadata':meta})
  payload=write_jsonl(md/'predictions.jsonl',preds)
  (md/'predictions.seal.json').write_text(json.dumps({'case_count':len(preds),'predictions_sha256':sha256_bytes(payload),'answer_config_sha256':cfgh,'gold_loaded_during_generation':False},indent=2))
  print(m,'sealed',len(preds))
if __name__=='__main__': main()
