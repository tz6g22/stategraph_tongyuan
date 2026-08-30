import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_state_transition_2case_test'; METHODS=['stategraph_v4','graphiti','mem0','letta']
sys.path.insert(0,str(ROOT/'evaluation_protocol')); from agent_memory_comparison_common import load_answer_config,answer_messages,bounded_context,call_deepseek,canonical_sha256,write_jsonl,sha256_bytes
def read(p):
 t=Path(p).read_text()
 try:
  x=json.loads(t); return x if isinstance(x,list) else [x]
 except: return [json.loads(z) for z in t.splitlines() if z.strip()]
cfg,_,ch=load_answer_config()
for m in METHODS:
 src=OUT/'stategraph_v4/stategraph_v4/runs/memoryagentbench_conflict/retrieval.jsonl' if False else OUT/'stategraph_v4/runs/memoryagentbench_conflict/retrieval.jsonl' if m=='stategraph_v4' else OUT/m/'retrieval.jsonl'; preds=[]
 for r in read(src):
  c,b=bounded_context(r.get('retrieved_context',[]),cfg); msgs=answer_messages(r['question'],c,cfg); a,meta=call_deepseek(msgs,cfg); preds.append({'case_id':r['case_id'],'question':r['question'],'final_answer':a,'answer_config_sha256':ch,'response_metadata':meta,'retrieval_sha256':canonical_sha256(r)})
 p=write_jsonl(OUT/m/'predictions.jsonl',preds); (OUT/m/'predictions.seal.json').write_text(json.dumps({'case_count':len(preds),'predictions_sha256':sha256_bytes(p),'answer_config_sha256':ch,'gold_loaded_during_generation':False},indent=2)); print(m,'sealed')
