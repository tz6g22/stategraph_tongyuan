import json,hashlib
from pathlib import Path
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_state_transition_small_test'
OUT.mkdir(parents=True,exist_ok=True)
spec=[(4,52),(4,81),(5,51)]
rows=pq.read_table('/home/cody/data/memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet',columns=['context','questions','metadata']).to_pylist()
cases=[]; mem=[]; mem_by_id={}
for ri,qi in spec:
    r=rows[ri]; mid=f'memoryagentbench-conflict-row{ri}'; text=r['context']
    if mid not in mem_by_id:
        mem_by_id[mid]={'memory_id':mid,'origin':'MemoryAgentBench/Conflict_Resolution','observations':[{'timestamp':'2025-01-01T00:00:00+00:00','text':text}],'content_sha256':hashlib.sha256(text.encode()).hexdigest()}; mem.append(mem_by_id[mid])
    cases.append({'case_id':f'row{ri}-question{qi}','memory_id':mid,'question':r['questions'][qi],'question_time':'2025-01-02T00:00:00+00:00','question_index':qi,'row_index':ri})
scope={'dataset':'MemoryAgentBench','split':'Conflict Resolution','case_ids':[c['case_id'] for c in cases],'memory_ids':[m['memory_id'] for m in mem],'methods':['stategraph_v4','graphiti','mem0','letta'],'selection_basis':'fixed offline mechanism-targeted cases selected from direct current-state conflict slots; full context trajectory retained; answers excluded','gold_loaded_during_generation':False}
payload={'dataset':'MemoryAgentBench-Conflict-Resolution','selection':scope['selection_basis'],'cases':cases,'memory_groups':mem,'gold_fields_present':False}
(OUT/'prepared_gold_free.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
scope['prepared_sha256']=hashlib.sha256((OUT/'prepared_gold_free.json').read_bytes()).hexdigest()
(OUT/'MECHANISM_SCOPE.json').write_text(json.dumps(scope,ensure_ascii=False,indent=2))
