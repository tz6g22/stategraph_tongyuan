from pathlib import Path
import json,hashlib,sys,asyncio
try:
 import pyarrow.parquet as pq
except ModuleNotFoundError:
 pq=None
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_state_transition_2case_test'; OUT.mkdir(parents=True,exist_ok=True)
if pq:
 rows=pq.read_table('/home/cody/data/memoryagentbench/data/Conflict_Resolution-00000-of-00001.parquet',columns=['context','questions']).to_pylist(); r=rows[4]; text=r['context']; questions=r['questions']
else:
 old=json.loads((ROOT/'outputs/stategraph_v4_memoryagentbench_state_transition_small_test/prepared_gold_free.json').read_text()); text=old['memory_groups'][0]['observations'][0]['text']; questions=[None]*100
 # fixed questions are only used in the gold-free manifest; recover them from the frozen selection file
 questions[52]='What is the name of the current head of state in United States of America?'; questions[81]='What is the name of the current head of state in Soviet Union?'
mid='memoryagentbench-conflict-row4'; cases=[{'case_id':'row4-question52','memory_id':mid,'question':questions[52],'question_time':'2025-01-02T00:00:00+00:00','question_index':52,'row_index':4},{'case_id':'row4-question81','memory_id':mid,'question':questions[81],'question_time':'2025-01-02T00:00:00+00:00','question_index':81,'row_index':4}]
mem={'memory_id':mid,'origin':'MemoryAgentBench/Conflict_Resolution','observations':[{'timestamp':'2025-01-01T00:00:00+00:00','text':text}],'content_sha256':hashlib.sha256(text.encode()).hexdigest()}; payload={'dataset':'MemoryAgentBench-Conflict-Resolution','selection':'fixed row4 transition cases; complete trajectory; gold excluded','cases':cases,'memory_groups':[mem],'gold_fields_present':False}; prep=OUT/'prepared_gold_free.json'; prep.write_text(json.dumps(payload,ensure_ascii=False,indent=2)); (OUT/'MECHANISM_SCOPE.json').write_text(json.dumps({'case_ids':[c['case_id'] for c in cases],'memory_id':mid,'methods':['stategraph_v4','graphiti','mem0','letta'],'gold_loaded_during_generation':False,'prepared_sha256':hashlib.sha256(prep.read_bytes()).hexdigest()},indent=2))
def sg():
 sys.path.insert(0,str(ROOT)); from stategraph.evaluation import run_stategraph_10 as x; x.RUN_ROOT=OUT/'stategraph_v4'; x.DATASET_FILES={'memoryagentbench_conflict':prep}; asyncio.run(x._run('memoryagentbench_conflict'))
def base(name):
 sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'outputs/stategraph_formal_v3_4x10_20260824/scripts')); import formal_baseline_isolated as f; od=OUT/name; f.run_memory(name,prep,od,mid); shard=od/'shards'/mid; lines=(shard/'retrieval.jsonl').read_text(); (od/'retrieval.jsonl').write_text(lines); (od/'ingestion_trace.json').write_text((shard/'ingestion_trace.json').read_text())
if __name__=='__main__': sg()
