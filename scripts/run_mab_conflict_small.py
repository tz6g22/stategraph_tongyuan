from __future__ import annotations
import asyncio, hashlib, json, shutil, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/stategraph_v4_memoryagentbench_conflict_small_test'
SRC=ROOT/'outputs/stategraph_v3_small_scale_diagnostic/prepared_gold_free/memoryagentbench_conflict.json'
CASE_IDS=['row0-question4','row0-question6','row0-question0']
METHODS=['stategraph_v4','graphiti','mem0','letta']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
 p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf8'); t.replace(p)

def prepare():
 d=json.loads(SRC.read_text()); cases=[c for c in d['cases'] if c['case_id'] in CASE_IDS]
 assert len(cases)==3
 mids=sorted({c['memory_id'] for c in cases}); mem=[m for m in d['memory_groups'] if m['memory_id'] in mids]
 scope={'run_type':'CONFLICT-RESOLUTION SMALL TEST','dataset':'MemoryAgentBench','split':'Conflict Resolution','selection_basis':'first three prepared row-0 questions in frozen gold-free input; fixed before any API call; no answer/gold inspection','case_ids':CASE_IDS,'memory_ids':mids,'memory_observation_counts':{m['memory_id']:len(m['observations']) for m in mem},'methods':METHODS,'source_sha256':sha(SRC),'answer_config_sha256':sha(ROOT/'evaluation_protocol/shared_answer_generation.yaml'),'gold_loaded_during_generation':False}
 atomic(OUT/'SMALL_SCOPE.json',scope)
 atomic(OUT/'prepared_gold_free.json',{'dataset':d['dataset'],'selection':scope['selection_basis'],'cases':cases,'memory_groups':mem,'gold_fields_present':False,'source_sha256':sha(SRC)})
 return scope

def run_stategraph(prep):
 sys.path.insert(0,str(ROOT)); from stategraph.evaluation import run_stategraph_10 as r
 r.RUN_ROOT=OUT/'stategraph_v4'; r.DATASET_FILES={'memoryagentbench_conflict':prep}
 asyncio.run(r._run('memoryagentbench_conflict'))

def run_baseline(name,prep):
 sys.path.insert(0,str(ROOT))
 sys.path.insert(0,str(ROOT/'outputs/stategraph_formal_v3_4x10_20260824/scripts'))
 import formal_baseline_isolated as f
 od=OUT/name; f.run_memory(name,prep,od,'memoryagentbench-conflict-row0')
 shard=od/'shards/memoryagentbench-conflict-row0'; lines=[json.loads(x) for x in (shard/'retrieval.jsonl').read_text().splitlines() if x.strip()]
 atomic(od/'retrieval.jsonl',lines); atomic(od/'ingestion_trace.json',json.loads((shard/'ingestion_trace.json').read_text())); atomic(od/'memory_input.json',{'gold_fields_present':False,'memory_groups':json.loads(prep.read_text())['memory_groups']})

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--only',choices=['all','stategraph','graphiti','mem0','letta'],default='all'); a=ap.parse_args()
 scope=prepare(); prep=OUT/'prepared_gold_free.json'; atomic(OUT/'RUN_STATUS.json',{'status':'RUNNING','scope':scope})
 if a.only in ('all','stategraph'): run_stategraph(prep)
 if a.only in ('all','graphiti'): run_baseline('graphiti',prep)
 if a.only in ('all','mem0'): run_baseline('mem0',prep)
 if a.only in ('all','letta'): run_baseline('letta',prep)
 atomic(OUT/'RUN_STATUS.json',{'status':'RETRIEVAL_COMPLETE','scope':scope})
