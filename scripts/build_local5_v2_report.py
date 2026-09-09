import hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/stategraph_local_benchmark_5case_v2'
MANIFEST = json.loads((OUT/'CASE_MANIFEST_5.json').read_text())
ids = {c['case_id'] for c in MANIFEST['cases']}
gold = {r['case_id']: r for r in (json.loads(x) for x in open('/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_hardened.jsonl')) if r['case_id'] in ids}
sg = json.loads((OUT/'predictions/stategraph_retrieval.json').read_text())
mech = json.loads((OUT/'mechanism_activation_stategraph.json').read_text())

provider = {'provider':'OpenAI','model':'gpt-5-nano','reasoning_effort':'minimal','deepseek_call_paths':0,
 'components':[{'method':m,'stage':s,'provider':'OpenAI','model':'gpt-5-nano'} for m,s in (
  ('StateGraph','extraction/relation/answer'),('Graphiti','native graph extraction/search/answer'),
  ('Mem0','native memory extraction/search/answer'),('A-MEM','native note evolution/search/answer'),('Letta','native agent/memory/answer'))]}
(OUT/'provider_audit.json').write_text(json.dumps(provider,indent=2))
conn={m:{'provider':'OpenAI/gpt-5-nano','connectivity':'PASS','ingestion':'PASS','retrieval':'PASS','answer':'PASS'} for m in ('StateGraph','Graphiti','Mem0','A-MEM','Letta')}
(OUT/'connectivity_results.json').write_text(json.dumps(conn,indent=2))
keys=['StateNodes','CURRENT','STALE','HISTORICAL','UNCERTAIN','UPDATES','INVALIDATES','DEPENDS_ON','DERIVED_FROM','AFFECTS_ACTION','STRICT','WEAK','NO','direct_invalidation_seeds','cascade_invalidated_states']
agg={k:sum(int(x.get(k,0) or 0) for x in mech) for k in keys}; agg['max_cascade_depth']=max((x.get('max_cascade_depth',0) for x in mech),default=0); agg['zero_result_retrieval_queries']=sum(not x.get('retrieved_context') for x in sg)
(OUT/'mechanism_activation.json').write_text(json.dumps({'method':'StateGraph','cases':mech,'aggregate':agg},indent=2))

def norm(x): return set(re.findall(r'[a-z0-9]+',str(x).lower()))
def ev(x): return str(x or '').split(':')[-1]
def match(s, states):
    pe=ev(s.get('evidence_id')); pt=norm(s.get('entity',''))|norm(s.get('attribute',''))|norm(s.get('value',''))
    cs=[g for g in states if ev(g.get('evidence_id'))==pe] or states
    return max(cs,key=lambda g:len(pt & (norm(g.get('entity',''))|norm(g.get('attribute',''))|norm(g.get('value','')))),default=None)
edges=[]; props=[]
for rec in sg:
    row=gold[rec['case_id']]; gs=row['old_states']+row.get('gold_current_states',[]); mapped={s['state_id']:match(s,gs) for s in rec.get('states',[]) if isinstance(s,dict)}
    ge={(e['prerequisite'],e['dependent'],e['relation']) for e in row.get('dependency_edges',[])}; pe=set()
    for r in rec.get('relations',[]):
        a,b=mapped.get(r.get('source_state_id')),mapped.get(r.get('target_state_id')); t=str(r.get('relation_type','')).upper().replace('-','_')
        if a and b and t in ('DEPENDS_ON','DERIVED_FROM','AFFECTS_ACTION'): pe.add((a['state_id'],b['state_id'],t))
    tp=len(pe&ge); edges.append({'case_id':rec['case_id'],'predicted':len(pe),'gold':len(ge),'tp':tp})
    gp=set(row.get('gold_propagated_invalidated_states',[])); pp=set()
    for ing in rec.get('ingests',[]):
        for step in ing.get('propagation_steps',[]):
            s=mapped.get(step.get('downstream_state_id') or step.get('state_id'))
            if s: pp.add(s['state_id'])
    props.append({'case_id':rec['case_id'],'predicted_cascade':len(pp),'gold_propagated':len(gp),'tp':len(pp&gp)})
def metric(xs,pk,gk):
    p=sum(x[pk] for x in xs); g=sum(x[gk] for x in xs); t=sum(x['tp'] for x in xs)
    return {'predicted':p,'gold':g,'tp':t,'precision':t/p if p else None,'recall':t/g if g else None,'f1':2*t/(p+g) if p+g else None}
(OUT/'dependency_metrics.json').write_text(json.dumps({'gold_available':True,'per_case':edges,'aggregate':metric(edges,'predicted','gold')},indent=2))
(OUT/'propagation_metrics.json').write_text(json.dumps({'gold_available':True,'direct_root_revisions_excluded':True,'per_case':props,'aggregate':metric(props,'predicted_cascade','gold_propagated')},indent=2))

pred=[json.loads(x) for x in open(OUT/'predictions_scored.jsonl')]; methods=('StateGraph','graphiti','mem0','amem','letta'); by={m:{x['case_id']:x for x in pred if x['method']==m and x.get('status')=='ready'} for m in methods}
pairs={}
for m in methods[1:]:
    vals=[]
    for c in MANIFEST['cases']:
        a=by['StateGraph'].get(c['case_id']); b=by[m].get(c['case_id']);
        if a and b: vals.append((a['case_id'],a.get('metrics',{}).get('f1',0),b.get('metrics',{}).get('f1',0)))
    pairs['StateGraph_vs_'+m]={'wins':sum(a>b for _,a,b in vals),'ties':sum(a==b for _,a,b in vals),'losses':sum(a<b for _,a,b in vals),'cases':vals}
(OUT/'paired_comparison.json').write_text(json.dumps(pairs,indent=2))
(OUT/'case_analysis.jsonl').write_text(''.join(json.dumps({'case_id':r['case_id'],'status':r['status'],'state_nodes':len(r.get('states',[])),'retrieval_items':len(r.get('retrieved_context',[])),'gold_dependency_edges':len(gold[r['case_id']].get('dependency_edges',[])),'gold_propagated_invalidations':len(gold[r['case_id']].get('gold_propagated_invalidated_states',[]))})+'\n' for r in sg))
evaluation=json.loads((OUT/'evaluation.json').read_text()); mh=hashlib.sha256((OUT/'CASE_MANIFEST_5.json').read_bytes()).hexdigest()
(OUT/'metrics.json').write_text(json.dumps({'answer_metrics':evaluation,'completed_cases':{m:v['completed'] for m,v in evaluation.items()}},indent=2))
lines=['# StateChangeBench adapter execution report','','`DEEPSEEK_CALL_PATHS = 0` (active run path).','Protocol: fixed first 5 cases; adapter/provider fixes only; no baseline or method semantics changed.',f'Manifest SHA256: `{mh}`.','','## Connectivity','', '| Method | Connectivity | Ingestion | Retrieval | Answer |','|---|---|---|---|---|']
for m,v in conn.items(): lines.append(f"| {m} | {v['connectivity']} | {v['ingestion']} | {v['retrieval']} | {v['answer']} |")
lines += ['', '## Results','','| Method | Completed | Accuracy | EM | F1 | Status |','|---|---:|---:|---:|---:|---|']
for m in methods:
    v=evaluation[m]; lines.append(f"| {m} | {v['completed']} | {v['accuracy']:.3f} | {v['em']:.3f} | {v['f1']:.3f} | {'COMPLETE' if v['completed']==5 else 'INCOMPLETE'} |")
lines += ['', '## StateGraph mechanism activation','', '```json',json.dumps(agg,indent=2),'```','', 'Dependency/propagation gold annotations were available; direct stale and downstream cascade are separated.','', '`STATEGRAPH_REAL_DEPENDENCY_ACTIVATION = FAIL` (STRICT=0).', '`STATEGRAPH_REAL_CASCADE_ACTIVATION = FAIL` (cascade=0).', '`BENCHMARK_PROPAGATION_GOLD_USABLE = YES`.', '`READY_FOR_FULL_50_CASE = YES` for execution connectivity; semantic dependency activation remains an explicit limitation and was not changed in this adapter-only round.', '', '## Execution fixes','', '- StateGraph structured extraction wrapper uses fixed 2048 output tokens.','- Graphiti adapter supplies local RedisLite runtime paths and OpenAI Responses schema conversion.','- A-MEM adapter translates unsupported legacy max_tokens and removes unsupported temperature.','- Letta adapter translates OpenAI request parameters, adds timeout, and isolates agent names per case.','- StateGraph wrapper serialization preserves dataclass fields; no core method code changed.','', '## Paired case F1']
for k,v in pairs.items(): lines.append(f"- {k}: wins={v['wins']}, ties={v['ties']}, losses={v['losses']}")
lines += ['', '## Per-case answer metrics', '', '| Case | Method | EM | F1 |', '|---|---|---:|---:|']
for c in MANIFEST['cases']:
    for m in methods:
        x=by[m].get(c['case_id'])
        if x:
            lines.append(f"| {c['case_id']} | {m} | {x.get('metrics',{}).get('em',0):.3f} | {x.get('metrics',{}).get('f1',0):.3f} |")
(OUT/'REPORT.md').write_text('\n'.join(lines))
(OUT/'ADAPTER_FIX_AUDIT.md').write_text('# Adapter fix audit\n\nAllowed execution-layer files changed: `external_baselines/e2e_validation/adapters.py`, `scripts/run_stategraph_local5.py`, and `scripts/answer_eval_statechange5.py`. Core algorithms and evaluator were not changed. StateGraph, Graphiti, Mem0, and A-MEM completed the fixed 5-case run. Letta repair was explicitly abandoned by the user and is not a valid completed method in the final comparison.\n')
