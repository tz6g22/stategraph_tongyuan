"""Offline evaluation after all six dataset/method prediction seals exist."""
from __future__ import annotations
import json, re, unicodedata, hashlib, asyncio
import sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__import__('os').environ.get('MATRIX_OUT', ROOT/'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1'))
sys.path.insert(0, str(ROOT))

def norm(s): return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',str(s)).casefold(),flags=re.UNICODE))
def f1(a,b):
    x,y=norm(a).split(),norm(b).split()
    if not x or not y:return float(x==y)
    o=sum((Counter(x)&Counter(y)).values())
    if not o:return 0.0
    p=o/len(x); r=o/len(y); return 2*p*r/(p+r)
def load_rows(ds,method):
    p=OUT/ds/f'{method}_predictions.jsonl'; return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def ref_map(ds):
    if ds=='statechangebench':
        source = Path(__import__('os').environ.get(
            'STATECHANGEBENCH_SOURCE',
            '/home/cody/data/stategraphbenchmark/statechangebench_cases_001_050_v3_all_easy.jsonl',
        ))
        return {x['case_id']:[x['gold_answer']] for x in map(json.loads,source.read_text().splitlines()) if x['case_id'] in {'SCB_012','SCB_013','SCB_017'}}
    if ds=='longmemeval':
        return {x['question_id']:[x['answer']] if isinstance(x['answer'],str) else x['answer'] for x in json.loads(Path('/home/cody/data/longmemeval/longmemeval_oracle.json').read_text()) if x['question_id'] in {'5d3d2817','7527f7e2','c960da58'}}
    if ds=='longmemeval_v2':
        return {x['id']:[x['answer']] if isinstance(x['answer'],str) else x['answer'] for x in (json.loads(line) for line in Path('/home/cody/data/longmemeval_v2/questions.jsonl').read_text().splitlines()) if x['id']=='0f970f01'}
    if ds=='mab_conflict':
        source = OUT/'mab_gold_after_seal.json'
        if not source.exists():
            source = ROOT/'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1/mab_gold_after_seal.json'
        return json.loads(source.read_text())
    return {}

async def memora_eval(predictions):
    from stategraph.evaluation.evaluate_predictions import _evaluate_memora
    # The released rubric protocol is reused unchanged; only sealed predictions enter it.
    return await _evaluate_memora(predictions, OUT/'memora')

def main():
    datasets=['statechangebench','stale','longmemeval','longmemeval_v2','memora','mab_conflict']; methods=['StateGraph','Mem0']; all_metrics={}; paired={}; failures=[]
    for ds in datasets:
        all_metrics[ds]={}; rows_by={m:load_rows(ds,m) for m in methods}
        manifest_path=OUT/ds/'input_manifest.json'
        manifest=json.load(open(manifest_path)) if manifest_path.exists() else {}
        planned_ids=manifest.get('case_ids') or [c.get('case_id') for c in manifest.get('cases',[])]
        for m,rows in rows_by.items():
            ready=[r for r in rows if r.get('prediction_status')=='ready']
            ready_ids={r.get('base_case_id',str(r.get('case_id','')).split('::',1)[0]) for r in ready}
            item={'planned_cases':len(planned_ids),'completed_cases':len(ready_ids),'status':'COMPLETE' if ready_ids and len(ready_ids)==len(planned_ids) else 'INCOMPLETE'}
            item['failures']=[{'case_id':r.get('case_id'),'error_class':r.get('error_class'),'error':r.get('error')} for r in rows if r.get('prediction_status')!='ready']
            if ds=='stale':
                item.update({'final_answer_accuracy':None,'normalized_em':None,'token_f1':None,'stale_premise_rejection':None,'metric_note':'Official STALE judge not run in this minimal provider diagnostic; native query predictions sealed.','completed_query_predictions':len(ready)})
            elif ds=='memora' and ready:
                # filled after loop below
                item.update({'final_answer_accuracy':None,'normalized_em':None,'token_f1':None,'primary_rubric':None})
            else:
                refs=ref_map(ds); case_scores=[]
                for r in ready:
                    vals=refs.get(r.get('case_id'),[]); scores=[(float(norm(r.get('answer',''))==norm(g)),f1(r.get('answer',''),g)) for g in vals] or [(0.0,0.0)]
                    em,ff=max(scores,key=lambda z:z[1]); case_scores.append({'case_id':r.get('case_id'),'exact_match':em,'token_f1':ff,'prediction':r.get('answer'),'references':vals})
                item.update({'normalized_em':sum(x['exact_match'] for x in case_scores)/len(case_scores) if case_scores else None,'token_f1':sum(x['token_f1'] for x in case_scores)/len(case_scores) if case_scores else None,'cases':case_scores})
            all_metrics[ds][m]=item
            failures.extend([{'dataset':ds,'method':m,**x} for x in item['failures']])
        # Memora released rubric evaluation, after both method seals.
        for m in methods:
            ready=[r for r in rows_by[m] if r.get('prediction_status')=='ready']
            if ds=='memora' and ready:
                result=asyncio.run(memora_eval(ready)); by_case={}
                for j in result.get('cases',[]): by_case.setdefault(j['case_id'],[]).append(bool(j['correct']))
                all_metrics[ds][m].update({'primary_rubric':result['metrics'].get('rubric_accuracy'),'memory_presence':result['metrics'].get('memory_presence_accuracy'),'forgetting_absence':result['metrics'].get('forgetting_absence_accuracy'),'rubric_evaluation':result,'case_rubric_scores':{k:sum(v)/len(v) for k,v in by_case.items()}})
        # paired rows use generic score where available; Memora uses rubric score per case.
        ids=planned_ids
        pr=[]
        for cid in ids:
            sg=next((r for r in rows_by['StateGraph'] if (r.get('base_case_id',r.get('case_id'))==cid) and r.get('prediction_status')=='ready'),None); mm=next((r for r in rows_by['Mem0'] if (r.get('base_case_id',r.get('case_id'))==cid) and r.get('prediction_status')=='ready'),None)
            pr.append({'case_id':cid,'stategraph_status':'ready' if sg else 'INCOMPLETE','mem0_status':'ready' if mm else 'INCOMPLETE','stategraph_score':None,'mem0_score':None,'winner':None})
            if ds!='stale' and ds!='memora':
                refs=ref_map(ds)
                for tag,row in [('stategraph',sg),('mem0',mm)]:
                    if row: pr[-1][tag+'_score']=max(f1(row.get('answer',''),g) for g in refs.get(cid,['']))
            if sg and mm and pr[-1]['stategraph_score'] is not None:
                    a,b=pr[-1]['stategraph_score'],pr[-1]['mem0_score']; pr[-1]['winner']='StateGraph' if a>b else 'Mem0' if b>a else 'tie'
            elif ds=='memora' and sg and mm:
                a=all_metrics[ds]['StateGraph'].get('case_rubric_scores',{}).get(cid); b=all_metrics[ds]['Mem0'].get('case_rubric_scores',{}).get(cid)
                pr[-1]['stategraph_score']=a; pr[-1]['mem0_score']=b
                if a is not None and b is not None: pr[-1]['winner']='StateGraph' if a>b else 'Mem0' if b>a else 'tie'
        paired[ds]=pr
    provider = __import__('os').environ.get('MATRIX_PROVIDER', 'DeepSeek')
    model = __import__('os').environ.get('MATRIX_MODEL', 'deepseek-chat')
    write={'provider':provider,'model':model,'gold_loaded_after_all_prediction_seals':True,'metrics':all_metrics,'paired':paired,'failures':failures}
    (OUT/'RESULTS.json').write_text(json.dumps(write,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (OUT/'PAIRED_COMPARISON.json').write_text(json.dumps(paired,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'FAILURE_ATTRIBUTION.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[f'# Minimal StateGraph vs Mem0 diagnostic ({provider}/{model})','', '| Dataset | N | StateGraph | Mem0 | SG Wins | Mem0 Wins | Ties | Status |','|---|---:|---:|---:|---:|---:|---:|---|']
    for ds in datasets:
        n=len(paired[ds]); a=all_metrics[ds]['StateGraph']; b=all_metrics[ds]['Mem0']; score=lambda x: x.get('token_f1') if ds!='memora' else x.get('primary_rubric')
        c=paired.get(ds,{}); wins=sum(x.get('winner')=='StateGraph' for x in c); losses=sum(x.get('winner')=='Mem0' for x in c); ties=sum(x.get('winner')=='tie' for x in c)
        lines.append(f"| {ds} | {n} | {score(a) if score(a) is not None else 'N/A'} ({a['completed_cases']}/{a['planned_cases']}) | {score(b) if score(b) is not None else 'N/A'} ({b['completed_cases']}/{b['planned_cases']}) | {wins} | {losses} | {ties} | {a['status']}/{b['status']} |")
    lines += ['', f'All scores in this report are diagnostic; INCOMPLETE executions are not scored as zero. StateGraph and Mem0 used {provider}/{model} and the same answer budget. No StateGraph tuning was performed.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({ds:{m:{k:v for k,v in all_metrics[ds][m].items() if k in ('planned_cases','completed_cases','status','token_f1','normalized_em','primary_rubric','memory_presence','forgetting_absence')} for m in methods} for ds in datasets},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
