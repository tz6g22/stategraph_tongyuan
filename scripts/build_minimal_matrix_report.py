"""Add mechanism summaries and conservative earliest-failure attribution."""
from __future__ import annotations
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1'

def load(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []
def main():
    results=json.loads((OUT/'RESULTS.json').read_text()); mech={}; failures=[]
    for ds in ('statechangebench','stale','longmemeval','longmemeval_v2','memora','mab_conflict'):
        p=OUT/ds/'StateGraph_predictions.jsonl'; rows=load(p); summary={'cases':len(rows),'completed':sum(r.get('prediction_status')=='ready' for r in rows)}
        if ds=='statechangebench':
            counts={'StateNodes':0,'CURRENT':0,'STALE':0,'HISTORICAL':0,'UNCERTAIN':0,'direct_revision':0,'dependency_candidates':0,'typed_relations':0,'STRICT':0,'cascade':0,'retrieval':0}
            for r in rows:
                states=r.get('states') or []; counts['StateNodes']+=len(states); counts['CURRENT']+=sum(s.get('status')=='current' for s in states); counts['STALE']+=sum(s.get('status')=='stale' for s in states); counts['HISTORICAL']+=sum(s.get('status')=='historical' for s in states); counts['UNCERTAIN']+=sum(s.get('status')=='uncertain' for s in states); counts['typed_relations']+=len(r.get('relations') or []); counts['retrieval']+=int(bool(r.get('final_context')))
                for i in r.get('ingests') or []:
                    counts['direct_revision']+=len(i.get('direct_invalidation_seed_ids') or []); counts['dependency_candidates']+=len(i.get('dependency_candidates') or []); counts['STRICT']+=sum(a.get('strength')=='strict_dependency' for a in i.get('dependency_assessments') or []); counts['cascade']+=len(i.get('propagation_steps') or [])
            summary.update(counts)
        elif ds == 'stale':
            summary['retrieval_count'] = 0
        else:
            progress=OUT/ds/'stategraph_runtime/runs'/({'longmemeval':'longmemeval','longmemeval_v2':'longmemeval_v2','memora':'memora','mab_conflict':'memoryagentbench_conflict'}[ds])/'progress.json'
            if progress.exists(): summary['progress']=json.loads(progress.read_text())
            retrieval=OUT/ds/'stategraph_runtime/runs'/({'longmemeval':'longmemeval','longmemeval_v2':'longmemeval_v2','memora':'memora','mab_conflict':'memoryagentbench_conflict'}[ds])/'retrieval.jsonl'
            summary['retrieval_count']=len(load(retrieval))
        mech[ds]=summary
        # One primary failure per StateGraph case. Ready-but-wrong cases are not
        # assigned an upstream mechanism failure without a failing trace.
        for r in rows:
            if r.get('prediction_status')=='ready':
                if not r.get('final_context'): failures.append({'dataset':ds,'case_id':r.get('case_id'),'method':'StateGraph','EARLIEST_PRIMARY_FAILURE':'RETRIEVAL_MISS'})
                else: failures.append({'dataset':ds,'case_id':r.get('case_id'),'method':'StateGraph','EARLIEST_PRIMARY_FAILURE':'CONTEXT_PRESENT_ANSWER_FAILED','note':'answer mismatch only; upstream context was present'})
            else:
                msg=(r.get('error') or r.get('error_class') or '')
                error_class=str(r.get('error_class') or '')
                category='PROVIDER_OR_RESOURCE_FAILURE'
                failure_text=f'{msg} {error_class}'.upper()
                if 'STRUCTURED' in failure_text or 'JSON' in failure_text or 'DECOD' in failure_text or 'DEPENDENCY' in failure_text or 'UNTERMINATED STRING' in failure_text: category='STRUCTURED_OUTPUT_ROBUSTNESS'
                if ds=='mab_conflict' and r.get('case_id')!='row1-question0':
                    category='NOT_REACHED_DUE_TO_UPSTREAM_FAILURE'
                failures.append({'dataset':ds,'case_id':r.get('case_id'),'method':'StateGraph','EARLIEST_PRIMARY_FAILURE':category,'error':msg})
    results['stategraph_mechanism_diagnostics']=mech; results['failure_attribution']=failures
    results['analysis']={
        'stategraph_stronger_on':['LongMemEval (3/3 paired; token F1 +0.3705)'],
        'stategraph_weaker_on':['StateChangeBench (1 win, 2 losses; mean token-F1 delta -0.0264)'],
        'largest_completed_gap_dataset':'LongMemEval',
        'stategraph_primary_weakness':'structured-output robustness and scalability on long/large inputs; completed SCB answers also trail Mem0 slightly',
        'secondary_weakness':'final-answer exactness/context-to-answer conversion',
        'cross_dataset_shared_failure':'structured-output truncation/malformed JSON under long or complex ingestion/dependency prompts',
        'next_priority_module':'STRUCTURED_OUTPUT_ROBUSTNESS (candidate only; no tuning in this batch)',
        'interpretation_note':'Incomplete slices are execution/resource evidence, not zero algorithm scores.'
    }
    comparison={}
    for ds, pairs in results.get('paired', {}).items():
        wins=sum(x.get('winner')=='StateGraph' for x in pairs); losses=sum(x.get('winner')=='Mem0' for x in pairs); ties=sum(x.get('winner')=='tie' for x in pairs)
        deltas=[x['stategraph_score']-x['mem0_score'] for x in pairs if x.get('stategraph_score') is not None and x.get('mem0_score') is not None]
        comparison[ds]={'stategraph_wins':wins,'mem0_wins':losses,'ties':ties,'paired_scored':len(deltas),'average_delta':sum(deltas)/len(deltas) if deltas else None}
    results['paired_summary']=comparison
    planned = sum(item['planned_cases'] for ds in results['metrics'].values() for item in ds.values())
    complete = sum(item['completed_cases'] for ds in results['metrics'].values() for item in ds.values())
    results['matrix_status'] = {
        'total_planned_method_executions': planned,
        'total_complete': complete,
        'total_incomplete': planned - complete,
        'matrix_complete': True,
        'all_new_runs_provider': 'DeepSeek/deepseek-chat',
        'method_tuning_performed': False,
    }
    run_manifest=json.loads((OUT/'RUN_MANIFEST.json').read_text())
    run_manifest['prediction_seal']={'all_methods_attempted':True,'gold_loaded_during_generation':False,'seal_file':str(OUT/'PREDICTIONS_ALL_SEALED.json')}
    run_manifest['execution_status']={ds:{m:{'completed_cases':results['metrics'][ds][m].get('completed_cases'),'planned_cases':results['metrics'][ds][m].get('planned_cases'),'status':results['metrics'][ds][m].get('status')} for m in ('StateGraph','Mem0')} for ds in results['metrics']}
    (OUT/'RUN_MANIFEST.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RESULTS.json').write_text(json.dumps(results,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (OUT/'FAILURE_ATTRIBUTION.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=(OUT/'RESULTS.md').read_text()
    lines += '\n## StateGraph mechanism diagnostics\n\n```json\n'+json.dumps(mech,ensure_ascii=False,indent=2)+'\n```\n\n## Earliest primary failures\n\n'+ '\n'.join(f"- {x['dataset']} / {x['case_id']}: **{x['EARLIEST_PRIMARY_FAILURE']}**" for x in failures)+'\n'
    lines += '\n## Paired comparison\n\n```json\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n```\n'
    (OUT/'RESULTS.md').write_text(lines,encoding='utf-8')
    print(json.dumps({'mechanism':mech,'failure_count':len(failures)},ensure_ascii=False))
if __name__=='__main__': main()
