"""Seal explicit INCOMPLETE slices after bounded resource/provider runs."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/minimal_all_dataset_stategraph_vs_mem0_deepseek_v1'

def seal(dataset, method, ids, error):
    d=OUT/dataset; d.mkdir(parents=True, exist_ok=True)
    path=d/f'{method}_predictions.jsonl'
    rows=[{'dataset':dataset,'method':method,'case_id':i,'status':'INCOMPLETE','prediction_status':'INCOMPLETE','error_class':error,'gold_loaded_during_generation':False,'model_provider':'DeepSeek','model':'deepseek-chat'} for i in ids]
    text='\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n'; path.write_text(text,encoding='utf-8')
    (d/f'{method}_PREDICTION_SEAL.json').write_text(json.dumps({'status':'SEALED_INCOMPLETE','dataset':dataset,'method':method,'provider':'DeepSeek','model':'deepseek-chat','prediction_count':len(rows),'completed':0,'sha256':hashlib.sha256(text.encode()).hexdigest(),'gold_loaded_during_generation':False,'failure':error},ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    seal('longmemeval_v2','StateGraph',['0f970f01'],'INCOMPLETE_RESOURCE_BLOCKED_OFFICIAL_510_OBSERVATIONS')
    seal('longmemeval_v2','Mem0',['0f970f01'],'INCOMPLETE_RESOURCE_BLOCKED_OFFICIAL_510_OBSERVATIONS')
    seal('memora','StateGraph',['activity_todos_158','content_project_proposal_158_project_proposal_2','content_email_writeup_158_email_writeup_1'],'INCOMPLETE_PRODUCTION_EXTRACTION_STRUCTURED_OUTPUT')
    seal('mab_conflict','StateGraph',['row0-question0','row1-question0','row2-question0'],'INCOMPLETE_STRUCTURED_OUTPUT_DEPENDENCY_DISCOVERY')
    (OUT/'PREDICTIONS_ALL_SEALED.json').write_text(json.dumps({'status':'SEALED_AFTER_ALL_METHOD_ATTEMPTS','gold_loaded_during_generation':False},indent=2),encoding='utf-8')

if __name__=='__main__': main()
