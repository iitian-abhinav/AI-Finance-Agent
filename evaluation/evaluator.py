# evaluator.py
from __future__ import annotations
import re, statistics
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class EvaluationResult:
    case_id:str; category:str; ticker:str; repetition:int
    success:bool; status:str; error:Optional[str]
    latency_ms:float
    response_chars:int; response_words:int; response_paragraphs:int
    required_term_coverage_pct:float
    structure_score:float; citation_score:float; evidence_score:float
    missing_data_score:float; hallucination_penalty:float
    instruction_adherence_score:float
    forbidden_violations:int; forbidden_phrases_found:list[str]
    rating:Optional[str]; response:str

def _percentile(v,q):
    v=sorted(v)
    if not v:return 0.0
    if len(v)==1:return v[0]
    p=(len(v)-1)*q; lo=int(p); hi=min(lo+1,len(v)-1); w=p-lo
    return v[lo]*(1-w)+v[hi]*w

def detect_runtime_error(r:str)->Optional[str]:
    t=(r or "").lower()
    pats={"rate_limit":["rate limit reached","rate_limit_exceeded","too many requests","http/1.1 429"],
          "authentication":["invalid api key","authentication","unauthorized"],
          "context_limit":["request too large","context length","maximum context"],
          "provider_error":["error calling groq api","error in agent run","service unavailable"]}
    for k,ps in pats.items():
        if any(p in t for p in ps): return k
    return None

def classify_status(error,runtime):
    if runtime:return "INFRASTRUCTURE_FAILURE"
    if error:return "AGENT_FAILURE"
    return "SUCCESS"

def evaluate_response(case,response,latency_ms,repetition,error=None):
    text=response or ""; low=text.lower()
    runtime=detect_runtime_error(text)
    err=error or runtime
    status=classify_status(error,runtime)
    success=status=="SUCCESS"
    req=case.get("required_terms",[])
    cov=(sum(x.lower() in low for x in req)/max(len(req),1))*100
    secs=case.get("expected_sections",[])
    structure=(sum(s.lower() in low for s in secs)/max(len(secs),1))*20 if secs else 20
    cites=["https://","http://","sec.gov","source:","url:"]
    citation=(sum(c in low for c in cites)/len(cites))*20
    ev=["evidence","confirmed","interpretation","reported"]
    evidence=(sum(e in low for e in ev)/len(ev))*25
    miss=["unavailable","not retrieved","cannot determine","insufficient evidence"]
    missing=(sum(m in low for m in miss)/len(miss))*10
    viol=[p for p in case.get("forbidden_phrases",[]) if p.lower() in low]
    hall=min(len(viol)*10,20)
    score=round(max(0,min(100,cov*0.25+structure+citation+evidence+missing-hall)),2)
    m=re.findall(r"(?:rating|recommendation|final\s+rating)?\s*[:\-]?\s*(STRONG BUY|BUY|HOLD|SELL|AVOID|STRONG SELL)",text,re.I)
    rating=m[-1].upper() if m else None
    return EvaluationResult(case["case_id"],case["category"],case["ticker"],repetition,success,status,err,round(latency_ms,2),len(text),len(re.findall(r"\b\w+\b",text)),len([p for p in text.split("\n\n") if p.strip()]),round(cov,2),round(structure,2),round(citation,2),round(evidence,2),round(missing,2),hall,score,len(viol),viol,rating,text)

def summarize_results(results):
    if not results:return {}
    lat=[r.latency_ms for r in results]
    comp=[r for r in results if r.status=="SUCCESS"]
    infra=[r for r in results if r.status=="INFRASTRUCTURE_FAILURE"]
    agent=[r for r in results if r.status=="AGENT_FAILURE"]
    cats={}
    for c in sorted({r.category for r in results}):
        g=[r for r in results if r.category==c]
        gl=[r.latency_ms for r in g]
        cats[c]={"runs":len(g),"completed_runs":sum(r.status=="SUCCESS" for r in g),"infrastructure_failures":sum(r.status=="INFRASTRUCTURE_FAILURE" for r in g),"agent_failures":sum(r.status=="AGENT_FAILURE" for r in g),"mean_latency_ms":round(statistics.mean(gl),2),"p50_latency_ms":round(statistics.median(gl),2),"p95_latency_ms":round(_percentile(gl,.95),2),"mean_instruction_score":round(statistics.mean(r.instruction_adherence_score for r in g),2)}
    return {"total_runs":len(results),"completed_runs":len(comp),"infrastructure_failures":len(infra),"agent_failures":len(agent),"success_rate_pct":round(len(comp)/len(results)*100,2),"mean_latency_ms":round(statistics.mean(lat),2),"p50_latency_ms":round(statistics.median(lat),2),"p95_latency_ms":round(_percentile(lat,.95),2),"min_latency_ms":round(min(lat),2),"max_latency_ms":round(max(lat),2),"latency_stddev_ms":round(statistics.pstdev(lat),2),"mean_instruction_score":round(statistics.mean(r.instruction_adherence_score for r in results),2),"mean_structure_score":round(statistics.mean(r.structure_score for r in results),2),"mean_citation_score":round(statistics.mean(r.citation_score for r in results),2),"mean_evidence_score":round(statistics.mean(r.evidence_score for r in results),2),"mean_missing_data_score":round(statistics.mean(r.missing_data_score for r in results),2),"total_forbidden_violations":sum(r.forbidden_violations for r in results),"by_category":cats}

def result_to_dict(result):
    return asdict(result)
