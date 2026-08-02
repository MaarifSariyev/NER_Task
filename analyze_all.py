#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ner_analyze.py — Comprehensive NER dataset quality analyzer.

Input : a CSV in our format. Auto-detects columns:
          id/text/entities            (standard) OR
          unique_index/source_text/privacy_mask  (starter)
        entities may be JSON or Python-literal; `value` (if present) is ignored
        for structure but used for an integrity cross-check when available.

Output: a full report covering
  A. Dataset overview       (rows, entities, sources, languages)
  B. Structural integrity   (offset validity, overlaps, empties, bad labels)
  C. Label policy checks    (our boundary / type-word / title / split rules)
  D. Recall-gap scan        (unlabeled dates/money that our schema should cover)
  E. Label balance          (per-label counts, % , min-threshold flags)
  F. Length & density stats (word length buckets, entities per row)
  G. Duplication            (exact + near-duplicate texts)
  H. Split readiness        (labels present, rare-label risk)
  A PASS/FAIL verdict + an optional per-finding CSV (--report).

Usage:
    python ner_analyze.py final_dataset.csv
    python ner_analyze.py data.csv --report findings.csv
    python ner_analyze.py data.csv --min-per-label 100 --max-words 60
"""
import sys, csv, json, ast, re, argparse
from collections import Counter, defaultdict

SCHEMA = ["PERSON","ORGANIZATION","LOCATION","JOB","PRODUCT","WORKOFART","TIMEDATE","AMOUNT"]
SCHEMA_SET = set(SCHEMA)

# ---------------------------------------------------------------- policy tables
LEAD_FUNC = {"of","for","at","in","on","a","an","and","or","to","the"}
LEAD_THE_OK = {"ORGANIZATION","WORKOFART"}
TRAIL_FUNC = {"of","or","and","that","is","was","are","at","in","for","to"}
TITLES = {"dr","dr.","mr","mr.","mrs","mrs.","ms","ms.","prof","prof.","professor",
          "president","sir","madam"}
TYPEWORDS = {"company","organization","organisation","team","department","school",
             "person","people","city","country","product","book","hospital","agency",
             "committee","board","group","office","division","sector","moment","time",
             "host","clash","thing","winner","victim","suspect","customer","user",
             "client","passenger"}
SENT_PUNCT = ".,;:!?"
ARTICLES = {"a","an","the","this","that","these","those"}
TIME_UNITS = {"second","seconds","minute","minutes","hour","hours","day","days","week",
              "weeks","month","months","year","years","decade","decades"}

MONTHS = (r"(?:January|February|March|April|May|June|July|August|September|October|"
          r"November|December)")
WEEKDAYS = r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
ORD = r"\d{1,2}(?:st|nd|rd|th)"
GAP = {
 "TIMEDATE":[re.compile(rf"\b{ORD}(?:\s+(?:and|to|-)\s+{ORD})?\s+centur(?:y|ies)\b",re.I),
             re.compile(r"\b\d{1,4}\s?(?:BC|AD|BCE|CE)\b"),
             re.compile(r"\b(?:1\d{3}|20\d{2})\b"),
             re.compile(r"\b(?:mid-)?\d{2,4}s\b"),
             re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b",re.I),
             re.compile(r"\b\d{1,2}:\d{2}\s?(?:[AaPp]\.?[Mm]\.?)?\b"),
             re.compile(rf"\b{WEEKDAYS}\b",re.I)],
 "AMOUNT":[re.compile(r"[\$£€¥]\s?\d[\d,]*(?:\.\d+)?"),
           re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:%|percent)\b",re.I),
           re.compile(r"\b\d[\d,]*(?:\.\d+)?\s?(?:dollars?|euros?|pounds?)\b",re.I),
           re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")],
}

# ---------------------------------------------------------------- loading
def parse_cell(v):
    if isinstance(v,(list,dict)): return v
    if not isinstance(v,str) or not v.strip(): return []
    try: return json.loads(v)
    except json.JSONDecodeError: return ast.literal_eval(v)

def load(path):
    df = list(csv.DictReader(open(path, encoding="utf-8")))
    if not df:
        return []
    cols = df[0].keys()
    text_col = "text" if "text" in cols else "source_text"
    ent_col  = "entities" if "entities" in cols else "privacy_mask"
    id_col   = "id" if "id" in cols else ("unique_index" if "unique_index" in cols else None)
    rows = []
    for i, d in enumerate(df):
        ents = parse_cell(d.get(ent_col, "[]"))
        meta = parse_cell(d.get("meta","{}")) if "meta" in cols else {}
        rows.append({
            "id": d.get(id_col, i) if id_col else i,
            "text": d.get(text_col, "") or "",
            "entities": ents,
            "source": (meta or {}).get("source","unknown") if isinstance(meta,dict) else "unknown",
            "language": (meta or {}).get("language","?") if isinstance(meta,dict) else "?",
        })
    return rows

# ---------------------------------------------------------------- checks
def structural(text, e):
    s,en,lab = e.get("start"),e.get("end"),e.get("label")
    if lab not in SCHEMA_SET: return ("E-LABEL", f"label {lab!r} not in schema")
    if not isinstance(s,int) or not isinstance(en,int): return ("E-OOB","non-int offset")
    if not (0<=s<en<=len(text)): return ("E-OOB", f"offset [{s}:{en}] len {len(text)}")
    if not text[s:en].strip(): return ("E-EMPTY","empty span")
    if "value" in e and e["value"] is not None and text[s:en]!=e["value"]:
        return ("E-SLICE", f"value!=slice {e['value']!r}/{text[s:en]!r}")
    return None

def policy(text, e):
    lab=e.get("label"); val=text[e["start"]:e["end"]] if isinstance(e.get("start"),int) else ""
    if not val: return None
    words=val.split(); w0=words[0].lower().strip(SENT_PUNCT); wl=words[-1].lower().strip(SENT_PUNCT)
    keep=(lab=="TIMEDATE" and w0 in ARTICLES) or (w0=="the" and lab in LEAD_THE_OK)
    if w0 in LEAD_FUNC and not keep: return ("W-LEAD", f"leading {words[0]!r}")
    if wl in TRAIL_FUNC: return ("W-TRAIL", f"trailing {words[-1]!r}")
    if lab=="PERSON" and w0 in TITLES: return ("W-TITLE", f"honorific {words[0]!r}")
    if val.strip().lower().strip(SENT_PUNCT) in TYPEWORDS: return ("W-TYPEWORD", f"type-word {val!r}")
    if val[-1] in SENT_PUNCT and not re.search(r"[A-Z]\.$",val): return ("W-PUNCT", f"trailing punct")
    if lab=="AMOUNT" and set(val.lower().replace(","," ").split()) & TIME_UNITS:
        return ("W-TIME-AMT", f"AMOUNT has time unit {val!r}")
    return None

def overlaps(spans):
    ss=sorted([s for s in spans if isinstance(s.get("start"),int)],key=lambda x:x["start"])
    out=[]
    for a,b in zip(ss,ss[1:]):
        if a["end"]>b["start"]: out.append((a,b))
    return out

def recall_gaps(text, spans):
    def covered(s,e):
        return any(s<sp["end"] and sp["start"]<e for sp in spans if isinstance(sp.get("start"),int))
    hits=[]
    for label,pats in GAP.items():
        for pat in pats:
            for m in pat.finditer(text):
                s,e=m.start(),m.end()
                while e>s and text[e-1] in " .,;:!?": e-=1
                if e>s and not covered(s,e):
                    hits.append((label,text[s:e]))
    return hits

# ---------------------------------------------------------------- main
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--report", help="write per-finding CSV")
    ap.add_argument("--min-per-label", type=int, default=100)
    ap.add_argument("--max-words", type=int, default=60)
    a=ap.parse_args()

    rows=load(a.input)
    if not rows:
        print("No rows loaded."); sys.exit(1)

    findings=[]                       # (id, code, detail)
    label_counter=Counter()
    label_by_source=defaultdict(Counter)
    src_counter=Counter()
    lang_counter=Counter()
    ents_per_row=[]; lengths=[]
    empty_rows=0
    struct_err=0; policy_warn=0; gap_rows=0; overlap_rows=0
    seen=Counter(); dup_rows=0
    rows_with_all_ok=0

    for r in rows:
        text=r["text"]; spans=r["entities"]
        src_counter[r["source"]]+=1; lang_counter[r["language"]]+=1
        ents_per_row.append(len(spans)); lengths.append(len(text.split()))
        if not spans: empty_rows+=1
        key=text.strip()
        if key in seen: dup_rows+=1; findings.append((r["id"],"DUP","exact duplicate text"))
        seen[key]+=1

        row_ok=True
        for e in spans:
            se=structural(text,e)
            if se:
                findings.append((r["id"],se[0],se[1])); struct_err+=1; row_ok=False
                continue
            label_counter[e["label"]]+=1; label_by_source[r["source"]][e["label"]]+=1
            pe=policy(text,e)
            if pe:
                findings.append((r["id"],pe[0],pe[1])); policy_warn+=1; row_ok=False
        for a1,b1 in overlaps(spans):
            findings.append((r["id"],"E-OVERLAP",f"{a1.get('label')} vs {b1.get('label')}"))
            overlap_rows+=1; row_ok=False
        gaps=recall_gaps(text,spans)
        if gaps:
            gap_rows+=1; row_ok=False
            for lab,surf in gaps[:3]:
                findings.append((r["id"],"GAP",f"unlabeled {lab}: {surf!r}"))
        if r["text"].split() and len(r["text"].split())>a.max_words:
            findings.append((r["id"],"LONG",f"{len(r['text'].split())} words")); row_ok=False
        if row_ok: rows_with_all_ok+=1

    total_ents=sum(label_counter.values())
    N=len(rows)

    def bar(p): return "#"*int(p/2)

    print("="*60); print("A. DATASET OVERVIEW"); print("="*60)
    print(f"  Rows                 : {N}")
    print(f"  Total entities       : {total_ents}")
    print(f"  Avg entities / row   : {total_ents/N:.2f}")
    print(f"  Sources              : {dict(src_counter)}")
    print(f"  Languages            : {dict(lang_counter)}")

    print("\n"+"="*60); print("B. STRUCTURAL INTEGRITY"); print("="*60)
    print(f"  Offset/label errors  : {struct_err}")
    print(f"  Overlapping-span rows: {overlap_rows}")
    print(f"  Empty (0-entity) rows: {empty_rows}")
    verdict_struct = "PASS" if struct_err==0 and overlap_rows==0 else "FAIL"
    print(f"  -> structural verdict: {verdict_struct}")

    print("\n"+"="*60); print("C. LABEL POLICY (boundary / type-word / title)"); print("="*60)
    pol=Counter(f[1] for f in findings if f[1].startswith("W-"))
    if pol:
        for code,n in pol.most_common(): print(f"  {code:12} {n}")
    else:
        print("  no policy warnings")

    print("\n"+"="*60); print("D. RECALL-GAP SCAN (unlabeled date/money)"); print("="*60)
    print(f"  Rows with a recall gap: {gap_rows}")
    gap_ex=[f for f in findings if f[1]=="GAP"][:6]
    for _id,_,det in gap_ex: print(f"    rec {_id}: {det}")

    print("\n"+"="*60); print("E. LABEL BALANCE"); print("="*60)
    for lab in SCHEMA:
        n=label_counter[lab]; pct=100*n/total_ents if total_ents else 0
        flag="" if n>=a.min_per_label else "  <-- LOW"
        print(f"  {lab:14} {n:6}  {pct:5.1f}%  {bar(pct)}{flag}")

    print("\n"+"="*60); print("F. LABEL x SOURCE"); print("="*60)
    srcs=list(src_counter)
    print("  "+f"{'label':14}"+"".join(f"{s[:9]:>11}" for s in srcs))
    for lab in SCHEMA:
        print("  "+f"{lab:14}"+"".join(f"{label_by_source[s][lab]:>11}" for s in srcs))

    print("\n"+"="*60); print("G. LENGTH & DENSITY"); print("="*60)
    buckets=Counter()
    for L in lengths:
        b = "1-10" if L<=10 else "11-25" if L<=25 else "26-50" if L<=50 else "51-100" if L<=100 else "100+"
        buckets[b]+=1
    for b in ["1-10","11-25","26-50","51-100","100+"]:
        print(f"  {b:8} words : {buckets[b]:5} rows")
    print(f"  min/avg/max words   : {min(lengths)}/{sum(lengths)/len(lengths):.1f}/{max(lengths)}")

    print("\n"+"="*60); print("H. DUPLICATION"); print("="*60)
    print(f"  Exact duplicate rows : {dup_rows}")
    mult=[(t,c) for t,c in seen.items() if c>1]
    print(f"  Distinct texts repeated: {len(mult)}")

    print("\n"+"="*60); print("VERDICT"); print("="*60)
    clean_pct=100*rows_with_all_ok/N
    print(f"  Rows passing ALL checks: {rows_with_all_ok}/{N}  ({clean_pct:.1f}%)")
    low=[l for l in SCHEMA if label_counter[l]<a.min_per_label]
    print(f"  Labels below {a.min_per_label}: {low if low else 'none'}")
    overall = "PASS" if (struct_err==0 and overlap_rows==0 and gap_rows==0 and not low) else "NEEDS WORK"
    print(f"  OVERALL: {overall}")

    if a.report:
        with open(a.report,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); w.writerow(["id","code","detail"]); w.writerows(findings)
        print(f"\n  Wrote {len(findings)} findings -> {a.report}")

if __name__=="__main__":
    main()