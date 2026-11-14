"""Fetch the pinned public E5 snapshot; no API key or remote model code."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
FILES = ["1_Pooling/config.json", "README.md", "config.json", "model.safetensors",
         "modules.json", "sentence_bert_config.json", "sentencepiece.bpe.model",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]


def main():
    cfg=json.loads((ROOT/"configs/strategies.json").read_text(encoding="utf-8"))
    revision=cfg["revision"]
    if revision=="main": raise ValueError("Resolve and pin a model commit SHA before downloading")
    target=ROOT/".local/model-snapshot"; target.mkdir(parents=True,exist_ok=True)
    old_path=ROOT/"artifacts/model_download.json"
    old=json.loads(old_path.read_text(encoding="utf-8")) if old_path.exists() else {}
    old_files={r["file"]:r for r in old.get("files",[])} if old.get("revision")==revision else {}
    def fetch(name):
        p=target/name; p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and name in old_files and hashlib.sha256(p.read_bytes()).hexdigest()==old_files[name]["sha256"]:
            return old_files[name]
        url=f"https://huggingface.co/{cfg['model']}/resolve/{revision}/{name}"
        print("Downloading",name,flush=True)
        if name in ("model.safetensors", "tokenizer.json"):
            with requests.get(url,headers={"Range":"bytes=0-0"},timeout=(30,45)) as probe:
                probe.raise_for_status()
                if probe.status_code != 206: raise ValueError("Range request not supported; cannot safely resume")
                total=int(probe.headers["Content-Range"].split("/")[-1])
            block=1024*1024
            parts=ROOT/".local/model-download-parts"/revision/name;parts.mkdir(parents=True,exist_ok=True)
            def part(i):
                start=i*block;end=min(total,start+block)-1;size=end-start+1
                part_path=parts/f"{i:05}.part"
                if part_path.exists() and part_path.stat().st_size==size:return part_path
                for attempt in range(4):
                    try:
                        with requests.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=(30,60)) as r:
                            r.raise_for_status()
                            if r.status_code!=206 or r.headers.get("Content-Range")!=f"bytes {start}-{end}/{total}":raise ValueError("Range mismatch")
                            content=r.content
                            if len(content)!=size:raise ValueError("Truncated range")
                            part_path.write_bytes(content)
                        if i%40==0:print(name, "part",i+1,"/",(total+block-1)//block,flush=True)
                        return part_path
                    except (requests.RequestException,ValueError):
                        if attempt==3:raise
                        time.sleep(attempt+1)
            with ThreadPoolExecutor(max_workers=6) as pool: part_paths=list(pool.map(part,range((total+block-1)//block)))
            partial=p.with_name(p.name+".partial");sha=hashlib.sha256();size=0
            with partial.open("wb") as f:
                for part_path in part_paths:
                    data=part_path.read_bytes();f.write(data);sha.update(data);size+=len(data)
            if size!=total:raise ValueError("Reassembled size mismatch")
            partial.replace(p)
            print("Downloaded",name,size,"bytes (verified ranges)",flush=True)
            return {"file":name,"source":url,"bytes":size,"sha256":sha.hexdigest()}
        with requests.get(url,stream=True,timeout=(30,60)) as response:
            response.raise_for_status(); partial=p.with_name(p.name+".partial"); sha=hashlib.sha256(); size=0
            with partial.open("wb") as f:
                for chunk in response.iter_content(1024*1024):
                    if chunk: f.write(chunk);sha.update(chunk);size+=len(chunk)
            expected=response.headers.get("Content-Length")
            if expected and size!=int(expected): raise ValueError(f"Incomplete download {name}")
            partial.replace(p)
        print("Downloaded",name,size,"bytes",flush=True)
        return {"file":name,"source":url,"bytes":size,"sha256":sha.hexdigest()}
    with ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(fetch,FILES))
    old_path.parent.mkdir(parents=True,exist_ok=True)
    old_path.write_text(json.dumps({"downloaded_at":datetime.now(timezone.utc).isoformat(),"model":cfg["model"],
          "revision":revision,"local_path":".local/model-snapshot","files":rows},indent=2)+"\n",encoding="utf-8")
    print("Pinned snapshot ready.",flush=True)


if __name__=="__main__": main()
