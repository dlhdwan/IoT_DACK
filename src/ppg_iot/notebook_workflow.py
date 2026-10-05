"""Auxiliary export of main notebook scientific cells."""
from __future__ import annotations
import copy
import json
import math
import os

import platform
import random
import subprocess
import sys
import time

import warnings
from pathlib import Path
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd

import scipy
from scipy import signal as sp_signal
import matplotlib
if "get_ipython" in globals():
    get_ipython().run_line_magic("matplotlib", "inline")
else: matplotlib.use("Agg")
import matplotlib.pyplot as plt


import torch
from torch import nn
from torch.ao.quantization import QuantStub, DeQuantStub, fuse_modules
from torch.ao.quantization import get_default_qconfig, prepare, convert
from ppg_iot.config import ROOT, SUBJECTS, SPLITS, load_config, project_path

from ppg_iot.provenance import RunRecorder, hash_object, sha256_file, write_json, now_utc, code_snapshot, fingerprint
from ppg_iot.data.loader import load_subject, dataset_root
from ppg_iot.data.alignment import validate_alignment
from ppg_iot.execution import create_notebook_workspace, require_unlocked
from ppg_iot.data.preprocessing import filter_continuous, normalize_ppg, normalize_acc
from ppg_iot.metrics import join_predictions, compute_metrics, train_median_fallback
from ppg_iot.metrics import paired_subset_metrics, summarize_seeds, ACTIVITIES
from ppg_iot.hardware import environment_names


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def read_frame(path):
    """Preserve reference values exactly across CSV round trips."""
    return pd.read_csv(path, float_precision="round_trip")

def cached_arrays(ctx,subject):
    return ctx["arrays"][subject]

def log_test_access(ctx, access_type, *, status="started", detail=None):
    path = ctx["root"] / "results/test_access.jsonl"
    record = {"at": now_utc(), "access_type": access_type, "status": status,
              "subjects": list(SPLITS["test"]), "detail": detail,
              "preprocess_hash": ctx["index"]["preprocess_hash"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

def notebook_download(url,path):
    import urllib.request
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".part")
    request=urllib.request.Request(url,headers={"User-Agent":"PPG-DaLiA-final-project/1.0"})
    total=0
    with urllib.request.urlopen(request,timeout=45) as response, temporary.open("wb") as output:
        while chunk:=response.read(1024*1024):
            output.write(chunk); total+=len(chunk)
            if total%(256*1024*1024)<1024*1024: print(f"Downloaded {total//1024**2} MiB",flush=True)
    if not total: raise RuntimeError(f"Empty download: {url}")
    temporary.replace(path)
    return path

def notebook_zip_members(archive):
    import re
    members={}
    for item in archive.infolist():
        match=re.fullmatch(r"(?:[^/]+/)*(S(?:[1-9]|1[0-5]))/\1\.pkl",item.filename)
        if match:
            if match[1] in members: raise ValueError(f"Duplicate source member: {match[1]}")
            members[match[1]]=item
    if set(members)!=set(SUBJECTS): raise ValueError("Source archive must contain all 15 subjects")
    return members

def notebook_extract_subjects(archive,data_root):
    import shutil
    sources={}
    for subject,member in notebook_zip_members(archive).items():
        target=data_root/subject/f"{subject}.pkl"; target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+".part")
        with archive.open(member) as source,temporary.open("wb") as output:
            shutil.copyfileobj(source,output,length=1024*1024)
        if temporary.stat().st_size!=member.file_size: raise ValueError(f"Truncated {subject}")
        temporary.replace(target)
        sources[subject]={**fingerprint(target),"archive_member":member.filename,
                          "zip_crc32":f"{member.CRC:08x}"}
        print(f"{subject}: extracted and ZIP CRC verified",flush=True)
    return sources

def notebook_write_source_report(root,readme,sources):
    result={"schema_version":1,"acquired_at":now_utc(),"dataset":"PPG-DaLiA",
        "dataset_page":"https://archive.ics.uci.edu/dataset/495/ppg%2Bdalia",
        "license":"Scientific/non-commercial use with attribution (author host)",
        "archive":{"url":"https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4JpP/download",
                   "selection":"15 synchronized pickles; member CRC verified; partial range source"},
        "readme":{**fingerprint(readme),"url":"https://archive.ics.uci.edu/ml/machine-learning-databases/00495/readme.pdf"},
        "subjects":sources,"scope":"official synchronized pickles; no evaluation performed"}
    report=root/"results/data_sources.json"; report.parent.mkdir(parents=True,exist_ok=True)
    write_json(report,result)
    return result

def notebook_local_source_report(root,data_root):
    sources={s:fingerprint(data_root/s/f"{s}.pkl") for s in SUBJECTS}
    readme=data_root/"readme.pdf"
    if not readme.is_file():
        base="https://archive.ics.uci.edu/ml/machine-learning-databases/00495/"
        readme=notebook_download(base+"readme.pdf",readme)
    report={"schema_version":1,"acquired_at":now_utc(),"dataset":"PPG-DaLiA",
        "origin":"locally supplied synchronized pickles; archive origin/CRC not independently verified",
        "readme":fingerprint(readme),"subjects":sources,
        "scope":"Actual file hashes; schema/alignment checked by the loader"}
    write_json(root/"results/data_sources.json",report)
    return report


def notebook_acquire_if_missing(root):
    import zipfile
    from ppg_iot.data.remote_zip import HTTPRangeFile
    require_unlocked(root)
    data_root=dataset_root(); report=root/"results/data_sources.json"
    missing=[s for s in SUBJECTS if not (data_root/s/f"{s}.pkl").is_file()]
    if not missing:
        return read_json(report) if report.is_file() else notebook_local_source_report(root,data_root)
    base="https://archive.ics.uci.edu/ml/machine-learning-databases/00495/"
    readme=notebook_download(base+"readme.pdf",data_root/"readme.pdf")
    remote=HTTPRangeFile("https://uni-siegen.sciebo.de/s/pfHzlTepXkiJ4JpP/download")
    try:
        with zipfile.ZipFile(remote) as archive:
            sources=notebook_extract_subjects(archive,data_root)
    finally: remote.close()
    return notebook_write_source_report(root,readme,sources)


def notebook_load_raw_subjects(root):
    sources=notebook_acquire_if_missing(root)
    records={}
    for subject,source in sources["subjects"].items():
        if subject not in SUBJECTS: raise ValueError(f"Unexpected dataset subject: {subject}")
        record=load_subject(subject,dataset_root(),source)
        validate_alignment(record)
        records[subject]=record
        print(f"{subject}: source SHA and raw PPG/ACC/ECG/HR/activity alignment verified",flush=True)
    if set(records)!=set(SUBJECTS): raise ValueError("Need all 15 synchronized subjects")
    return sources,records

def notebook_source_tables(sources,records):
    inventory=[]; alignment=[]
    for subject in SUBJECTS:
        record=records[subject]
        duration=len(record.ppg64)/record.ppg_hz
        inventory.append({"subject_id":subject,"source_sha256":record.source_sha256,
            "ppg_samples":len(record.ppg64),"acc_samples":len(record.acc32),
            "ecg_samples":record.ecg_sample_count,"hr_windows":len(record.hr_labels),
            "activity_codes":list(map(int,np.unique(record.activity_labels))),
            "duration_seconds":duration,"finite_ppg":bool(np.isfinite(record.ppg64).all()),
            "finite_acc":bool(np.isfinite(record.acc32).all()),
            "finite_hr":bool(np.isfinite(record.hr_labels).all()),
            "nonfinite_ppg_samples":int((~np.isfinite(record.ppg64)).sum()),
            "nonfinite_acc_values":int((~np.isfinite(record.acc32)).sum()),
            "invalid_hr_labels":int((~np.isfinite(record.hr_labels)|(record.hr_labels<=0)).sum()),
            "schema":record.schema_details,
            "split":next(key for key,values in SPLITS.items() if subject in values)})
        alignment.append({"subject_id":subject,"passed":True,"ppg_hz":record.ppg_hz,
            "acc_hz":record.acc_hz,"ecg_hz":record.ecg_hz,"activity_hz":record.activity_hz,
            "hr_windows":len(record.hr_labels),"first_hr":float(record.hr_labels[0]),
            "middle_hr":float(record.hr_labels[len(record.hr_labels)//2]),
            "last_hr":float(record.hr_labels[-1]),"label_seconds":"[2*i, 2*i+8)"})
    return pd.DataFrame(inventory),{"schema_version":1,"passed":True,
        "scope":"15 original subjects; exact sampled-rate alignment; ECG-derived HR labels",
        "readme_sha256":sources["readme"]["sha256"],"subjects":alignment}

def notebook_write_source_checks(root,inventory,alignment):
    require_unlocked(root)
    inventory_path=root/"results/data_inventory.json"
    alignment_path=root/"results/alignment.json"
    write_json(inventory_path,{"schema_version":1,"passed":True,"gate":"G02",
        "subjects":inventory.to_dict("records"),"errors":[]})
    write_json(alignment_path,alignment)
    return {"inventory":inventory_path,"alignment":alignment_path,
            "inventory_sha256":sha256_file(inventory_path),
            "alignment_sha256":sha256_file(alignment_path)}


def notebook_record_user_decision(ctx):
    review=ctx["root"]/"results/benchmark_review.json"
    decision={"schema_version":1,"decision":"use_original_ecg_final32",
        "authorized_by":"user","user_message":"Chốt dùng ECG gốc và giao thức cuối kỳ để train",
        "reference_verified":False,"review_sha256":sha256_file(review),
        "scope":"Use original ECG-derived HR labels and the final-project 32 Hz protocol; retain D01 benchmark as unverified."}
    path=ctx["root"]/"results/benchmark_source_decision.json"
    write_json(path,decision)
    return decision

def filter_subject_ppg(record,config):
    from scipy.signal import butter,resample_poly,sosfiltfilt
    band=config["filter"]["band_hz"]
    sos=butter(config["filter"]["prototype_order"],band,btype="bandpass",
               fs=record.ppg_hz,output="sos")
    filtered=sosfiltfilt(sos,record.ppg64,padtype="odd",padlen=None)
    output=resample_poly(filtered,up=1,down=2,window=("kaiser",5.0),padtype="constant")
    if len(output)!=len(record.acc32) or not np.isfinite(output).all():
        raise ValueError(f"{record.subject_id}: invalid 64→32 Hz filtered signal")
    return output

def merge_acc_moments(state,values):
    count,mean,moment=state
    current_count=len(values)
    current_mean=values.mean(axis=0)
    current_moment=np.square(values-current_mean).sum(axis=0)
    delta=current_mean-mean
    total=count+current_count
    return (total,mean+delta*current_count/total,
            moment+current_moment+delta**2*count*current_count/total)

def fit_notebook_acc_statistics(records,epsilon=1e-8):
    state=(0,np.zeros(3,dtype=np.float64),np.zeros(3,dtype=np.float64))
    subjects=[]
    for subject in SPLITS["train"]:
        record=records[subject]
        values=record.acc32.astype(np.float64,copy=False)
        state=merge_acc_moments(state,values)
        subjects.append({"subject_id":subject,"samples":len(values),
                         "source_sha256":record.source_sha256})
    count,mean,moment=state
    std=np.sqrt(moment/count)
    stats={"schema_version":1,"fit_split":"train","domain":"continuous_aligned_record",
        "ddof":0,"epsilon":epsilon,"n_samples_per_axis":count,"mean":mean.tolist(),
        "std":std.tolist(),"denominator":np.maximum(std,epsilon).tolist(),"subjects":subjects}
    stats["sha256"]=hash_object(stats)
    return stats

def normalize_notebook_acc(values,stats):
    unhashed={key:value for key,value in stats.items() if key!="sha256"}
    if stats["sha256"]!=hash_object(unhashed) or stats["fit_split"]!="train":
        raise ValueError("ACC normalization must use frozen Train-only statistics")
    result=(values-np.asarray(stats["mean"]))/np.asarray(stats["denominator"])
    if not np.isfinite(result).all(): raise ValueError("Nonfinite normalized ACC")
    return result.astype(np.float32)

def normalize_notebook_ppg(window,epsilon=1e-8):
    if window.shape!=(256,) or not np.isfinite(window).all():
        raise ValueError("One finite 8 s PPG window at 32 Hz is required")
    scale=max(float(window.std(ddof=0)),epsilon)
    return ((window-window.mean())/scale).astype(np.float32)

def assign_window_activity(labels,start_s,end_s,fs=4,min_fraction=.8):
    begin,stop=int(np.floor(start_s*fs)),int(np.ceil(end_s*fs))
    indices=np.arange(begin,stop)
    durations=np.maximum(0,np.minimum((indices+1)/fs,end_s)-np.maximum(indices/fs,start_s))
    codes=np.unique(labels[begin:stop])
    totals=np.array([durations[labels[begin:stop]==code].sum() for code in codes])
    dominant=int(codes[np.argmax(totals)]); fraction=float(totals.max()/(end_s-start_s))
    if dominant not in range(9): status,activity="unknown",None
    elif dominant>0 and fraction>=min_fraction: status,activity="assigned",dominant
    else: status,activity="mixed_transition",None
    return {"activity_id":activity,"activity_fraction":fraction,
            "activity_status":status,"dominant_activity_code":dominant}

def notebook_build_manifest(record,filtered,preprocess_hash,config):
    split=next(name for name,ids in SPLITS.items() if record.subject_id in ids)
    rows=[]
    for index,reference in enumerate(record.hr_labels):
        start=2*index; end=start+8
        p0,p1=start*32,end*32
        if p1>len(filtered): raise ValueError("Incomplete final label window")
        activity=assign_window_activity(record.activity_labels,start,end,
                                         min_fraction=config["signal"]["activity_min_fraction"])
        reason="nonfinite_hr_reference" if not np.isfinite(reference) else (
               "nonpositive_hr_reference" if reference<=0 else "")
        rows.append({"window_id":f"{config['run']['protocol_version']}:{record.subject_id}:{index:06d}",
            "subject_id":record.subject_id,"split":split,"label_index":index,
            "start_s":start,"end_s":end,"ppg_start_idx":p0,"ppg_end_idx":p1,
            "acc_start_idx":p0,"acc_end_idx":p1,"ppg_native_start_idx":p0*2,"ppg_native_end_idx":p1*2,
            "hr_ref_bpm":float(reference),**activity,"eligible":not reason,
            "exclusion_reason":reason,"ppg_flat":bool(filtered[p0:p1].std(ddof=0)==0),
            "source_sha256":record.source_sha256,"preprocess_hash":preprocess_hash})
    return pd.DataFrame(rows)

def notebook_preprocess_records(root,records,config):
    stats=fit_notebook_acc_statistics(records,config["signal"]["normalization_epsilon"])
    filtered={subject:filter_subject_ppg(record,config) for subject,record in records.items()}
    source_hashes={subject:records[subject].source_sha256 for subject in SUBJECTS}
    recipe={"config_sha256":hash_object(config),"alignment":"ECG labels 8 s/2 s; PPG 64→32 Hz",
        "filter":"Butterworth order 2, 0.5-4 Hz, SOS zero-phase on full subject; polyphase 1/2",
        "window":"256 samples at 32 Hz; 8 seconds; 2-second step; PPG bounds use actual 32 Hz indices",
        "sources":source_hashes,"stats_sha256":stats["sha256"],
        "normalization":"PPG per-window population; ACC continuous Train-only population"}
    preprocess_hash=hash_object(recipe)
    manifest=pd.concat([notebook_build_manifest(records[s],filtered[s],preprocess_hash,config)
                        for s in SUBJECTS],ignore_index=True)
    if manifest.window_id.duplicated().any(): raise ValueError("Duplicate common manifest IDs")
    return stats,filtered,manifest,preprocess_hash,recipe

def notebook_write_cache(root,records,stats,filtered,manifest,preprocess_hash,recipe,config):
    out=root/"results/manifests"; out.mkdir(parents=True,exist_ok=True)
    cache=root/"data/processed"/preprocess_hash; cache.mkdir(parents=True,exist_ok=True)
    entries={}
    for subject in SUBJECTS:
        ppg_path=cache/f"{subject}_ppg32.npy"
        acc_path=cache/f"{subject}_acc32_normalized.npy"
        acc=normalize_notebook_acc(records[subject].acc32,stats)
        np.save(ppg_path,filtered[subject],allow_pickle=False)
        np.save(acc_path,acc,allow_pickle=False)
        entries[subject]={"source_sha256":records[subject].source_sha256,
            "ppg32":{"file":ppg_path.name,"sha256":sha256_file(ppg_path)},
            "acc32_normalized":{"file":acc_path.name,"sha256":sha256_file(acc_path)}}
    manifest_path=out/"windows.csv"; manifest.to_csv(manifest_path,index=False)
    stats_path=out/"acc_stats.json"; write_json(stats_path,stats)
    recipe_path=out/"preprocessing.json"; write_json(recipe_path,{"preprocess_hash":preprocess_hash,"recipe":recipe})
    return cache,out,entries,manifest_path,stats_path,recipe_path

def notebook_write_indices(root,records,cache,out,entries,manifest_path,stats_path,recipe_path,
                           preprocess_hash,inventory,alignment,manifest):
    index_path=cache/"processed_index.json"
    index={"schema_version":1,"passed":True,"preprocess_hash":preprocess_hash,"subjects":entries,
           "manifest_sha256":sha256_file(manifest_path),"acc_stats_sha256":sha256_file(stats_path)}
    write_json(index_path,index)
    pointer={"cache_relative_path":index_path.relative_to(root).as_posix(),
        "index_sha256":sha256_file(index_path),"preprocess_hash":preprocess_hash,
        "manifest_sha256":sha256_file(manifest_path)}
    write_json(out/"processed_index.json",pointer)
    write_json(out/"checks.json",{"gate":"G03","passed":True,"n_subjects":len(records),
        "n_candidates":len(manifest),
        "n_eligible":int(inventory.eligible_windows.sum()),
        "preprocess_hash":preprocess_hash,"manifest_sha256":sha256_file(manifest_path)})
    write_json(root/"results/data_inventory.json",{"schema_version":1,"passed":True,
        "gate":"G02","subjects":inventory.to_dict("records"),"errors":[]})
    write_json(root/"results/alignment.json",alignment)
    return index_path,pointer

def notebook_prepare_project(root,sources,records,inventory,alignment):
    require_unlocked(root)
    config=load_config(root/"configs/default.toml",root=root)
    stats,filtered,manifest,preprocess_hash,recipe=notebook_preprocess_records(root,records,config)
    inventory=inventory.assign(eligible_windows=[
        int(manifest.loc[manifest.subject_id==subject,"eligible"].sum()) for subject in SUBJECTS])
    cache,out,entries,manifest_path,stats_path,recipe_path=notebook_write_cache(
        root,records,stats,filtered,manifest,preprocess_hash,recipe,config)
    index_path,pointer=notebook_write_indices(root,records,cache,out,entries,manifest_path,
        stats_path,recipe_path,preprocess_hash,inventory,alignment,manifest)
    arrays={subject:{"ppg32":filtered[subject],
        "acc32_normalized":normalize_notebook_acc(records[subject].acc32,stats)} for subject in SUBJECTS}
    ctx={"root":root,"config":config,"manifest":manifest,"manifest_path":manifest_path,
        "index":read_json(index_path),"index_path":index_path,"cache":cache,"stats":stats,
        "arrays":arrays,"raw_records":records,"sources":sources,"inventory":inventory}
    return ctx,inventory

def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path

def _eda_subject_quality(record,frame,item):
    hr=record.hr_labels[np.isfinite(record.hr_labels)&(record.hr_labels>0)]
    return {"subject_id":record.subject_id,"split":frame.split.iloc[0],
        "duration_minutes":len(record.ppg64)/64/60,"n_hr_labels":len(record.hr_labels),
        "n_eligible":int(frame.eligible.sum()),"n_excluded":int((~frame.eligible).sum()),
        "nonfinite_ppg":item["nonfinite_ppg_samples"],
        "nonfinite_acc":item["nonfinite_acc_values"],"invalid_hr":item["invalid_hr_labels"],
        "hr_min":float(hr.min()),"hr_q05":float(np.quantile(hr,.05)),
        "hr_median":float(np.median(hr)),"hr_mean":float(hr.mean()),
        "hr_q95":float(np.quantile(hr,.95)),"hr_max":float(hr.max()),
        "source_sha256":record.source_sha256}

def _eda_activity_time(record,split):
    return [{"subject_id":record.subject_id,"split":split,"activity_code":code,
             "activity":ACTIVITIES.get(code,"transition"),
             "duration_minutes":float(np.sum(record.activity_labels==code)/4/60)}
            for code in range(9)]

def _eda_window_features(record,arrays,frame):
    rows=[]
    for row in frame.loc[frame.eligible].to_dict("records"):
        p0,p1=int(row["ppg_start_idx"]),int(row["ppg_end_idx"])
        ppg=arrays["ppg32"][p0:p1]; acc=record.acc32[p0:p1]
        power=abs(np.fft.rfft((ppg-ppg.mean())*np.hanning(256)))**2
        probability=power/max(float(power.sum()),1e-20)
        rows.append({"window_id":row["window_id"],"subject_id":record.subject_id,
            "split":row["split"],"hr_ref_bpm":row["hr_ref_bpm"],
            "activity_id":row["activity_id"],"activity_status":row["activity_status"],
            "ppg_std_filtered":float(ppg.std(ddof=0)),
            "motion_rms_source_units":float(np.sqrt(np.mean(np.sum((acc-acc.mean(0))**2,axis=1)))),
            "spectral_entropy":float(-np.sum(probability*np.log2(np.maximum(probability,1e-20))))})
    return rows

def _eda_waveform(ctx,record,arrays,frame,out):
    activity=1 if record.subject_id=="S1" else 7
    candidates=frame.loc[frame.eligible&(frame.activity_id==activity)]
    row=(candidates if len(candidates) else frame.loc[frame.eligible]).iloc[0]
    p0,p1=int(row.ppg_start_idx),int(row.ppg_end_idx); start=float(row.start_s)
    fig,axes=plt.subplots(3,1,figsize=(10,7),sharex=True)
    axes[0].plot(start+np.arange(512)/64,record.ppg64[2*p0:2*p1],color="#0072B2")
    axes[0].set_ylabel("Raw PPG")
    axes[1].plot(start+np.arange(256)/32,arrays["ppg32"][p0:p1],color="#D55E00")
    axes[1].set_ylabel("Filtered PPG")
    for axis,(style,color) in enumerate(zip(("-","--",":"),("#0072B2","#E69F00","#009E73"))):
        axes[2].plot(start+np.arange(256)/32,record.acc32[p0:p1,axis],style,color=color,label=f"ACC {axis}")
    axes[2].legend(); axes[2].set_xlabel("Time (s)"); axes[2].set_ylabel("ACC source units")
    fig.suptitle(f"{row.window_id} | {row.split} | ECG HR={row.hr_ref_bpm:.2f} BPM")
    return save_figure(fig,out/f"waveform_{record.subject_id}.png").relative_to(ctx["root"]).as_posix()

def _eda_collect_subjects(ctx,out):
    sources=ctx.get("sources") or read_json(ctx["root"]/"results/data_sources.json")
    inventory=ctx.get("inventory")
    if inventory is None: inventory=read_json(ctx["root"]/"results/data_inventory.json")["subjects"]
    if isinstance(inventory,pd.DataFrame): inventory=inventory.to_dict("records")
    quality,activities,descriptors,figures=[],[],[],[]
    for item in inventory:
        subject=item["subject_id"]
        record=ctx.get("raw_records",{}).get(subject)
        if record is None: record=load_subject(subject,dataset_root(),sources["subjects"][subject])
        arrays=cached_arrays(ctx,subject)
        frame=ctx["manifest"].loc[ctx["manifest"].subject_id==subject]
        quality.append(_eda_subject_quality(record,frame,item))
        activities.extend(_eda_activity_time(record,frame.split.iloc[0]))
        descriptors.extend(_eda_window_features(record,arrays,frame))
        if subject in ("S1","S11"): figures.append(_eda_waveform(ctx,record,arrays,frame,out))
        print(f"EDA {subject}: quality, activities, HR and motion",flush=True)
    tables={"subject_quality":pd.DataFrame(quality),"activity_duration":pd.DataFrame(activities),
            "window_descriptors":pd.DataFrame(descriptors)}
    return tables,figures

def _eda_duration_plot(ctx,tables,out):
    fig,axes=plt.subplots(1,2,figsize=(12,4)); q=tables["subject_quality"]
    split_colors={"train":"#0072B2","validation":"#E69F00","test":"#009E73"}
    axes[0].bar(q.subject_id,q.duration_minutes,color=[split_colors[s] for s in q.split],edgecolor="white")
    axes[0].tick_params(axis="x",rotation=45)
    axes[0].set_ylabel("Actual record duration (min)")
    groups=[ctx["manifest"].loc[ctx["manifest"].subject_id==s,"hr_ref_bpm"].dropna() for s in SUBJECTS]
    boxes=axes[1].boxplot(groups,tick_labels=list(SUBJECTS),showfliers=False,patch_artist=True)
    for i,box in enumerate(boxes["boxes"]): box.set_facecolor(split_colors["train" if i<10 else "validation" if i<12 else "test"]); box.set_alpha(.65)
    axes[1].tick_params(axis="x",rotation=45); axes[1].set_ylabel("ECG reference HR (BPM)")
    return save_figure(fig,out/"duration_hr_subjects.png").relative_to(ctx["root"]).as_posix()

def _eda_activity_plot(ctx,tables,out):
    matrix=tables["activity_duration"].pivot(index="subject_id",columns="activity",
                                           values="duration_minutes").reindex(SUBJECTS)
    fig,ax=plt.subplots(figsize=(11,5))
    image=ax.imshow(matrix.values,aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha="right")
    ax.set_yticks(range(15),SUBJECTS)
    fig.colorbar(image,ax=ax,label="Actual activity duration (min)")
    return save_figure(fig,out/"activity_duration_heatmap.png").relative_to(ctx["root"]).as_posix()

def _eda_hr_motion_plot(ctx,tables,out):
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    split_colors={"train":"#0072B2","validation":"#E69F00","test":"#009E73"}
    for split,style in zip(SPLITS,("-","--",":")):
        values=ctx["manifest"].loc[(ctx["manifest"].split==split)&ctx["manifest"].eligible,"hr_ref_bpm"]
        axes[0].hist(values,bins=np.arange(35,220,5),histtype="step",linewidth=1.8,
                     color=split_colors[split],linestyle=style,label=split,density=True)
    axes[0].set_xlabel("ECG HR (BPM)"); axes[0].set_ylabel("Density"); axes[0].legend()
    frame=tables["window_descriptors"]; sample=frame.iloc[::max(1,len(frame)//4000)]
    points=axes[1].scatter(sample.motion_rms_source_units,sample.hr_ref_bpm,s=7,c=sample.activity_id,cmap="turbo",alpha=.5)
    fig.colorbar(points,ax=axes[1],label="Activity ID")
    axes[1].set_xlabel("Motion RMS (source units)"); axes[1].set_ylabel("ECG HR (BPM)")
    return save_figure(fig,out/"hr_distribution_motion.png").relative_to(ctx["root"]).as_posix()

def _eda_save_tables(ctx,tables,out):
    evidence={}
    for name,frame in tables.items():
        path=out/f"{name}.csv"; frame.to_csv(path,index=False); evidence[name]=path
    counts=ctx["manifest"].groupby(["split","activity_status"],dropna=False).size().reset_index(name="n_windows")
    counts.to_csv(out/"eligibility_activity_counts.csv",index=False)
    evidence["counts"]=out/"eligibility_activity_counts.csv"
    train=tables["window_descriptors"].loc[tables["window_descriptors"].split=="train"]
    columns=["hr_ref_bpm","motion_rms_source_units","ppg_std_filtered","spectral_entropy"]
    train[columns].corr(method="spearman").to_csv(out/"train_only_spearman.csv")
    evidence["train_correlation"]=out/"train_only_spearman.csv"
    return evidence

def _eda_save_index(ctx,tables,figures,out):
    evidence=_eda_save_tables(ctx,tables,out)
    info={"passed":True,"scope":"descriptive EDA; Train-only correlations; no Test tuning",
          "n_subjects":15,"n_candidates":len(ctx["manifest"]),
          "n_eligible":int(ctx["manifest"].eligible.sum()),
          "tables":{k:p.relative_to(ctx["root"]).as_posix() for k,p in evidence.items()},
          "figures":figures,"notes":["Activity duration counted on original 4 Hz labels.",
              "Window counts overlap; they are not independent durations.",
              "Descriptive Test distributions cannot justify tuning."]}
    write_json(out/"index.json",info); evidence["index"]=out/"index.json"
    return info,evidence

def analysis_stage(ctx):
    out=ctx["root"]/"results/eda"
    run=RunRecorder(ctx["root"],out/"runs",ctx["config"],ecode="EDA",kind="final",
                    inputs={"manifest":ctx["manifest_path"]})
    log_test_access(ctx,"EDA_descriptive",detail={"run_id":run.record["run_id"],"selection":False})
    try:
        tables,figures=_eda_collect_subjects(ctx,out)
        for draw in (_eda_duration_plot,_eda_activity_plot,_eda_hr_motion_plot):
            figures.append(draw(ctx,tables,out))
        info,evidence=_eda_save_index(ctx,tables,figures,out)
        run.finish(evidence=evidence)
        log_test_access(ctx,"EDA_descriptive",status="complete")
        return {"summary":info,"tables":tables}
    except BaseException as error:
        run.finish(error=str(error))
        log_test_access(ctx,"EDA_descriptive",status="failed",detail=str(error))
        raise

def fft_prediction(window, fs=32):
    """Hann, 4096 bins, amplitude peak in 0.6–3.5 Hz; flat windows retained."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 1 or not np.isfinite(window).all() or len(window) < 2 or fs <= 0:
        raise ValueError("FFT requires a complete finite signal")
    spectrum = np.abs(np.fft.rfft((window-window.mean()) *
                                sp_signal.windows.hann(len(window), sym=True), n=4096))
    frequencies = np.fft.rfftfreq(4096, d=1/fs)
    indices = np.flatnonzero((frequencies >= .6) & (frequencies <= 3.5))
    selected = indices[np.argmax(spectrum[indices])]
    return {"hr_pred_bpm": float(60*frequencies[selected]), "prediction_valid": True,
            "prediction_reason": "low_signal" if spectrum[indices].max() == 0 else "ok"}

def ibi_prediction(window, fs=32):
    """No reference information enters peak detection or IBI estimation."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 1 or not np.isfinite(window).all() or len(window) < 2 or fs <= 0:
        raise ValueError("IBI requires a complete finite signal")
    peaks, _ = sp_signal.find_peaks(window, distance=round(.33*fs),
                                    prominence=.3*window.std(ddof=0))
    valid = len(peaks) >= 2
    return {"hr_pred_bpm": float(60 / np.diff(peaks).mean() * fs) if valid else np.nan,
            "prediction_valid": valid, "prediction_reason": "ok" if valid else "insufficient_peaks",
            "peak_count": int(len(peaks))}

def _baseline_subject(ctx,subject,fallback):
    arrays=cached_arrays(ctx,subject)
    frame=ctx["manifest"].loc[ctx["manifest"].eligible&(ctx["manifest"].subject_id==subject)]
    fft_rows,ibi_rows,full_rows=[],[],[]
    for meta in frame.to_dict("records"):
        p0,p1=int(meta["ppg_start_idx"]),int(meta["ppg_end_idx"])
        window=arrays["ppg32"][p0:p1]
        fft=fft_prediction(window); ibi=ibi_prediction(window)
        full=dict(ibi,ibi_valid=ibi["prediction_valid"],used_fallback=not ibi["prediction_valid"],
                  prediction_valid=True)
        if full["used_fallback"]:
            full["hr_pred_bpm"]=fallback["hr_bpm"]
            full["prediction_reason"]="train_hr_median_fallback"
        fft_rows.append(dict(meta,**fft,ecode="E0"))
        ibi_rows.append(dict(meta,**ibi,ecode="E1_pure"))
        full_rows.append(dict(meta,**full,ecode="E1"))
    print(f"{subject}: {len(frame)} common FFT/IBI windows",flush=True)
    return fft_rows,ibi_rows,full_rows

def _baseline_save_method(ctx,run,name,rows):
    predictions=pd.DataFrame(rows)
    joined=join_predictions(ctx["manifest"],predictions,allow_invalid=name=="E1_pure")
    path=run.directory/f"{name}_predictions.csv"; predictions.to_csv(path,index=False)
    metrics={"all_subject_diagnostic":compute_metrics(joined,expected_subjects=list(SUBJECTS)),
             "splits":{split:compute_metrics(joined.loc[joined.split==split],
                        expected_subjects=list(subjects)) for split,subjects in SPLITS.items()}}
    metric_path=run.directory/f"{name}_metrics.json"; write_json(metric_path,metrics)
    record={"predictions":path.relative_to(ctx["root"]).as_posix(),
            "metrics":metric_path.relative_to(ctx["root"]).as_posix(),
            "predictions_sha256":sha256_file(path)}
    return record,{f"{name}_predictions":path,f"{name}_metrics":metric_path}

def _baseline_save_index(ctx,run,rows,fallback):
    methods={}; evidence={}
    for name,values in zip(("E0","E1_pure","E1"),rows):
        methods[name],files=_baseline_save_method(ctx,run,name,values)
        evidence.update(files)
    index={"passed":True,"gate":"G04_real_data","protocol":ctx["config"]["baseline"],
           "preprocess_hash":ctx["index"]["preprocess_hash"],
           "manifest_sha256":sha256_file(ctx["manifest_path"]),
           "run_id":run.record["run_id"],"methods":methods,"fallback":fallback}
    path=ctx["root"]/"results/baselines/index.json"; write_json(path,index)
    write_json(run.directory/"fallback.json",fallback)
    evidence["index"]=path
    return index,evidence

def baseline_stage(ctx):
    run=RunRecorder(ctx["root"],ctx["root"]/"results/baselines/runs",ctx["config"],
                    ecode="E0-E1",kind="final",inputs={"manifest":ctx["manifest_path"]})
    log_test_access(ctx,"DSP_baseline_diagnostic",detail={"run_id":run.record["run_id"],"selection":False})
    try:
        fallback=train_median_fallback(ctx["manifest"]); combined=([],[],[])
        for subject in SUBJECTS:
            for result,target in zip(_baseline_subject(ctx,subject,fallback),combined):
                target.extend(result)
        index,evidence=_baseline_save_index(ctx,run,combined,fallback)
        run.finish(evidence=evidence)
        log_test_access(ctx,"DSP_baseline_diagnostic",status="complete")
        return index
    except BaseException as error:
        run.finish(error=str(error))
        log_test_access(ctx,"DSP_baseline_diagnostic",status="failed",detail=str(error))
        raise

def _reference_value(table,domain,group):
    if domain=="test":
        selected=table["splits"]["test"]
        if group=="all": return selected["overall"]["macro_subject"]["mae_bpm"]
        activity=next(row for row in selected["by_activity"] if row["activity"]==group)
        return activity["macro_subject"]["mae_bpm"]
    selected=table["all_subject_diagnostic"]
    if group=="all": rows=selected["by_subject"]
    else: rows=next(row for row in selected["by_activity"] if row["activity"]==group)["by_subject"]
    return next(row for row in rows if row["subject_id"]=="S1")["mae_bpm"]

def _reference_comparison(ctx):
    index=read_json(ctx["root"]/"results/baselines/index.json")
    fft=read_json(ctx["root"]/index["methods"]["E0"]["metrics"])
    ibi=read_json(ctx["root"]/index["methods"]["E1_pure"]["metrics"])
    references=[("S1","FFT","all",11.68),("S1","IBI_pure","all",9.66),
        ("S1","FFT","sitting",3.67),("S1","FFT","stairs",32.42),
        ("test","FFT","all",12.71),("test","FFT","sitting",1.25),
        ("test","FFT","stairs",27.80),("test","FFT","walking",23.59)]
    rows=[]
    for domain,method,group,expected in references:
        actual=_reference_value(fft if method=="FFT" else ibi,domain,group)
        rows.append({"domain":domain,"method":method,"group":group,"reference_bpm":expected,
            "measured_final32_bpm":actual,"relative_deviation":abs(actual-expected)/expected if actual is not None else None,
            "same_protocol_verified":False,"reference_gate_passed":False})
    return rows

def review_benchmark(ctx):
    rows=_reference_comparison(ctx)
    pd.DataFrame(rows).to_csv(ctx["root"]/"results/baselines/benchmark_comparison.csv",index=False)
    report={"gate":"G05","passed":False,"status":"source_protocol_unresolved",
        "reason":"Old D01 code uses FFT-derived HR labels; exact benchmark protocol/mask/aggregation is not verified.",
        "comparison_scope":"Observed final32 numbers only; not a certified D01 reproduction",
        "threshold":.15,"preprocess_hash":ctx["index"]["preprocess_hash"],"comparison":rows,
        "baseline_index_sha256":sha256_file(ctx["root"]/"results/baselines/index.json"),
        "decision_file":"results/benchmark_source_decision.json"}
    write_json(ctx["root"]/"results/benchmark_review.json",report)
    return report

def require_training_gate(ctx):
    """Accept a verified reproduction or an explicit, recorded source decision."""
    report = read_json(ctx["root"]/"results/benchmark_review.json")
    if report["preprocess_hash"] != ctx["index"]["preprocess_hash"]:
        raise ValueError("G05 refers to a different preprocessing protocol")
    if report.get("passed") and report.get("same_protocol_verified"):
        return {"type": "verified_D01", "review_sha256": sha256_file(ctx["root"]/"results/benchmark_review.json")}
    path = ctx["root"]/"results/benchmark_source_decision.json"
    if not path.exists():
        raise RuntimeError("G05 unresolved: obtain an explicit source/protocol decision before pilot or long training")
    decision = read_json(path)
    if decision.get("authorized_by") != "user" or decision.get("decision") != "use_original_ecg_final32":
        raise ValueError("No valid user source decision")
    if decision.get("review_sha256") != sha256_file(ctx["root"]/"results/benchmark_review.json"):
        raise ValueError("Source decision does not cover the current review")
    if not decision.get("user_message") or decision.get("reference_verified") is not False:
        raise ValueError("Decision must preserve the unverified reference status")
    return {"type": "explicit_source_decision", "decision_sha256": sha256_file(path),
            "reference_verified": False}

def build_network(input_channels=4):
    """Three dilated Conv blocks, symmetric padding, GAP and a BPM regression head."""
    class NotebookWindowNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            if input_channels not in (1,4): raise ValueError("Use PPG-only or PPG+ACC")
            self.quant=QuantStub()
            self.conv1=nn.Conv1d(input_channels,32,7,padding=3,bias=False)
            self.bn1=nn.BatchNorm1d(32); self.relu1=nn.ReLU(); self.pool1=nn.MaxPool1d(2,2)
            self.conv2=nn.Conv1d(32,64,5,dilation=2,padding=4,bias=False)
            self.bn2=nn.BatchNorm1d(64); self.relu2=nn.ReLU(); self.pool2=nn.MaxPool1d(2,2)
            self.conv3=nn.Conv1d(64,64,3,dilation=4,padding=4,bias=False)
            self.bn3=nn.BatchNorm1d(64); self.relu3=nn.ReLU(); self.gap=nn.AdaptiveAvgPool1d(1)
            self.fc1=nn.Linear(64,32); self.relu4=nn.ReLU(); self.fc2=nn.Linear(32,1)
            self.dequant=DeQuantStub()
        def forward(self,x):
            x=self.quant(x)
            x=self.pool1(self.relu1(self.bn1(self.conv1(x))))
            x=self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x=self.relu3(self.bn3(self.conv3(x)))
            x=self.gap(x).flatten(1)
            return self.dequant(self.fc2(self.relu4(self.fc1(x))))
        def fuse_for_inference(self):
            if self.training: raise RuntimeError("Fusion requires eval")
            fuse_modules(self,[["conv1","bn1","relu1"],["conv2","bn2","relu2"],
                         ["conv3","bn3","relu3"],["fc1","relu4"]],inplace=True)
    return NotebookWindowNetwork()

def describe_window_network(channels):
    model = build_network(channels).eval()
    rows, handles = [], []
    def observe(module, _inputs, output):
        mac = output.shape[-1]*module.weight.numel() if isinstance(module, nn.Conv1d) else (
              module.weight.numel() if isinstance(module, nn.Linear) else 0)
        rows.append({"layer": type(module).__name__, "tensor": tuple(output.shape), "Conv_Linear_MAC": mac})
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.MaxPool1d, nn.AdaptiveAvgPool1d, nn.Linear)):
            handles.append(module.register_forward_hook(observe))
    with torch.inference_mode(): model(torch.zeros(1, channels, 256))
    for handle in handles: handle.remove()
    rf, jump = 1, 1
    for k, d, stride in [(7,1,1), (2,1,2), (5,2,1), (2,1,2), (3,4,1)]:
        rf += (k-1)*d*jump; jump *= stride
    stats = {"channels": channels, "parameters": sum(p.numel() for p in model.parameters()),
             "Conv_Linear_MAC": sum(r["Conv_Linear_MAC"] for r in rows), "local_RF_samples": rf,
             "local_RF_seconds": rf/32, "GAP_window_seconds": 8}
    return stats, pd.DataFrame(rows)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    generator=torch.Generator(device="cpu"); generator.manual_seed(seed)
    return generator

def accelerator_info():
    info={"python":platform.python_version(),"torch":torch.__version__,"cuda_available":torch.cuda.is_available(),
          "torch_cuda_version":torch.version.cuda,"quantized_engines":list(torch.backends.quantized.supported_engines)}
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        info.update(gpu=torch.cuda.get_device_name(0),free_vram_bytes=free,total_vram_bytes=total)
    return info

def materialize_split(ctx, split, device="cpu"):
    """Vectorized population z-score, preserving exactly the frozen manifest rows."""
    frame=ctx["manifest"].loc[ctx["manifest"].eligible &
            ((ctx["manifest"].split==split) if split!="all" else True)].reset_index(drop=True)
    if not len(frame): raise ValueError("No eligible windows")
    features=np.empty((len(frame),4,256),dtype=np.float32)
    offsets=np.arange(256)
    for subject,group in frame.groupby("subject_id",sort=False):
        arrays=cached_arrays(ctx,subject)
        indices=group.ppg_start_idx.to_numpy(dtype=np.int64)[:,None]+offsets
        ppg=arrays["ppg32"][indices]
        denominator=np.maximum(ppg.std(axis=1,ddof=0,keepdims=True),ctx["config"]["signal"]["normalization_epsilon"])
        ppg=((ppg-ppg.mean(axis=1,keepdims=True))/denominator).astype(np.float32)
        positions=group.index.to_numpy()
        features[positions,0,:]=ppg
        features[positions,1:,:]=arrays["acc32_normalized"][indices].transpose(0,2,1)
    target=frame.hr_ref_bpm.to_numpy(dtype=np.float32)[:,None]
    if not np.isfinite(features).all() or not np.isfinite(target).all(): raise ValueError("Non-finite tensors")
    return torch.from_numpy(features).to(device),torch.from_numpy(target).to(device),frame

def predict_tensor(model, features, batch_size=256):
    model.eval(); values=[]
    with torch.inference_mode():
        for begin in range(0,len(features),batch_size):
            prediction=model(features[begin:begin+batch_size])
            if prediction.shape!=(min(batch_size,len(features)-begin),1):
                raise ValueError("Expected (B,1), including B=1")
            values.append(prediction[:,0].detach().cpu().numpy())
    result=np.concatenate(values).astype(np.float64)
    if not np.isfinite(result).all(): raise ValueError("Non-finite network output")
    return result

def prediction_frame(frame, values, ecode, seed):
    if len(frame)!=len(values): raise ValueError("Missing prediction rows")
    result=frame.copy()
    result["hr_pred_bpm"]=values; result["prediction_valid"]=True
    result["prediction_reason"]="ok"; result["ecode"]=ecode; result["seed"]=seed
    return join_predictions(frame,result)

def train_epoch(model,optimizer,features,targets,generator,batch_size=64):
    model.train(); total=0.; n=0
    permutation=torch.randperm(len(features),generator=generator)
    for begin in range(0,len(features),batch_size):
        index=permutation[begin:begin+batch_size].to(features.device)
        x,y=features[index],targets[index]
        optimizer.zero_grad(set_to_none=True)
        prediction=model(x)
        if prediction.shape!=y.shape or y.ndim!=2 or y.shape[1]!=1:
            raise ValueError("Target broadcasting is forbidden")
        loss=nn.functional.l1_loss(prediction,y)
        if not torch.isfinite(loss): raise ValueError("Non-finite training loss")
        loss.backward(); optimizer.step()
        total+=float(loss.detach())*len(index); n+=len(index)
    return total/n

def capture_rng(generator):
    numpy_state=np.random.get_state()
    return {"python":random.getstate(),"numpy":numpy_state,
            "torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "permutation":generator.get_state()}

def restore_rng(state,generator):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state["cuda"])
    generator.set_state(state["permutation"])

def save_inference_model(model,path,*,input_channels,quantized,backend="onednn"):
    """Same model-only state_dict envelope for FP32 and INT8, no optimizer state."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    packet={"format":"ppg-window-state-v1","input_channels":input_channels,
            "quantized":bool(quantized),"backend":backend,"state_dict":model.cpu().state_dict()}
    torch.save(packet,path)
    return {"path":path,"sha256":sha256_file(path),"bytes":path.stat().st_size,"format":packet["format"]}

def load_inference_model(path,expected_sha256=None):
    path=Path(path)
    if expected_sha256 is not None and sha256_file(path)!=expected_sha256:
        raise ValueError("Artifact checksum mismatch")
    packet=torch.load(path,map_location="cpu",weights_only=True)
    if packet.get("format")!="ppg-window-state-v1": raise ValueError("Unknown artifact format")
    model=build_network(packet["input_channels"]).eval()
    if packet["quantized"]:
        backend=packet["backend"]
        if backend not in torch.backends.quantized.supported_engines: raise ValueError("Backend unavailable")
        torch.backends.quantized.engine=backend; model.fuse_for_inference()
        model.qconfig=get_default_qconfig(backend)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prepare(model,inplace=True); convert(model,inplace=True)
        # This reconstructs operator topology only; all calibrated qparams are loaded below.
    model.load_state_dict(packet["state_dict"],strict=True)
    return model.eval()

def accelerator_smoke(ctx,batches=10):
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; use Colab GPU")
    generator=seed_everything(42)
    x,y,frame=materialize_split(ctx,"train","cuda")
    model=build_network(4).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    torch.cuda.synchronize(); start=time.perf_counter()
    n=min(len(x),batches*64)
    loss=train_epoch(model,optimizer,x[:n],y[:n],generator,64)
    torch.cuda.synchronize()
    result={"passed":True,"scope":"CUDA feasibility on real Train only, not final training/MAE",
            "n_train_windows":n,"batches":math.ceil(n/64),"loss_bpm":loss,
            "wall_seconds":time.perf_counter()-start,"runtime":accelerator_info(),
            "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),
            "preprocess_hash":ctx["index"]["preprocess_hash"]}
    write_json(ctx["root"]/"results/accelerator_smoke.json",result)
    del model,optimizer,x,y; torch.cuda.empty_cache()
    return result

def _train_initialize(ctx,ecode,seed,data,kind,budget):
    channels=1 if ecode=="E2" else 4
    generator=seed_everything(seed)
    model=build_network(channels).to(data[0].device)
    policy=ctx["config"]["training"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=policy["learning_rate"],
                               weight_decay=policy["weight_decay"])
    return {"model":model,"optimizer":optimizer,"generator":generator,
            "ecode":ecode,"seed":seed,"kind":kind,"budget":budget,"channels":channels,
            "history":[],"best":float("inf"),"best_epoch":0,"stale":0,
            "start_epoch":1,"best_state":None,"config_sha256":hash_object(ctx["config"]),
            "preprocess_hash":ctx["index"]["preprocess_hash"]}

def _train_resume(ctx,state,parent,expected_hash):
    if not parent.resolve().is_relative_to(ctx["root"].resolve()): raise ValueError("Resume checkpoint outside workspace")
    if sha256_file(parent)!=expected_hash: raise ValueError("Resume checkpoint changed")
    saved=torch.load(parent,map_location="cpu",weights_only=False)
    runtime=accelerator_info()
    if any(saved.get("runtime",{}).get(k)!=runtime[k] for k in ("python","torch","gpu")):
        raise ValueError("Resume runtime changed; start a fresh execution")
    for key in ("ecode","seed","kind","budget","config_sha256","preprocess_hash"):
        if saved[key]!=state[key]: raise ValueError(f"Resume mismatch: {key}")
    state["model"].load_state_dict(saved["model"])
    state["optimizer"].load_state_dict(saved["optimizer"])
    restore_rng(saved["rng"],state["generator"])
    for key in ("best","best_epoch","stale","history","best_state"): state[key]=saved[key]
    state["start_epoch"]=saved["epoch"]+1

def _train_validation(ctx,state,val_data):
    x,_,frame=val_data
    values=predict_tensor(state["model"],x[:,:state["channels"],:])
    joined=prediction_frame(frame,values,state["ecode"],state["seed"])
    metrics=compute_metrics(joined,expected_subjects=list(SPLITS["validation"]))
    score=metrics["overall"]["macro_subject"]["mae_bpm"]
    if score is None or not math.isfinite(score): raise ValueError("Invalid Validation macro")
    return score,metrics["overall"]["pooled_window"]["mae_bpm"]

def _train_update_best(state,score,epoch):
    improved=score<state["best"]
    if improved:
        state["best"]=score
        state["best_epoch"]=epoch
        state["stale"]=0
        state["best_state"]={k:v.detach().cpu().clone() for k,v in state["model"].state_dict().items()}
    else:
        state["stale"]+=1
    return improved

def _train_save_resume(ctx,state,run,pointer,epoch):
    saved={key:state[key] for key in ("best","best_epoch","stale","best_state","history",
           "config_sha256","preprocess_hash","ecode","seed","kind","budget")}
    saved.update(model=state["model"].state_dict(),optimizer=state["optimizer"].state_dict(),
                 rng=capture_rng(state["generator"]),epoch=epoch,runtime=accelerator_info())
    checkpoint=run.directory/"last.pt"; temporary=run.directory/"last.pt.tmp"
    torch.save(saved,temporary); temporary.replace(checkpoint)
    pd.DataFrame(state["history"]).to_csv(run.directory/"history.csv",index=False)
    write_json(pointer,{"checkpoint":checkpoint.relative_to(ctx["root"]).as_posix(),
                        "sha256":sha256_file(checkpoint)})

def _train_fit(ctx,state,run,pointer,train_data,val_data):
    x,y,_=train_data; policy=ctx["config"]["training"]
    if state["stale"]>=policy["patience"]: return
    for epoch in range(state["start_epoch"],state["budget"]+1):
        start=time.perf_counter()
        loss=train_epoch(state["model"],state["optimizer"],x[:,:state["channels"],:],y,
                         state["generator"],policy["batch_size"])
        score,pooled=_train_validation(ctx,state,val_data)
        improved=_train_update_best(state,score,epoch)
        state["history"].append({"epoch":epoch,"train_l1_bpm":loss,
            "val_macro_mae_bpm":score,"val_pooled_mae_bpm":pooled,"selected":improved,
            "wall_seconds":time.perf_counter()-start})
        _train_save_resume(ctx,state,run,pointer,epoch)
        print(f"{state['kind']} {state['ecode']} seed={state['seed']} epoch={epoch}: "
              f"Train={loss:.3f}, Val={score:.3f}, best={state['best']:.3f}",flush=True)
        if state["stale"]>=policy["patience"]: break

def _train_export_selected(ctx,state,run,val_data):
    if state["best_state"] is None: raise ValueError("No selected checkpoint")
    model=state["model"].cpu()
    model.load_state_dict(state["best_state"])
    path=run.directory/"model_fp32.pt"
    saved=save_inference_model(model,path,input_channels=state["channels"],quantized=False)
    reloaded=load_inference_model(path,saved["sha256"])
    x,_,frame=val_data; x=x[:,:state["channels"],:].cpu()
    before=predict_tensor(model,x); after=predict_tensor(reloaded,x)
    np.testing.assert_allclose(before,after,atol=0,rtol=0)
    predictions=prediction_frame(frame,after,state["ecode"],state["seed"])
    predictions.to_csv(run.directory/"validation_predictions.csv",index=False)
    metrics=compute_metrics(predictions,expected_subjects=list(SPLITS["validation"]))
    write_json(run.directory/"validation_metrics.json",metrics)
    return saved,metrics

def _train_finish(ctx,state,run,saved,metrics,decision):
    result={key:state[key] for key in ("ecode","seed","kind","best_epoch","preprocess_hash","config_sha256")}
    result.update(passed=True,n_epochs=len(state["history"]),budget_epochs=state["budget"],
        selection="validation_macro_subject_mae",decision=decision,
        val_macro_mae_bpm=metrics["overall"]["macro_subject"]["mae_bpm"],
        artifact=saved["path"].relative_to(ctx["root"]).as_posix(),
        artifact_sha256=saved["sha256"],artifact_bytes=saved["bytes"],
        history=(run.directory/"history.csv").relative_to(ctx["root"]).as_posix(),
        validation_predictions=(run.directory/"validation_predictions.csv").relative_to(ctx["root"]).as_posix(),
        runtime=accelerator_info(),run_id=run.record["run_id"])
    write_json(run.directory/"checks.json",result)
    names=("history.csv","last.pt","model_fp32.pt","validation_predictions.csv","validation_metrics.json","checks.json")
    run.finish(evidence={name:run.directory/name for name in names})
    return result

def _train_load_registry(ctx,kind):
    path=ctx["root"]/("results/training_index.json" if kind=="final" else "results/pilot_index.json")
    index=read_json(path) if path.exists() else {"schema_version":1,"kind":kind,"runs":{}}
    for key,expected in (("preprocess_hash",ctx["index"]["preprocess_hash"]),("config_sha256",hash_object(ctx["config"]))):
        if index.get(key,expected)!=expected: raise ValueError("Training registry belongs to another protocol")
    return path,index

def training_record_matches(ctx,previous,experiment,seed,kind):
    if not previous or not previous.get("passed"): return False
    expected={"ecode":experiment,"seed":seed,"kind":kind,
        "config_sha256":hash_object(ctx["config"]),"preprocess_hash":ctx["index"]["preprocess_hash"]}
    if any(previous.get(k)!=v for k,v in expected.items()): return False
    current=accelerator_info(); runtime=previous.get("runtime",{})
    if any(runtime.get(k)!=current[k] for k in ("python","torch")): return False
    artifact=(ctx["root"]/previous.get("artifact","")).resolve()
    return (artifact.is_relative_to(ctx["root"].resolve()) and artifact.is_file()
        and artifact.stat().st_size==previous.get("artifact_bytes")
        and sha256_file(artifact)==previous.get("artifact_sha256"))


def training_registry_complete(ctx,index,kind):
    expected={f"{e}_{s}" for e in ("E2","E3") for s in (42,123,999)}
    if kind!="final" or set(index["runs"])!=expected: return False
    return all(training_record_matches(ctx,index["runs"][f"{e}_{s}"],e,s,kind)
        for e in ("E2","E3") for s in (42,123,999))


def _train_run_registry(ctx,kind,experiments,seeds,train_data,val_data,resume):
    require_unlocked(ctx["root"])
    path,index=_train_load_registry(ctx,kind)
    if not resume:
        index={"schema_version":1,"kind":kind,"runs":{},"passed":False,
               "preprocess_hash":ctx["index"]["preprocess_hash"],
               "config_sha256":hash_object(ctx["config"])}
        write_json(path,index)
    for experiment in experiments:
        for seed in seeds:
            key=f"{experiment}_{seed}"; previous=index["runs"].get(key)
            if resume and training_record_matches(ctx,previous,experiment,seed,kind):
                print(f"Reuse completed {key} ({kind})",flush=True)
                continue
            result=train_one(ctx,experiment,seed,train_data,val_data,kind=kind,resume=resume)
            index["runs"][key]=result
            index.update(preprocess_hash=ctx["index"]["preprocess_hash"],config_sha256=hash_object(ctx["config"]))
            index["passed"]=training_registry_complete(ctx,index,kind)
            write_json(path,index)
    index["passed"]=training_registry_complete(ctx,index,kind)
    write_json(path,index)
    return index

def train_one(ctx,ecode,seed,train_data,val_data,*,kind="final",pilot_epochs=2,resume=False):
    require_unlocked(ctx["root"])
    if ecode not in ("E2","E3") or seed not in (42,123,999): raise ValueError("Unknown experiment")
    decision=require_training_gate(ctx)
    budget=ctx["config"]["training"]["max_epochs"] if kind=="final" else pilot_epochs
    state=_train_initialize(ctx,ecode,seed,train_data,kind,budget)
    pointer=ctx["root"]/f"models/{ecode}/seed_{seed}/resume_{kind}.json"
    previous=read_json(pointer) if resume and pointer.exists() else None
    parent=ctx["root"]/previous["checkpoint"] if previous else None
    run=RunRecorder(ctx["root"],ctx["root"]/f"results/training/{kind}",ctx["config"],
        ecode=ecode,kind=kind,seed=seed,parent_checkpoint=parent,
        inputs={"manifest":ctx["manifest_path"],"stats":ctx["root"]/"results/manifests/acc_stats.json"})
    try:
        if previous: _train_resume(ctx,state,parent,previous["sha256"])
        _train_fit(ctx,state,run,pointer,train_data,val_data)
        saved,metrics=_train_export_selected(ctx,state,run,val_data)
        return _train_finish(ctx,state,run,saved,metrics,decision)
    except BaseException as error:
        run.finish(error=str(error))
        raise

def training_stage(ctx,kind="final",experiments=("E2","E3"),seeds=(42,123,999),resume=True):
    require_unlocked(ctx["root"])
    require_training_gate(ctx)
    if sys.version_info[:2]!=(3,11) or torch.__version__.split("+")[0]!="2.10.0":
        raise RuntimeError("Training requires Python3.11 / PyTorch2.10.0")
    device="cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(1)
    train_data=materialize_split(ctx,"train",device)
    val_data=materialize_split(ctx,"validation",device)
    result=_train_run_registry(ctx,kind,experiments,seeds,train_data,val_data,resume)
    del train_data,val_data
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result

def largest_remainder(counts,total):
    if total<0 or total>sum(counts.values()): raise ValueError("Insufficient strata capacity")
    keys=sorted(counts); capacity=sum(counts.values())
    exact={key:total*counts[key]/capacity for key in keys}
    quotas={key:int(math.floor(exact[key])) for key in keys}
    remaining=total-sum(quotas.values())
    for key in sorted(keys,key=lambda key:(-(exact[key]-quotas[key]),key))[:remaining]:
        quotas[key]+=1
    return quotas

def _cal_subject_quotas(train):
    capacities={subject:int(sum(train.subject_id==subject)) for subject in SPLITS["train"]}
    quotas={subject:min(100,capacity) for subject,capacity in capacities.items()}
    while sum(quotas.values())<1000:
        for subject in SPLITS["train"]:
            if quotas[subject]<capacities[subject] and sum(quotas.values())<1000:
                quotas[subject]+=1
    return quotas

def _cal_sample_strata(train,quotas):
    rng=np.random.default_rng(42); selected=[]; audit=[]
    train["stratum"]=[f"activity_{int(a)}" if status=="assigned" else status
                     for a,status in zip(train.activity_id,train.activity_status)]
    for subject in SPLITS["train"]:
        frame=train.loc[train.subject_id==subject]
        counts=frame.groupby("stratum").size().to_dict()
        for stratum,n in largest_remainder(counts,quotas[subject]).items():
            population=frame.loc[frame.stratum==stratum].sort_values("window_id")
            positions=rng.choice(len(population),size=n,replace=False)
            selected.append(population.iloc[positions])
            audit.append({"subject_id":subject,"stratum":stratum,"capacity":int(counts[stratum]),"selected":n})
    return pd.concat(selected).iloc[rng.permutation(1000)].reset_index(drop=True),audit

def calibration_manifest(ctx):
    require_unlocked(ctx["root"])
    train=ctx["manifest"].loc[ctx["manifest"].eligible&(ctx["manifest"].split=="train")].copy()
    if len(train)<1000: raise ValueError("Need 1000 unique real Train windows")
    quotas=_cal_subject_quotas(train); sample,audit=_cal_sample_strata(train,quotas)
    if sample.window_id.duplicated().any() or set(sample.split)!={"train"}: raise ValueError("Calibration leakage")
    path=ctx["root"]/"results/manifests/calibration.csv"
    if path.exists() and read_frame(path).window_id.tolist()!=sample.window_id.tolist():
        raise ValueError("Shared calibration IDs changed")
    sample.to_csv(path,index=False)
    recipe={"passed":True,"n_windows":1000,"seed":42,"replacement":False,"split":"train",
        "subject_quota":quotas,"strata":audit,"window_ids_sha256":hash_object(sample.window_id.tolist()),
        "calibration_sha256":sha256_file(path),"preprocess_hash":ctx["index"]["preprocess_hash"],
        "manifest_sha256":sha256_file(ctx["manifest_path"]),"batch_size":32}
    write_json(path.with_suffix(".json"),recipe)
    return sample,recipe

def audit_quantized_model(model,window):
    modules=[]; observed=[]; handles=[]
    for name,module in model.named_modules():
        if name in ("quant","conv1","conv2","conv3","gap","fc1","fc2","dequant"):
            def hook(_module,_inputs,output,label=name):
                observed.append({"name":label,"dtype":str(output.dtype),"quantized":output.is_quantized})
            handles.append(module.register_forward_hook(hook))
        weight=getattr(module,"weight",None)
        if callable(weight):
            tensor=weight()
            if isinstance(tensor,torch.Tensor):
                modules.append({"name":name,"class":type(module).__name__,"dtype":str(tensor.dtype),
                                "is_quantized":tensor.is_quantized,"qscheme":str(tensor.qscheme()) if tensor.is_quantized else None})
    with torch.inference_mode(): model(window)
    for handle in handles: handle.remove()
    quantized=[row for row in modules if row["is_quantized"] and row["dtype"]=="torch.qint8"]
    if len(quantized)!=5 or not any(row["name"]=="quant" and row["quantized"] for row in observed):
        raise ValueError("Expected 3 quantized convolutions, 2 linears and activation Quant")
    if observed[-1]["quantized"]: raise ValueError("Final BPM must be dequantized")
    return {"weights":modules,"activations":observed,
            "remaining_float":"Input/output, float bias and host DSP; GAP preserves quantized activation"}

def _ptq_fuse_and_check(original,validation):
    fused=copy.deepcopy(original).eval()
    fused.fuse_for_inference()
    before=predict_tensor(original,validation[:100])
    after=predict_tensor(fused,validation[:100])
    np.testing.assert_allclose(after,before,atol=1e-6,rtol=1e-5)
    return fused

def _ptq_calibrate(fused,windows,backend,batch_size):
    fused.qconfig=get_default_qconfig(backend)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        prepare(fused,inplace=True)
        with torch.inference_mode():
            for begin in range(0,len(windows),batch_size):
                fused(windows[begin:begin+batch_size])
        quantized=convert(fused,inplace=False)
    return quantized,sorted(set(str(item.message) for item in captured)),str(fused.qconfig)

def _ptq_export_validate(ctx,run,model,validation,frame,seed,backend):
    predicted=predict_tensor(model,validation)
    path=run.directory/"model_int8.pt"
    saved=save_inference_model(model,path,input_channels=4,quantized=True,backend=backend)
    reloaded=load_inference_model(path,saved["sha256"])
    roundtrip=predict_tensor(reloaded,validation)
    np.testing.assert_allclose(roundtrip,predicted,atol=0,rtol=0)
    joined=prediction_frame(frame,roundtrip,"E4",seed)
    joined.to_csv(run.directory/"validation_predictions.csv",index=False)
    metrics=compute_metrics(joined,expected_subjects=list(SPLITS["validation"]))
    write_json(run.directory/"validation_metrics.json",metrics)
    return saved,metrics

def _ptq_finish(ctx,run,seed,source,saved,metrics,calibration,audit,qconfig,warning_text):
    score=metrics["overall"]["macro_subject"]["mae_bpm"]
    result={"passed":True,"ecode":"E4","seed":seed,"config_sha256":hash_object(ctx["config"]),
        "artifact":saved["path"].relative_to(ctx["root"]).as_posix(),
        "artifact_sha256":saved["sha256"],"artifact_bytes":saved["bytes"],
        "source_artifact":source["artifact"],"source_sha256":source["artifact_sha256"],
        "calibration":calibration,"backend":ctx["config"]["quantization"]["backend"],
        "qconfig":qconfig,"warnings":warning_text,"audit":audit,"roundtrip_exact":True,
        "fusion_atol":1e-6,"fusion_rtol":1e-5,"val_macro_mae_bpm":score,
        "val_delta_vs_fp32_bpm":score-source["val_macro_mae_bpm"],
        "preprocess_hash":ctx["index"]["preprocess_hash"],"runtime":accelerator_info()}
    write_json(run.directory/"checks.json",result)
    names=("model_int8.pt","validation_predictions.csv","validation_metrics.json","checks.json")
    run.finish(evidence={name:run.directory/name for name in names})
    print(f"PTQ seed={seed}: Val={score:.3f}; bytes={saved['bytes']}",flush=True)
    return result

def _ptq_one(ctx,seed,source,calibration,cx,vx,frame):
    backend=ctx["config"]["quantization"]["backend"]
    run=RunRecorder(ctx["root"],ctx["root"]/"results/ptq/runs",ctx["config"],
        ecode="E4",kind="final",seed=seed,
        inputs={"calibration":ctx["root"]/"results/manifests/calibration.csv"},
        parent_checkpoint=ctx["root"]/source["artifact"])
    try:
        original=load_inference_model(ctx["root"]/source["artifact"],source["artifact_sha256"])
        fused=_ptq_fuse_and_check(original,vx)
        quantized,text,qconfig=_ptq_calibrate(fused,cx,backend,ctx["config"]["quantization"]["calibration_batch_size"])
        audit=audit_quantized_model(quantized,vx[:1])
        saved,metrics=_ptq_export_validate(ctx,run,quantized,vx,frame,seed,backend)
        return _ptq_finish(ctx,run,seed,source,saved,metrics,calibration,audit,qconfig,text)
    except BaseException as error:
        run.finish(error=str(error))
        raise

def ptq_record_matches(ctx,previous,source,calibration,seed):
    if not previous or not previous.get("passed"): return False
    expected={"ecode":"E4","seed":seed,"source_artifact":source["artifact"],
        "source_sha256":source["artifact_sha256"],"config_sha256":hash_object(ctx["config"]),
        "preprocess_hash":ctx["index"]["preprocess_hash"],"calibration":calibration,
        "backend":ctx["config"]["quantization"]["backend"]}
    if calibration.get("split")!="train" or any(previous.get(k)!=v for k,v in expected.items()): return False
    current=accelerator_info(); runtime=previous.get("runtime",{})
    if any(runtime.get(k)!=current[k] for k in ("python","torch")): return False
    parent=(ctx["root"]/source["artifact"]).resolve()
    artifact=(ctx["root"]/previous.get("artifact","")).resolve()
    return (parent.is_relative_to(ctx["root"].resolve()) and parent.is_file()
        and sha256_file(parent)==source["artifact_sha256"]
        and artifact.is_relative_to(ctx["root"].resolve()) and artifact.is_file()
        and artifact.stat().st_size==previous.get("artifact_bytes")
        and sha256_file(artifact)==previous.get("artifact_sha256"))


def ptq_registry_complete(ctx,index,training,calibration):
    if set(index["runs"])!={f"E4_{s}" for s in (42,123,999)}: return False
    if not all(f"E3_{s}" in training["runs"] for s in (42,123,999)): return False
    return all(ptq_record_matches(ctx,index["runs"][f"E4_{s}"],
        training["runs"][f"E3_{s}"],calibration,s) for s in (42,123,999))


def _ptq_run_registry(ctx,training,calibration,cx,vx,frame,seeds):
    require_unlocked(ctx["root"])
    path=ctx["root"]/"results/ptq_index.json"
    index=read_json(path) if path.exists() else {"schema_version":1,"runs":{}}
    for seed in seeds:
        key=f"E4_{seed}"; previous=index["runs"].get(key)
        source=training["runs"][f"E3_{seed}"]
        if ptq_record_matches(ctx,previous,source,calibration,seed):
            print(f"Reuse verified {key}",flush=True)
        else:
            index["runs"][key]=_ptq_one(ctx,seed,source,calibration,cx,vx,frame)
        index.update(passed=ptq_registry_complete(ctx,index,training,calibration),calibration=calibration)
        write_json(path,index)
    return index


def ptq_stage(ctx,seeds=(42,123,999)):
    require_unlocked(ctx["root"])
    training=read_json(ctx["root"]/"results/training_index.json")
    if not training.get("passed"): raise ValueError("Six final FP32 runs required")
    sample,calibration=calibration_manifest(ctx)
    tx,_,frame=materialize_split(ctx,"train","cpu")
    positions=pd.Series(np.arange(len(frame)),index=frame.window_id).loc[sample.window_id].to_numpy()
    cx=tx[torch.as_tensor(positions)]
    vx,_,validation=materialize_split(ctx,"validation","cpu")
    backend=ctx["config"]["quantization"]["backend"]
    if backend not in torch.backends.quantized.supported_engines: raise ValueError("Backend unavailable")
    torch.backends.quantized.engine=backend
    torch.set_num_threads(1)
    return _ptq_run_registry(ctx,training,calibration,cx,vx,validation,seeds)

def notebook_source_hash(path):
    notebook=read_json(path)
    return hash_object([{"cell_type":cell["cell_type"],"source":cell["source"]} for cell in notebook["cells"]])

def _lock_collect_paths(ctx,training,ptq,decision):
    paths=["configs/default.toml","uv.lock","results/data_sources.json","results/data_inventory.json",
        "results/alignment.json","results/manifests/windows.csv","results/manifests/acc_stats.json",
        "results/manifests/preprocessing.json","results/manifests/calibration.csv","results/manifests/calibration.json",
        "results/baselines/index.json","results/benchmark_review.json","results/training_index.json","results/ptq_index.json"]
    if decision["type"]=="explicit_source_decision": paths.append("results/benchmark_source_decision.json")
    for record in list(training["runs"].values())+list(ptq["runs"].values()):
        if not record["passed"] or sha256_file(ctx["root"]/record["artifact"])!=record["artifact_sha256"]:
            raise ValueError("Artifact checksum/gate mismatch")
        paths.append(record["artifact"])
    for record in read_json(ctx["root"]/"results/baselines/index.json")["methods"].values():
        paths.extend([record["predictions"],record["metrics"]])
    return sorted(set(paths))

def lock_protocol(ctx):
    if (ctx["root"]/"results/protocol_lock.json").exists(): return verify_lock(ctx)
    decision=require_training_gate(ctx)
    training=read_json(ctx["root"]/"results/training_index.json")
    ptq=read_json(ctx["root"]/"results/ptq_index.json")
    if not training.get("passed") or not ptq.get("passed"): raise ValueError("G07/G08 incomplete")
    paths=_lock_collect_paths(ctx,training,ptq,decision)
    content={"schema_version":1,"passed":True,"created_at":now_utc(),"decision":decision,
        "preprocess_hash":ctx["index"]["preprocess_hash"],
        "files":{p:sha256_file(ctx["root"]/p) for p in paths},"code":code_snapshot(ctx["root"])["files"],
        "notebook_source_sha256":notebook_source_hash(ctx["root"]/"DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb"),
        "selection":"Train/Val only; Val macro checkpoint; default PTQ",
        "metrics":{"primary":"subject_macro","large_error":"abs(error)>10","seed_std_ddof":1},
        "test_history":"results/test_access.jsonl; inherited Test"}
    content["protocol_sha256"]=hash_object(content)
    write_json(ctx["root"]/"results/protocol_lock.json",content)
    return content

def verify_lock(ctx):
    lock=read_json(ctx["root"]/"results/protocol_lock.json")
    if not lock["passed"] or lock["preprocess_hash"]!=ctx["index"]["preprocess_hash"]:
        raise ValueError("Protocol lock unavailable/stale")
    copy_lock=dict(lock); claimed=copy_lock.pop("protocol_sha256")
    if hash_object(copy_lock)!=claimed: raise ValueError("Lock content corrupted")
    for relative,checksum in lock["files"].items():
        if sha256_file(ctx["root"]/relative)!=checksum: raise ValueError(f"Locked file changed: {relative}")
    if code_snapshot(ctx["root"])["files"]!=lock["code"]: raise ValueError("Code changed after lock")
    if notebook_source_hash(ctx["root"]/"DinhLeHoangDanh_DoAn_PPG_DaLiA.ipynb")!=lock["notebook_source_sha256"]:
        raise ValueError("Notebook code/markdown changed after lock")
    return lock

def _eval_predictions(ctx,frame,x):
    predictions={}; baseline=read_json(ctx["root"]/"results/baselines/index.json")
    for name,record in baseline["methods"].items():
        all_pred=read_frame(ctx["root"]/record["predictions"])
        current=all_pred.loc[all_pred.window_id.isin(frame.window_id)].reset_index(drop=True)
        predictions[name]=join_predictions(frame,current,allow_invalid=name=="E1_pure")
    training=read_json(ctx["root"]/"results/training_index.json")
    ptq=read_json(ctx["root"]/"results/ptq_index.json")
    for key,record in {**training["runs"],**ptq["runs"]}.items():
        model=load_inference_model(ctx["root"]/record["artifact"],record["artifact_sha256"])
        channels=1 if key.startswith("E2") else 4
        values=predict_tensor(model,x[:,:channels,:])
        predictions[key]=prediction_frame(frame,values,record["ecode"],record["seed"])
        print(f"Evaluation {key}: {len(values)} windows",flush=True)
    return predictions

def _eval_save_methods(ctx,run,predictions,expected):
    methods={}; metrics={}; evidence={}
    for key,joined in predictions.items():
        path=run.directory/f"{key}_predictions.csv"
        joined.to_csv(path,index=False)
        metrics[key]=compute_metrics(joined,expected_subjects=expected)
        metric_path=run.directory/f"{key}_metrics.json"; write_json(metric_path,metrics[key])
        evidence[f"{key}_predictions"]=path; evidence[f"{key}_metrics"]=metric_path
        methods[key]={"predictions":path.relative_to(ctx["root"]).as_posix(),
            "metrics":metric_path.relative_to(ctx["root"]).as_posix(),"prediction_sha256":sha256_file(path)}
    return methods,metrics,evidence

def _eval_seed_summary(metrics):
    summary={}
    for experiment in ("E2","E3","E4"):
        summary[experiment]={name:summarize_seeds({
            seed:metrics[f"{experiment}_{seed}"]["overall"]["macro_subject"][name]
            for seed in (42,123,999)}) for name in ("mae_bpm","rmse_bpm","error_gt_10_percent")}
    for left,right in (("E3","E2"),("E4","E3")):
        summary[f"delta_{left}_minus_{right}"]=summarize_seeds({
            seed:metrics[f"{left}_{seed}"]["overall"]["macro_subject"]["mae_bpm"]-
                 metrics[f"{right}_{seed}"]["overall"]["macro_subject"]["mae_bpm"]
            for seed in (42,123,999)})
    return summary

def _eval_finish(ctx,run,out,split,lock,predictions,expected):
    methods,metrics,evidence=_eval_save_methods(ctx,run,predictions,expected)
    write_json(run.directory/"seed_summary.json",_eval_seed_summary(metrics))
    write_json(run.directory/"ibi_paired_subset.json",paired_subset_metrics(predictions))
    index={"passed":True,"split":split,"diagnostic":split=="all","methods":methods,
        "protocol_sha256":lock["protocol_sha256"],
        "summary":(run.directory/"seed_summary.json").relative_to(ctx["root"]).as_posix(),
        "paired_subset":(run.directory/"ibi_paired_subset.json").relative_to(ctx["root"]).as_posix()}
    write_json(out/"index.json",index); evidence["index"]=out/"index.json"
    evidence["summary"]=run.directory/"seed_summary.json"; evidence["paired"]=run.directory/"ibi_paired_subset.json"
    run.finish(evidence=evidence)
    return index

def evaluation_stage(ctx,split="test"):
    lock=verify_lock(ctx)
    if split not in ("test","all"): raise ValueError("Test or labeled all-subject diagnostic")
    out=ctx["root"]/("results/final_test" if split=="test" else "results/all_subject_diagnostics")
    run=RunRecorder(ctx["root"],out/"runs",ctx["config"],ecode="G09",kind="final",
                    inputs={"lock":ctx["root"]/"results/protocol_lock.json"})
    log_test_access(ctx,"locked_network_evaluation",detail={"run_id":run.record["run_id"],"split":split})
    try:
        x,_,frame=materialize_split(ctx,split,"cpu")
        expected=list(SPLITS["test"] if split=="test" else SUBJECTS)
        if set(frame.subject_id)!=set(expected): raise ValueError("Missing eligible subject")
        torch.set_num_threads(1)
        predictions=_eval_predictions(ctx,frame,x)
        index=_eval_finish(ctx,run,out,split,lock,predictions,expected)
        log_test_access(ctx,"locked_network_evaluation",status="complete")
        return index
    except BaseException as error:
        run.finish(error=str(error))
        log_test_access(ctx,"locked_network_evaluation",status="failed",detail=str(error))
        raise

def measure_latency(model,windows,warmup=100,iterations=1000):
    if warmup<100 or iterations<1000 or windows.ndim!=3 or windows.device.type!="cpu":
        raise ValueError("CPU benchmark requires >=100 warmups, >=1000 measurements")
    if torch.get_num_threads()!=1: raise ValueError("Benchmark requires one CPU thread")
    model.eval(); times=[]
    with torch.inference_mode():
        for index in range(warmup): model(windows[index%len(windows):index%len(windows)+1])
        for index in range(iterations):
            tensor=windows[index%len(windows):index%len(windows)+1]
            start=time.perf_counter_ns(); model(tensor); stop=time.perf_counter_ns()
            times.append((stop-start)/1e6)
    values=np.asarray(times)
    return values,{"n":iterations,"warmup":warmup,"batch_size":1,"cpu_threads":1,
                  "mean_ms":float(values.mean()),"median_ms":float(np.median(values)),
                  "p95_ms":float(np.quantile(values,.95)),"scope":"network including Quant/DeQuant; DSP/load/plot excluded"}

def _bench_models(ctx,windows):
    training=read_json(ctx["root"]/"results/training_index.json")
    ptq=read_json(ctx["root"]/"results/ptq_index.json")
    rows=[]; raw=[]; policy=ctx["config"]["evaluation"]
    for key,record in {**training["runs"],**ptq["runs"]}.items():
        path=ctx["root"]/record["artifact"]
        model=load_inference_model(path,record["artifact_sha256"])
        channels=1 if key.startswith("E2") else 4
        times,stats=measure_latency(model,windows[:,:channels,:],
            policy["warmup_iterations"],policy["measurement_iterations"])
        rows.append({"method":key,"seed":record["seed"],"artifact_bytes":path.stat().st_size,
                     "artifact_sha256":record["artifact_sha256"],**stats})
        raw.extend({"method":key,"iteration":i,"latency_ms":float(value)} for i,value in enumerate(times))
    return pd.DataFrame(rows),pd.DataFrame(raw)

def _bench_pairs(table):
    rows=[]
    for seed in (42,123,999):
        fp=table.loc[table.method==f"E3_{seed}"].iloc[0]
        q=table.loc[table.method==f"E4_{seed}"].iloc[0]
        rows.append({"seed":seed,"fp32_bytes":int(fp.artifact_bytes),"int8_bytes":int(q.artifact_bytes),
            "compression_ratio":float(fp.artifact_bytes/q.artifact_bytes),
            "speedup_median":float(fp.median_ms/q.median_ms),"speedup_mean":float(fp.mean_ms/q.mean_ms)})
    return pd.DataFrame(rows)

def _bench_dsp(ctx):
    sources=read_json(ctx["root"]/"results/data_sources.json")
    record=load_subject("S11",dataset_root(),sources["subjects"]["S11"])
    start=time.perf_counter()
    filtered=filter_continuous(record,ctx["config"])
    normalize_acc(record.acc32,ctx["stats"])
    continuous_ms=(time.perf_counter()-start)*1000
    frame=ctx["manifest"].loc[(ctx["manifest"].subject_id=="S11")&ctx["manifest"].eligible]
    start=time.perf_counter()
    for row in frame.itertuples(): normalize_ppg(filtered[row.ppg_start_idx:row.ppg_end_idx])
    normalization_ms=(time.perf_counter()-start)*1000
    return {"subject_id":"S11","n_samples_native_ppg":len(record.ppg64),"n_windows":len(frame),
        "continuous_filter_resample_acc_ms":continuous_ms,"all_window_ppg_zscore_ms":normalization_ms,
        "amortized_ms_per_window":(continuous_ms+normalization_ms)/len(frame),
        "scope":"offline record cost; amortized cost is not streaming latency"}

def _bench_targets(ctx,pairs):
    final=read_json(ctx["root"]/"results/final_test/index.json")
    summary=read_json(ctx["root"]/final["summary"])
    accuracy=summary["E3"]["mae_bpm"]["mean"]; delta=summary["delta_E4_minus_E3"]["mean"]
    rows=[{"target":"E3 Test macro MAE mean3seed <=6 BPM","value":accuracy,"limit":6,"achieved":accuracy<=6},
          {"target":"mean3seed E4-E3 MAE <=0.5 BPM","value":delta,"limit":.5,"achieved":delta<=.5}]
    for pair in pairs.to_dict("records"):
        seed=pair["seed"]; size=pair["fp32_bytes"]; ratio=pair["compression_ratio"]
        rows.append({"target":f"FP32 seed {seed} <150000 bytes","value":size,"limit":150000,"achieved":size<150000})
        rows.append({"target":f"Compression seed {seed} >=2.5","value":ratio,"limit":2.5,"achieved":ratio>=2.5})
    for seed,value in zip((42,123,999),summary["delta_E4_minus_E3"]["values"]):
        rows.append({"target":f"E4-E3 seed {seed} <=0.5 BPM","value":value,"limit":.5,"achieved":value<=.5})
    return pd.DataFrame(rows)

def _bench_save(ctx,run,lock,table,raw,pairs,targets,dsp,frame):
    names=environment_names()
    runtime={"benchmark_device":"cpu",**names,"platform":names["os"],"python":platform.python_version(),
        "torch":torch.__version__,"backend":ctx["config"]["quantization"]["backend"],
        "threads":torch.get_num_threads(),"protocol_sha256":lock["protocol_sha256"],
        "window_ids_sha256":hash_object(frame.window_id.iloc[:1000].tolist())}
    tables={"model_benchmark.csv":table,"latency_raw.csv":raw,
            "paired_benchmark.csv":pairs,"target_status.csv":targets}
    for name,data in tables.items(): data.to_csv(run.directory/name,index=False)
    write_json(run.directory/"runtime.json",runtime)
    write_json(run.directory/"dsp_timing.json",dsp)
    index={"passed":True,"tables":{name:(run.directory/name).relative_to(ctx["root"]).as_posix() for name in tables},
           "runtime":runtime,"dsp":dsp,"scope":"host CPU; no MCU/energy claims"}
    path=ctx["root"]/"results/benchmark/index.json"; write_json(path,index)
    evidence={name:run.directory/name for name in (*tables,"runtime.json","dsp_timing.json")}
    evidence["index"]=path; run.finish(evidence=evidence)
    return index

def benchmark_stage(ctx):
    lock=verify_lock(ctx); torch.set_num_threads(1)
    x,_,frame=materialize_split(ctx,"validation","cpu")
    run=RunRecorder(ctx["root"],ctx["root"]/"results/benchmark/runs",ctx["config"],
        ecode="G10",kind="final",inputs={"lock":ctx["root"]/"results/protocol_lock.json"})
    try:
        table,raw=_bench_models(ctx,x[:1000])
        pairs=_bench_pairs(table)
        targets=_bench_targets(ctx,pairs)
        dsp=_bench_dsp(ctx)
        return _bench_save(ctx,run,lock,table,raw,pairs,targets,dsp,frame)
    except BaseException as error:
        run.finish(error=str(error))
        raise

def plot_signal_case(ctx,row,predictions,path,title):
    arrays=cached_arrays(ctx,row["subject_id"]); p0,p1=int(row["ppg_start_idx"]),int(row["ppg_end_idx"])
    t=float(row["start_s"])+np.arange(256)/32
    fig,axes=plt.subplots(2,1,figsize=(10,5),sharex=True)
    axes[0].plot(t,arrays["ppg32"][p0:p1],color="#0072B2",linewidth=1.7,label="Filtered PPG"); axes[0].set_ylabel("Filtered PPG")
    for axis,(style,color) in enumerate(zip(("-","--",":"),("#0072B2","#E69F00","#009E73"))):
        axes[1].plot(t,arrays["acc32_normalized"][p0:p1,axis],style,color=color,label=f"ACC {axis}")
    axes[0].legend(); axes[1].legend(); axes[1].set_ylabel("Train z-score ACC"); axes[1].set_xlabel("Time (s)")
    text="; ".join(f"{key}={value:.2f}" for key,value in predictions.items())
    fig.suptitle(f"{title}\n{row['window_id']} | {row['split']} | ECG={row['hr_ref_bpm']:.2f} | {text}")
    return save_figure(fig,path)

def _report_tables(ctx,index,diagnostic,out):
    summary=read_json(ctx["root"]/index["summary"]); rows=[]
    for name,values in summary.items():
        if name.startswith("delta"): continue
        for metric,data in values.items(): rows.append({"experiment":name,"metric":metric,**data})
    pd.DataFrame(rows).drop(columns=["seeds","values"]).to_csv(out/"three_seed_summary.csv",index=False)
    subjects=[]; activities=[]
    for method,record in diagnostic["methods"].items():
        metrics=read_json(ctx["root"]/record["metrics"])
        subjects.extend(dict(method=method,**row) for row in metrics["by_subject"])
    pd.DataFrame(subjects).to_csv(out/"fifteen_subjects.csv",index=False)
    for method,record in index["methods"].items():
        metrics=read_json(ctx["root"]/record["metrics"])
        for row in metrics["by_activity"]:
            activities.append({"method":method,"activity":row["activity"],"n_windows":row["pooled_window"]["n_total"],
                "coverage":row["pooled_window"]["coverage"],"n_subjects":row["macro_subject"]["n_subjects_with_data"],
                "macro_mae_bpm":row["macro_subject"]["mae_bpm"],"pooled_mae_bpm":row["pooled_window"]["mae_bpm"]})
    pd.DataFrame(activities).to_csv(out/"activities_test.csv",index=False)

def _report_error_plot(ctx,primary,out):
    residual=primary.hr_pred_bpm-primary.hr_ref_bpm
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].hist(residual,bins=50,facecolor="#56B4E9",edgecolor="white")
    axes[0].set_xlabel("E3 seed42 prediction - ECG HR (BPM)"); axes[0].set_ylabel("Test windows")
    mean=(primary.hr_pred_bpm+primary.hr_ref_bpm)/2
    axes[1].scatter(mean,residual,s=5,c="#0072B2",alpha=.3)
    bias=float(residual.mean()); sd=float(residual.std(ddof=1))
    for value,style,color in ((bias,"-","#D55E00"),(bias-1.96*sd,"--","#009E73"),(bias+1.96*sd,"--","#009E73")):
        axes[1].axhline(value,color=color,linestyle=style)
    axes[1].set_xlabel("Mean prediction / ECG HR (BPM)"); axes[1].set_ylabel("Difference (BPM)")
    return save_figure(fig,out/"errors_bland_altman.png").relative_to(ctx["root"]).as_posix()

def _report_timeline(ctx,predictions,out):
    primary=predictions["E3_42"]
    error=abs(primary.hr_pred_bpm-primary.hr_ref_bpm)
    scores=primary.assign(error=error).groupby("subject_id").error.mean(); figures=[]
    for label,subject in (("easy",scores.idxmin()),("hard",scores.idxmax())):
        fig,ax=plt.subplots(figsize=(12,4))
        palette=plt.get_cmap("tab10").colors
        for j,((name,frame),style) in enumerate(zip(predictions.items(),("-","--",":","-.",(0,(3,1,1,1))))):
            selected=frame.loc[frame.subject_id==subject].sort_values("start_s")
            ax.plot(selected.start_s/60,selected.hr_pred_bpm,linestyle=style,color=palette[j%len(palette)],linewidth=1,label=name)
        reference=primary.loc[primary.subject_id==subject].sort_values("start_s")
        ax.plot(reference.start_s/60,reference.hr_ref_bpm,color="#222222",linewidth=1.8,label="ECG reference")
        ax.legend(ncol=3); ax.set_xlabel("Time (min)"); ax.set_ylabel("BPM")
        ax.set_title(f"Post-lock illustrative {label} Test subject {subject}; seed42; unsmoothed")
        figures.append(save_figure(fig,out/f"timeline_{label}.png").relative_to(ctx["root"]).as_posix())
    return figures

def _report_failure_rules(predictions):
    primary=predictions["E3_42"]; error=abs(primary.hr_pred_bpm-primary.hr_ref_bpm)
    e2error=abs(predictions["E2_42"].hr_pred_bpm-primary.hr_ref_bpm)
    qerror=abs(predictions["E4_42"].hr_pred_bpm-primary.hr_ref_bpm)
    rules={"smallest_E3_error":error.idxmin(),"largest_E3_error":error.idxmax(),
        "largest_ACC_minus_PPG_error":(error-e2error).idxmax(),
        "largest_INT8_minus_FP32_error":(qerror-error).idxmax()}
    return rules,error,error-e2error,qerror-error

def _report_failure_plots(ctx,predictions,out):
    rules,error,acc_delta,q_delta=_report_failure_rules(predictions); cases=[]; figures=[]
    primary=predictions["E3_42"]
    for label,window_id in rules.items():
        row=primary.loc[window_id].to_dict(); row["window_id"]=window_id
        values={name:float(frame.loc[window_id].hr_pred_bpm) for name,frame in predictions.items()}
        path=plot_signal_case(ctx,row,values,out/f"failure_{label}.png",label)
        figures.append(path.relative_to(ctx["root"]).as_posix())
        cases.append({"rule":label,"window_id":window_id,"subject_id":row["subject_id"],"split":row["split"],
            "hr_ref_bpm":row["hr_ref_bpm"],"predictions":values,"e3_error_bpm":float(error.loc[window_id]),
            "acc_minus_ppg_error_bpm":float(acc_delta.loc[window_id]),
            "int8_minus_fp32_error_bpm":float(q_delta.loc[window_id])})
    write_json(out/"failure_cases.json",cases)
    return rules,figures

def _report_history_plot(ctx,out):
    registry=read_json(ctx["root"]/"results/training_index.json")
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    palette={"E2":"#0072B2","E3":"#D55E00"}; seed_style={42:"-",123:"--",999:":"}
    for name,record in registry["runs"].items():
        history=read_frame(ctx["root"]/record["history"])
        style={42:"-",123:"--",999:":"}[record["seed"]]
        width=.8 if name.startswith("E2") else 1.5
        color=palette[record["ecode"]]
        axes[0].plot(history.epoch,history.train_l1_bpm,style,color=color,linewidth=width,label=name)
        axes[1].plot(history.epoch,history.val_macro_mae_bpm,style,color=color,linewidth=width,label=name)
    for ax,label in zip(axes,("Train L1 BPM","Validation macro MAE BPM")):
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
    axes[1].legend(fontsize=7)
    return save_figure(fig,out/"training_history.png").relative_to(ctx["root"]).as_posix()

def report_assets_stage(ctx):
    verify_lock(ctx); out=ctx["root"]/"results/report"; out.mkdir(parents=True,exist_ok=True)
    index=read_json(ctx["root"]/"results/final_test/index.json")
    diagnostic=read_json(ctx["root"]/"results/all_subject_diagnostics/index.json")
    _report_tables(ctx,index,diagnostic,out)
    names=("E0","E1","E2_42","E3_42","E4_42")
    predictions={name:read_frame(ctx["root"]/index["methods"][name]["predictions"]).set_index("window_id") for name in names}
    figures=[_report_error_plot(ctx,predictions["E3_42"],out)]
    figures.extend(_report_timeline(ctx,predictions,out))
    rules,cases=_report_failure_plots(ctx,predictions,out); figures.extend(cases)
    figures.append(_report_history_plot(ctx,out))
    result={"passed":True,"figures":figures,
        "tables":["results/report/three_seed_summary.csv","results/report/fifteen_subjects.csv","results/report/activities_test.csv"],
        "failure_cases":"results/report/failure_cases.json","selection_rules":rules,
        "limitations":["Correlated-window Bland–Altman is descriptive, not clinical validation.",
                      "Train/Val rows are not independent Test.","ACC improvement does not prove waveform denoising."]}
    write_json(out/"index.json",result)
    return result

def _demo_select_windows(frame,n_windows):
    indices=[]
    for activity in (1,7):
        positions=frame.index[frame.activity_id==activity].tolist()
        indices.extend(positions[:max(1,n_windows//2)])
    indices=list(dict.fromkeys(indices))
    if not indices: indices=list(range(min(n_windows,len(frame))))
    return indices[:n_windows]

def _demo_predict(ctx,selected,tensor,seed):
    training=read_json(ctx["root"]/"results/training_index.json")
    ptq=read_json(ctx["root"]/"results/ptq_index.json"); artifacts={}
    for experiment,registry in (("E3",training),("E4",ptq)):
        entry=registry["runs"][f"{experiment}_{seed}"]
        model=load_inference_model(ctx["root"]/entry["artifact"],entry["artifact_sha256"])
        selected[f"{experiment}_bpm"]=predict_tensor(model,tensor)
        selected[f"{experiment}_error_bpm"]=selected[f"{experiment}_bpm"]-selected.hr_ref_bpm
        artifacts[experiment]=entry["artifact_sha256"]
    return artifacts

def _demo_record_waveforms(selected,tensor):
    records=[]
    for index,row in enumerate(selected.to_dict("records")):
        records.append({"window_id":row["window_id"],"subject_id":row["subject_id"],"start_s":row["start_s"],
            "activity_id":None if pd.isna(row["activity_id"]) else int(row["activity_id"]),
            "hr_ref_bpm":row["hr_ref_bpm"],"E3_bpm":row["E3_bpm"],"E4_bpm":row["E4_bpm"],
            "ppg":tensor[index,0].tolist(),"acc":tensor[index,1:].tolist()})
    return records

def _demo_figures(ctx,selected,out,seed,speed):
    figures=[]
    for row in (selected.iloc[0],selected.iloc[-1]):
        values={name:float(row[f"{name}_bpm"]) for name in ("E3","E4")}
        title=f"Recorded Validation | CPU | seed={seed} | {speed}x | offline"
        path=plot_signal_case(ctx,row.to_dict(),values,out/f"replay_{row.label_index}.png",title)
        figures.append(path.relative_to(ctx["root"]).as_posix())
    return figures

def demo_stage(ctx,seed=42,n_windows=24,speed=1):
    x,_,frame=materialize_split(ctx,"validation","cpu")
    torch.set_num_threads(1)
    indices=_demo_select_windows(frame,n_windows)
    selected=frame.iloc[indices].copy(); tensor=x[indices]
    artifacts=_demo_predict(ctx,selected,tensor,seed)
    out=ctx["root"]/"results/demo"; out.mkdir(parents=True,exist_ok=True)
    selected.to_csv(out/"replay.csv",index=False)
    figures=_demo_figures(ctx,selected,out,seed,speed)
    write_json(out/"replay_windows.json",_demo_record_waveforms(selected,tensor))
    info={"passed":True,"split":"validation","seed":seed,"n_windows":len(selected),"speed":speed,
        "mode":"offline recorded replay","inference_platform":platform.platform(),
        "csv":"results/demo/replay.csv","data":"results/demo/replay_windows.json","figures":figures,
        "preprocess_hash":ctx["index"]["preprocess_hash"],"artifacts":artifacts}
    write_json(out/"index.json",info)
    return info

def project_status(ctx):
    files={"data":"results/data_inventory.json","preprocessing":"results/manifests/checks.json",
           "eda":"results/eda/index.json","baseline":"results/baselines/index.json",
           "training":"results/training_index.json","ptq":"results/ptq_index.json",
           "test":"results/final_test/index.json","benchmark":"results/benchmark/index.json",
           "figures":"results/report/index.json","demo":"results/demo/index.json"}
    rows=[]
    for phase,relative in files.items():
        path=ctx["root"]/relative; data=read_json(path) if path.exists() else {}
        rows.append({"phase":phase,"status":"verified" if data.get("passed") else "pending",
                     "evidence":relative if path.exists() else None})
    return pd.DataFrame(rows)

def build_final_result_table(final, root):
    rows = []
    for method in ("E0", "E1", "E2", "E3", "E4"):
        keys = [method] if method in ("E0", "E1") else [f"{method}_{s}" for s in (42,123,999)]
        metrics = [read_json(root/final["methods"][key]["metrics"])["overall"] for key in keys]
        counts = [m["pooled_window"]["n_total"] for m in metrics]
        if len(set(counts)) != 1: raise ValueError("Seed window populations differ")
        row = {"method": method, "n_seeds": len(keys) if method not in ("E0","E1") else None,
               "Test_windows_per_run": counts[0], "coverage": min(m["pooled_window"]["coverage"] for m in metrics)}
        for name in ("mae_bpm", "rmse_bpm", "error_gt_10_percent"):
            values = [m["macro_subject"][name] for m in metrics]
            row[name+"_mean"] = float(np.mean(values))
            row[name+"_sample_std"] = float(np.std(values, ddof=1)) if len(values)>1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_activity_result_table(final, root):
    rows = []
    names = ["E0", "E1"] + [f"{e}_{s}" for e in ("E2","E3","E4") for s in (42,123,999)]
    for name in names:
        method = name.split("_")[0]
        for activity in read_json(root/final["methods"][name]["metrics"])["by_activity"]:
            if activity["activity_id"] is None: continue
            rows.append({"method": method, "activity": activity["activity"],
                "mae_bpm": activity["macro_subject"]["mae_bpm"],
                "n_windows": activity["pooled_window"]["n_total"],
                "n_subjects": activity["macro_subject"]["n_subjects_with_data"]})
    grouped = pd.DataFrame(rows).groupby(["method","activity"], sort=False)
    return grouped.agg(mae_mean=("mae_bpm","mean"), mae_sample_std=("mae_bpm","std"),
        n_windows_per_run=("n_windows","min"), n_subjects=("n_subjects","min")).reset_index()


def describe_final_results(table, sensor, quantization, pairs):
        scores = table.set_index("method")
        e3, e4 = scores.loc["E3"], scores.loc["E4"]
        fft_reduction = 100*(1-e3.mae_bpm_mean/scores.loc["E0","mae_bpm_mean"])
        ibi_reduction = 100*(1-e3.mae_bpm_mean/scores.loc["E1","mae_bpm_mean"])
        better = int((sensor.E3_minus_E2_bpm < 0).sum())
        slower = pairs.loc[pairs.speedup_mean < 1,"seed"].astype(str).tolist()
        timing_note = "INT8 chậm hơn theo mean ở seed " + ", ".join(slower) if slower else "INT8 nhanh hơn theo mean ở cả ba seed trong phép đo này"
        return Markdown(f"""
**Mạng nhẹ:** E3 đạt **{e3.mae_bpm_mean:.3f} ± {e3.mae_bpm_sample_std:.3f} BPM**, E4 đạt **{e4.mae_bpm_mean:.3f} ± {e4.mae_bpm_sample_std:.3f} BPM** (Test macro, sample std giữa 3 seed).
Mục tiêu 6 BPM: **{'đạt' if e3.mae_bpm_mean <= 6 else 'không đạt'}** cho mean E3 trong giao thức này; coverage {e3.coverage:.1%}, {int(e3.Test_windows_per_run):,} cửa sổ/seed.
E3 RMSE macro **{e3.rmse_bpm_mean:.3f} BPM**, tỷ lệ lỗi >10 BPM macro trung bình **{e3.error_gt_10_percent_mean:.2f}%**. MAE là trung bình, không là chặn trên sai số từng cửa sổ.

**So DSP cùng giao thức:** E3 giảm MAE tổng **{fft_reduction:.2f}%** so FFT E0 và **{ibi_reduction:.2f}%** so IBI E1.
Đây là so baseline final32 mới, **không xác nhận tái lập benchmark Đ01 cũ**; từng hoạt động có thể bất lợi cho mạng.

**Tác động ACC:** mean E3−E2 = **{sensor.E3_minus_E2_bpm.mean():+.4f} BPM**, E3 tốt hơn E2 ở **{better}/3 seed**.
Lợi ích ACC chưa nhất quán giữa các seed; đối chứng sai số HR chưa xác lập cơ chế khử nhiễu PPG hoặc ý nghĩa thống kê trên quần thể.

**PTQ:** mean E4−E3 = **{quantization.E4_minus_E3_bpm.mean():+.4f} BPM**; mọi cặp seed {'đạt' if (quantization.E4_minus_E3_bpm <= .5).all() else 'không cùng đạt'} ngưỡng tăng 0,5 BPM.
FP32 **{int(pairs.fp32_bytes.iloc[0]):,} byte**, INT8 **{int(pairs.int8_bytes.iloc[0]):,} byte**, tỷ số **{pairs.compression_ratio.min():.3f}–{pairs.compression_ratio.max():.3f} lần**.
Median speedup **{pairs.speedup_median.min():.3f}–{pairs.speedup_median.max():.3f} lần**; **{timing_note}**. Kết quả mean và median cho thấy tăng tốc chưa đồng đều.

**Giá trị đạt được:** một mạng nhỏ hồi quy HR từ PPG/ACC và đánh giá PTQ8 bằng accuracy, byte và timing thực, bao gồm các trường hợp bất lợi.
Đây là benchmark host CPU với DSP ngoại tuyến; chưa là bằng chứng triển khai AI/latency/RAM/Flash/năng lượng trên MCU, LOSO hay hiệu quả chẩn đoán.
""")

