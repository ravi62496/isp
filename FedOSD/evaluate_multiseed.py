import os
import json
import glob
import math
import re
import numpy as np
from collections import defaultdict

def analyze_multiseed(base_dir="UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"):
    json_files = glob.glob(os.path.join(base_dir, "*_unlearn.json"))
    retrain_files = glob.glob("UnlearningTask/**/FedAvg/*retrain*.json", recursive=True)
    json_files.extend(retrain_files)

    if not json_files:
        print(f"No unlearn JSON files found in {base_dir}")
        return

    runs = defaultdict(dict)

    for fpath in sorted(set(json_files)):
        fname = os.path.basename(fpath)
        
        seed_match = re.search(r"seed(\d+)", fname)
        seed = int(seed_match.group(1)) if seed_match else 1
        
        lam_match = re.search(r"lam([0-9]+(?:\.[0-9]+)?)", fname)
        eta_match = re.search(r"eta([0-9]+(?:\.[0-9]+)?)", fname)
        
        if "retrain" in fname.lower():
            config_key = "Retrain from Scratch (200 R)"
        elif "proj_one_sided" in fname and ("100000" in fname or (lam_match and float(lam_match.group(1)) >= 10000.0)):
            config_key = "One-Sided Hard (lam=Inf)"
        elif "wm_drop" in fname:
            config_key = "Adaptive Drop (lam=20, eta=0.4)"
        elif "wm_deficit" in fname:
            config_key = "Adaptive Deficit (lam=20, eta=0.4)"
        elif "wm_conflict" in fname:
            config_key = "Adaptive Conflict (lam=20, eta=0.4)"
        elif lam_match and eta_match:
            lam_val = float(lam_match.group(1))
            eta_val = float(eta_match.group(1))
            if lam_val >= 10000.0:
                config_key = "Original FedOSD (Hard)"
            elif eta_val == 0.0:
                config_key = f"Soft Uniform (lam={lam_val:g})"
            else:
                config_key = f"Soft (lam={lam_val:g}, eta={eta_val:g})"
        else:
            config_key = "Original FedOSD (Hard)"
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            client_hist = data["metric_log"]["client_metric_history"]
            unlearn_cids = data.get("params", {}).get("unlearn_client_id_list", [])
            
            unlearn_asrs = []
            f_accs = []
            mias = []
            for cid, hist in enumerate(client_hist):
                if cid in unlearn_cids or ("backdoor_test_accuracy" in hist and hist["backdoor_test_accuracy"] and hist["backdoor_test_accuracy"][-1] is not None):
                    b_acc = hist.get("backdoor_test_accuracy", [])
                    if b_acc and b_acc[-1] is not None:
                        unlearn_asrs.append(b_acc[-1])
                    f_acc = hist.get("forget_acc", [])
                    if f_acc and f_acc[-1] is not None:
                        f_accs.append(f_acc[-1])
                    m_auc = hist.get("mia_auc", [])
                    if m_auc and m_auc[-1] is not None:
                        mias.append(m_auc[-1])
                    if cid not in unlearn_cids:
                        unlearn_cids.append(cid)
            
            retained_accs = []
            retained_drops = []
            for cid, hist in enumerate(client_hist):
                if cid not in unlearn_cids and "test_accuracy" in hist and hist["test_accuracy"]:
                    t_acc = hist["test_accuracy"]
                    init_acc = t_acc[0]
                    final_acc = t_acc[-1]
                    retained_accs.append(final_acc)
                    retained_drops.append(init_acc - final_acc)
            
            if retained_accs:
                mean_acc = float(np.mean(retained_accs))
                std_acc = float(np.std(retained_accs))
                min_acc = float(np.min(retained_accs))
                max_acc = float(np.max(retained_accs))
                max_drop = float(np.max(retained_drops)) if retained_drops else 0.0
                asr_val = float(np.mean(unlearn_asrs)) if unlearn_asrs else 0.0
                f_val = float(np.mean(f_accs)) if f_accs else 0.0
                mia_val = float(np.mean(mias)) if mias else 0.5
                
                runs[config_key][seed] = {
                    "asr": asr_val,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "max_acc": max_acc,
                    "max_drop": max_drop,
                    "forget_clean": f_val,
                    "mia": mia_val
                }
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    print("=" * 145)
    print(f"{'Configuration':<34} | {'Seeds':<7} | {'ASR (%)':<8} | {'Retained Mean +/- Std':<24} | {'Worst-Off (Min)':<16} | {'Max Drop (%)':<14} | {'Forget Clean':<13} | {'MIA AUC':<8}")
    print("=" * 145)

    for config, seed_data in sorted(runs.items()):
        seeds = sorted(seed_data.keys())
        seed_str = ",".join(str(s) for s in seeds)
        
        asrs = [seed_data[s]["asr"] for s in seeds]
        means = [seed_data[s]["mean_acc"] for s in seeds]
        stds = [seed_data[s]["std_acc"] for s in seeds]
        mins = [seed_data[s]["min_acc"] for s in seeds]
        drops = [seed_data[s]["max_drop"] for s in seeds]
        f_cleans = [seed_data[s]["forget_clean"] for s in seeds]
        mias = [seed_data[s]["mia"] for s in seeds]
        
        avg_asr = float(np.mean(asrs))
        avg_mean = float(np.mean(means))
        avg_std = float(np.mean(stds))
        avg_min = float(np.min(mins)) if len(seeds) == 1 else float(np.mean(mins))
        avg_drop = float(np.mean(drops))
        avg_f = float(np.mean(f_cleans))
        avg_mia = float(np.mean(mias))
        
        mean_str = f"{avg_mean:.2f}% +/- {avg_std:.2f}%"
        min_str = f"{avg_min:.2f}%"
        drop_str = f"{avg_drop:.2f}%"
        f_str = f"{avg_f:.2f}%"
        mia_str = f"{avg_mia:.4f}"
            
        print(f"{config:<34} | {seed_str:<7} | {avg_asr:<7.2f}% | {mean_str:<24} | {min_str:<16} | {drop_str:<14} | {f_str:<13} | {mia_str:<8}")

    print("=" * 145)

if __name__ == "__main__":
    analyze_multiseed()
