import os
import json
import glob
import math
import re
from collections import defaultdict

def analyze_multiseed(base_dir="UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"):
    json_files = glob.glob(os.path.join(base_dir, "*_unlearn.json"))
    if not json_files:
        print(f"No unlearn JSON files found in {base_dir}")
        return

    runs = defaultdict(dict)

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        
        seed_match = re.search(r"seed(\d+)", fname)
        seed = int(seed_match.group(1)) if seed_match else 1
        
        lam_match = re.search(r"lam([0-9.]+)", fname)
        eta_match = re.search(r"eta([0-9.]+)", fname)
        
        if lam_match and eta_match:
            lam_val = float(lam_match.group(1))
            eta_val = float(eta_match.group(1))
            if lam_val >= 10000.0:
                config_key = "Original FedOSD (Hard)"
            else:
                config_key = f"Soft (lam={lam_val:g}, eta={eta_val:g})"
        else:
            config_key = "Original FedOSD (Hard)"
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            client_hist = data["metric_log"]["client_metric_history"]
            
            unlearn_cid = None
            asr_final = None
            for cid, hist in enumerate(client_hist):
                if "backdoor_test_accuracy" in hist and hist["backdoor_test_accuracy"]:
                    unlearn_cid = cid
                    asr_final = hist["backdoor_test_accuracy"][-1]
                    break
            
            retained_accs = []
            for cid, hist in enumerate(client_hist):
                if cid != unlearn_cid and "test_accuracy" in hist and hist["test_accuracy"]:
                    retained_accs.append(hist["test_accuracy"][-1])
            
            if retained_accs:
                mean_acc = sum(retained_accs) / len(retained_accs)
                var = sum((x - mean_acc) ** 2 for x in retained_accs) / len(retained_accs)
                std_acc = math.sqrt(var)
                min_acc = min(retained_accs)
                max_acc = max(retained_accs)
                
                runs[config_key][seed] = {
                    "asr": asr_final if asr_final is not None else 0.0,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "max_acc": max_acc
                }
        except Exception as e:
            print(f"Error parsing {fname}: {e}")

    print("=" * 115)
    print(f"{'Configuration':<30} | {'Seeds':<10} | {'Mean ASR':<10} | {'Retained Acc (Mean +/- Std)':<28} | {'Worst-Off Client (Min)':<22}")
    print("=" * 115)

    for config, seed_data in sorted(runs.items()):
        seeds = sorted(seed_data.keys())
        seed_str = ",".join(str(s) for s in seeds)
        
        asrs = [seed_data[s]["asr"] for s in seeds]
        means = [seed_data[s]["mean_acc"] for s in seeds]
        stds = [seed_data[s]["std_acc"] for s in seeds]
        mins = [seed_data[s]["min_acc"] for s in seeds]
        
        avg_asr = sum(asrs) / len(asrs)
        avg_mean = sum(means) / len(means)
        avg_std = sum(stds) / len(stds)
        avg_min = sum(mins) / len(mins)
        
        mean_str = f"{avg_mean:.2f}% +/- {avg_std:.2f}%"
        min_str = f"{avg_min:.2f}%"
        
        if len(seeds) > 1:
            mean_se = math.sqrt(sum((m - avg_mean)**2 for m in means) / len(means))
            min_se = math.sqrt(sum((m - avg_min)**2 for m in mins) / len(mins))
            mean_str += f" (SE +/- {mean_se:.2f}%)"
            min_str += f" (SE +/- {min_se:.2f}%)"
            
        print(f"{config:<30} | {seed_str:<10} | {avg_asr:<9.2f}% | {mean_str:<28} | {min_str:<22}")

    print("=" * 115)

if __name__ == "__main__":
    analyze_multiseed()
