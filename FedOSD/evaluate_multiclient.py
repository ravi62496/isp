import os
import json
import glob
import math
import re
from collections import defaultdict

def evaluate_folder(base_dir, un_label="UN=2"):
    json_files = glob.glob(os.path.join(base_dir, "*_unlearn.json"))
    if not json_files:
        print(f"No unlearn JSON files found in {base_dir}")
        return {}

    results = {}

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
            elif eta_val == 0.0:
                config_key = f"Soft Uniform (lam={lam_val:g}, eta=0.0)"
            else:
                config_key = f"Adaptive FedOSD (lam={lam_val:g}, eta={eta_val:g})"
        else:
            config_key = "Original FedOSD (Hard)"
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
                
            client_hist = data["metric_log"]["client_metric_history"]
            
            unlearn_cids = []
            unlearn_asrs = []
            for cid, hist in enumerate(client_hist):
                if "backdoor_test_accuracy" in hist and hist["backdoor_test_accuracy"]:
                    unlearn_cids.append(cid)
                    unlearn_asrs.append(hist["backdoor_test_accuracy"][-1])
                    
            retained_accs = []
            for cid, hist in enumerate(client_hist):
                if cid not in unlearn_cids and "test_accuracy" in hist and hist["test_accuracy"]:
                    retained_accs.append(hist["test_accuracy"][-1])
                    
            if retained_accs:
                mean_acc = sum(retained_accs) / len(retained_accs)
                var = sum((x - mean_acc) ** 2 for x in retained_accs) / len(retained_accs)
                std_acc = math.sqrt(var)
                min_acc = min(retained_accs)
                max_acc = max(retained_accs)
                mean_asr = sum(unlearn_asrs) / len(unlearn_asrs) if unlearn_asrs else 0.0
                
                results[config_key] = {
                    "un_label": un_label,
                    "unlearn_cids": unlearn_cids,
                    "mean_asr": mean_asr,
                    "mean_acc": mean_acc,
                    "std_acc": std_acc,
                    "min_acc": min_acc,
                    "max_acc": max_acc,
                    "retained_count": len(retained_accs)
                }
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            
    return results

def main():
    dir_un1 = "UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"
    dir_un2 = "UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN2/E1/C1.0/FedOSD"
    
    res_un1 = evaluate_folder(dir_un1, "UN=1")
    res_un2 = evaluate_folder(dir_un2, "UN=2")
    
    print("\n" + "=" * 115)
    print("Multi-Client Federated Unlearning Benchmark Comparison (UN = 1 vs. UN = 2)")
    print("=" * 115)
    print(f"{'Configuration':<32} | {'UN':<5} | {'Mean ASR':<10} | {'Retained Mean +/- Std':<26} | {'Worst-Off (Min)':<16} | {'Best-Off (Max)':<14}")
    print("-" * 115)
    
    for un_label, res_dict in [("UN=1", res_un1), ("UN=2", res_dict_un2 if 'res_dict_un2' in locals() else res_un2)]:
        if not res_dict:
            print(f"[{un_label}] Runs pending or in progress...")
            continue
        for cfg in sorted(res_dict.keys()):
            r = res_dict[cfg]
            mean_str = f"{r['mean_acc']:.2f}% +/- {r['std_acc']:.2f}%"
            min_str = f"{r['min_acc']:.2f}%"
            max_str = f"{r['max_acc']:.2f}%"
            asr_str = f"{r['mean_asr']:.2f}%"
            print(f"{cfg:<32} | {un_label:<5} | {asr_str:<10} | {mean_str:<26} | {min_str:<16} | {max_str:<14}")
        print("-" * 115)

if __name__ == "__main__":
    main()
