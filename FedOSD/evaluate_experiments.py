import os
import json
import glob
import math
import re

def analyze_logs(base_dir="UnlearningTask/CNN_CIFAR10/cifar10 pat balanced B200 NC2 N10/UN1/E1/C1.0/FedOSD"):
    json_files = glob.glob(os.path.join(base_dir, "*_unlearn.json"))
    if not json_files:
        print(f"No unlearn JSON files found in {base_dir}")
        return

    print("=" * 115)
    print(f"{'Configuration':<38} | {'Seed':<5} | {'Unlearn ASR':<12} | {'Retained Mean +/- Std (Min, Max)':<35}")
    print("=" * 115)

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        
        seed_match = re.search(r"seed(\d+)", fname)
        seed = seed_match.group(1) if seed_match else "1"

        lam_match = re.search(r"lam([0-9]+(?:\.[0-9]+)?)", fname)
        eta_match = re.search(r"eta([0-9]+(?:\.[0-9]+)?)", fname)
        wm_match = re.search(r"wm_([a-z]+)", fname)
        
        wm = wm_match.group(1) if wm_match else ""
        if "proj_one_sided" in fname:
            cfg_name = f"One-Sided Hard (lam=Inf)" if "100000" in fname else f"One-Sided Soft ({wm})"
        elif "wm_drop" in fname:
            cfg_name = "Adaptive Drop"
        elif "wm_deficit" in fname:
            cfg_name = "Adaptive Deficit"
        elif "wm_conflict" in fname:
            cfg_name = "Adaptive Conflict"
        elif lam_match and eta_match:
            lam_val = float(lam_match.group(1))
            eta_val = float(eta_match.group(1))
            if lam_val >= 10000.0:
                cfg_name = "Original FedOSD (Hard)"
            elif eta_val == 0.0:
                cfg_name = f"Soft Uniform (lam={lam_val:g})"
            else:
                cfg_name = f"Soft (lam={lam_val:g}, eta={eta_val:g})"
        else:
            cfg_name = "Original FedOSD (Hard)"
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            client_hist = data["metric_log"]["client_metric_history"]
            unlearn_cids = data.get("params", {}).get("unlearn_client_id_list", [])
            
            asr_final = None
            for cid, hist in enumerate(client_hist):
                if (cid in unlearn_cids or ("backdoor_test_accuracy" in hist and hist["backdoor_test_accuracy"] and hist["backdoor_test_accuracy"][-1] is not None)):
                    b_acc = hist.get("backdoor_test_accuracy", [])
                    if b_acc and b_acc[-1] is not None:
                        asr_final = b_acc[-1]
                        if cid not in unlearn_cids:
                            unlearn_cids.append(cid)
            
            retained_accs = []
            for cid, hist in enumerate(client_hist):
                if cid not in unlearn_cids and "test_accuracy" in hist and hist["test_accuracy"]:
                    retained_accs.append(hist["test_accuracy"][-1])
            
            asr_str = f"{asr_final:.2f}%" if asr_final is not None else "N/A"
            if retained_accs:
                mean_acc = sum(retained_accs) / len(retained_accs)
                variance = sum((x - mean_acc) ** 2 for x in retained_accs) / len(retained_accs)
                std_acc = math.sqrt(variance)
                min_acc = min(retained_accs)
                max_acc = max(retained_accs)
                ret_str = f"{mean_acc:.2f}% +/- {std_acc:.2f}% [{min_acc:.1f}%, {max_acc:.1f}%]"
            else:
                ret_str = "N/A"
                
            print(f"{cfg_name:<38} | {seed:<5} | {asr_str:<12} | {ret_str:<35}")
        except Exception as e:
            print(f"{cfg_name:<38} | {seed:<5} | Error reading: {e}")
            
    print("=" * 115)

if __name__ == "__main__":
    analyze_logs()
