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

    print("=" * 105)
    print(f"{'Configuration':<32} | {'lambda':<8} | {'eta':<6} | {'Unlearn ASR':<12} | {'Retained Mean +/- Std (Min, Max)':<35}")
    print("=" * 105)

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        
        # Parse lambda and eta if available
        lam_match = re.search(r"lam([0-9.]+)", fname)
        eta_match = re.search(r"eta([0-9.]+)", fname)
        
        if lam_match and eta_match:
            lam_val = lam_match.group(1)
            eta_val = eta_match.group(1)
            cfg_name = f"Soft (lam={lam_val}, eta={eta_val})"
        else:
            lam_val = "Inf"
            eta_val = "0.0"
            cfg_name = "Original FedOSD (Hard)"
            
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            
            client_hist = data["metric_log"]["client_metric_history"]
            
            asr_final = None
            if len(client_hist) > 2 and "backdoor_test_accuracy" in client_hist[2]:
                asr_history = client_hist[2]["backdoor_test_accuracy"]
                if asr_history:
                    asr_final = asr_history[-1]
            
            retained_accs = []
            for cid, hist in enumerate(client_hist):
                if cid != 2 and "test_accuracy" in hist and hist["test_accuracy"]:
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
                
            print(f"{cfg_name:<32} | {lam_val:<8} | {eta_val:<6} | {asr_str:<12} | {ret_str:<35}")
        except Exception as e:
            print(f"{cfg_name:<32} | Error reading: {e}")
            
    print("=" * 105)

if __name__ == "__main__":
    analyze_logs()
