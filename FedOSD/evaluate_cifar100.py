import os
import json
import glob
import math
import re

def evaluate_cifar100(base_dir="UnlearningTask/CNN_CIFAR10/cifar100 pat balanced B100 NC10 N10/UN1/E1/C1.0/FedOSD"):
    json_files = glob.glob(os.path.join(base_dir, "*_unlearn.json"))
    if not json_files:
        print(f"No CIFAR-100 unlearn JSON files found in {base_dir}")
        return

    print("=" * 115)
    print("CIFAR-100 Federated Unlearning Generalization Benchmark")
    print("=" * 115)
    print(f"{'Configuration':<32} | {'lambda':<8} | {'eta':<6} | {'Unlearn ASR':<12} | {'Retained Mean +/- Std (Min, Max)':<35}")
    print("-" * 115)

    for fpath in sorted(json_files):
        fname = os.path.basename(fpath)
        
        lam_match = re.search(r"lam([0-9.]+)", fname)
        eta_match = re.search(r"eta([0-9.]+)", fname)
        
        if lam_match and eta_match:
            lam_val = float(lam_match.group(1))
            eta_val = float(eta_match.group(1))
            if lam_val >= 10000.0:
                cfg_name = "Original FedOSD (Hard)"
            elif eta_val == 0.0:
                cfg_name = f"Soft Uniform (lam={lam_val:g}, eta=0.0)"
            else:
                cfg_name = f"Adaptive FedOSD (lam={lam_val:g}, eta={eta_val:g})"
        else:
            lam_val = "Inf"
            eta_val = 0.0
            cfg_name = "Original FedOSD (Hard)"
            
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
                
                asr_str = f"{mean_asr:.2f}%"
                acc_str = f"{mean_acc:.2f}% +/- {std_acc:.2f}% (min: {min_acc:.2f}%, max: {max_acc:.2f}%)"
                print(f"{cfg_name:<32} | {str(lam_val):<8} | {str(eta_val):<6} | {asr_str:<12} | {acc_str:<35}")
        except Exception as e:
            print(f"Error parsing {fname}: {e}")
            
    print("=" * 115)

if __name__ == "__main__":
    evaluate_cifar100()
