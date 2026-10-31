import os
import json
import glob
import math
import re

def evaluate_cifar100(base_dir="UnlearningTask/CNN_CIFAR10/cifar100 pat balanced B100 NC10 N10/UN1/E1/C1.0/FedOSD"):
    for stage_name, pattern in [("Unlearning Stage (Round 100)", "*_unlearn.json"), ("Full Recovery Stage (Round 200)", "*.json")]:
        if pattern == "*.json":
            json_files = [f for f in glob.glob(os.path.join(base_dir, pattern)) if not f.endswith("_unlearn.json")]
        else:
            json_files = glob.glob(os.path.join(base_dir, pattern))
            
        if not json_files:
            continue

        print("\n" + "=" * 115)
        print(f"CIFAR-100 Benchmark: {stage_name}")
        print("=" * 115)
        print(f"{'Configuration':<32} | {'lambda':<8} | {'eta':<6} | {'Unlearn ASR':<12} | {'Retained Mean +/- Std (Min, Max)':<35}")
        print("-" * 115)

        for fpath in sorted(json_files):
            fname = os.path.basename(fpath)
            
            lam_match = re.search(r"lam([0-9]+(?:\.[0-9]+)?)", fname)
            eta_match = re.search(r"eta([0-9]+(?:\.[0-9]+)?)", fname)
            
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
                per_client = {}
                for cid, hist in enumerate(client_hist):
                    if cid not in unlearn_cids and "test_accuracy" in hist and hist["test_accuracy"]:
                        acc = hist["test_accuracy"][-1]
                        retained_accs.append(acc)
                        per_client[cid] = acc
                        
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

    # Now print detailed per-client comparison at round 200
    print("\n" + "=" * 115)
    print("CIFAR-100 Per-Client Accuracy Breakdown (Post-Training / Round 200)")
    print("=" * 115)
    header = f"{'Client ID':<10} | {'Hard FedOSD':<15} | {'Soft Uniform':<15} | {'Adaptive FedOSD':<18} | {'Adaptive Delta':<15}"
    print(header)
    print("-" * 115)

    files_map = {
        "Hard": glob.glob(os.path.join(base_dir, "*lam100000.0*.json")),
        "Soft": glob.glob(os.path.join(base_dir, "*lam20.0 eta0.0.json")),
        "Adaptive": glob.glob(os.path.join(base_dir, "*lam20.0 eta0.4.json"))
    }
    
    # Filter out _unlearn.json
    for k in files_map:
        files_map[k] = [f for f in files_map[k] if not f.endswith("_unlearn.json")]

    acc_data = {}
    for k in ["Hard", "Soft", "Adaptive"]:
        if files_map[k]:
            with open(files_map[k][0], "r") as f:
                d = json.load(f)
            client_hist = d["metric_log"]["client_metric_history"]
            acc_data[k] = [hist["test_accuracy"][-1] if hist["test_accuracy"] else 0.0 for hist in client_hist]
            
    if "Hard" in acc_data and "Soft" in acc_data and "Adaptive" in acc_data:
        for cid in range(10):
            if cid == 2:
                # Unlearned client
                hard_val = f"{acc_data['Hard'][cid]:.2f}% (ASR)"
                soft_val = f"{acc_data['Soft'][cid]:.2f}% (ASR)"
                adap_val = f"{acc_data['Adaptive'][cid]:.2f}% (ASR)"
                delta = "Unlearned"
            else:
                hard_val = f"{acc_data['Hard'][cid]:.2f}%"
                soft_val = f"{acc_data['Soft'][cid]:.2f}%"
                adap_val = f"{acc_data['Adaptive'][cid]:.2f}%"
                d_val = acc_data['Adaptive'][cid] - acc_data['Soft'][cid]
                delta = f"{'+' if d_val >= 0 else ''}{d_val:.2f}%"
            print(f"{cid:<10} | {hard_val:<15} | {soft_val:<15} | {adap_val:<18} | {delta:<15}")
    print("=" * 115)

if __name__ == "__main__":
    evaluate_cifar100()
