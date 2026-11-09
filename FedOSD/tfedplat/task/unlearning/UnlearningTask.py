import tfedplat as fp
import numpy as np
import torch
import os
import copy
from scipy import stats
from torchvision.transforms import transforms
transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class UnlearningTask(fp.BasicTask):

    def __init__(self, name='UnlearningTask'):
        super().__init__(name)
        
        if self.params.get('results_dir'):
            self.algorithm.save_folder = self.params['results_dir'].rstrip('/\\') + '/'
        else:
            self.algorithm.save_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/C' + str(self.params['C']) + '/' + self.params['algorithm'] + '/'
        unlearn_pretrain_flag = self.params['unlearn_pretrain']
        retrain_flag = eval(str(self.params.get('retrain', 'False')))
        
        pretrained_model_folder = self.name + '/' + self.params['module'] + '/' + self.data_loader.nickname + '/UN' + str(self.params['unlearn_cn']) + '/E' + str(self.params['E']) + '/'
        self.algorithm.pretrained_model_folder = pretrained_model_folder

        if retrain_flag:
            # Retrain from scratch baseline: do not load pretrained weights
            print("[RETRAIN] Running retrain from scratch baseline on retained clients only.")
            init_seed_val = self.params.get('model_init_seed', -1)
            if init_seed_val > 0:
                print(f"[RETRAIN] Re-initializing model weights with model_init_seed={init_seed_val}")
                fp.setup_seed(seed=init_seed_val)
                self.algorithm.module.generate_model(self.data_loader.input_data_shape, self.data_loader.target_class_num)
                fp.setup_seed(seed=self.params['seed'])
                self.algorithm.save_name = f"seed{self.params['seed']} retrain_from_scratch init{init_seed_val} " + self.algorithm.save_name
            else:
                self.algorithm.save_name = f"seed{self.params['seed']} retrain_from_scratch " + self.algorithm.save_name
            if self.params.get('R', 200) != 200:
                self.algorithm.save_name += f" R{self.params['R']}"
            self.algorithm.init_model_params = self.algorithm.module.span_model_params_to_vec()
            self.algorithm.model_params = self.algorithm.module.span_model_params_to_vec()
        elif not unlearn_pretrain_flag:  
            model_path = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'
            if not os.path.exists(pretrained_model_folder):
                os.makedirs(pretrained_model_folder)
            if not os.path.isfile(model_path):
                raise RuntimeError(f'Please put the pretrained model in the path {model_path}.')
            self.algorithm.module.model.load_state_dict(torch.load(model_path))
            self.algorithm.init_model_params = self.algorithm.module.span_model_params_to_vec()  
            self.algorithm.model_params = self.algorithm.module.span_model_params_to_vec()  
        else:  
            self.algorithm.save_model = True
            self.algorithm.model_save_name = pretrained_model_folder + f'seed{self.params["seed"]}_unlearn_task_pretrained_model.pth'  
            if isinstance(self.algorithm, fp.UnlearnAlgorithm):
                raise RuntimeError(f'When setting unlearn_pretrain_flag=True, you cannot run unlearning FL algorithm.')
            self.algorithm.terminate_extra_execute = self.terminate_extra_execute
        
        self.params['unlearn_client_id_list'] = np.random.choice(self.algorithm.client_num, self.params['unlearn_cn'], replace=False).tolist()
        print('Unlearn clients:', self.params['unlearn_client_id_list'])
        self.algorithm.out_log += f"Unlearn clients: {self.params['unlearn_client_id_list']}\n"
        
        self.algorithm.metric_log['pre_acc'] = {}
        self.algorithm.metric_log['mia_auc'] = []

        # Split local test data into local_val_data and local_test_data to eliminate test-set leakage
        for client in self.algorithm.client_list:
            if hasattr(client, "local_test_data") and client.local_test_data is not None:
                n_batches = len(client.local_test_data)
                split_idx = max(1, n_batches // 2)
                val_batches = client.local_test_data[:split_idx]
                test_batches = client.local_test_data[split_idx:]
                setattr(client, "local_val_data", copy.deepcopy(val_batches))
                setattr(client, "local_val_number", sum(b[0].shape[0] for b in val_batches))
                client.local_test_data = test_batches
                client.local_test_number = sum(b[0].shape[0] for b in test_batches)

        for client in self.algorithm.client_list:
            if client.id in self.params['unlearn_client_id_list']:
                setattr(client, "unlearn_flag", True)
                pretrain_attack_portion = 0.8 if unlearn_pretrain_flag else 1.0
                self.modify_client(client, pretrain_attack_portion)  
            else:
                setattr(client, "unlearn_flag", False)
                client.test = self.ClientTest(self.algorithm.train_setting, self.algorithm.device)

        if retrain_flag:
            # Exclude unlearned clients completely from training online list
            self.algorithm.online_client_list = [c for c in self.algorithm.client_list if not c.unlearn_flag]

    @staticmethod
    def terminate_extra_execute(alg):
        alg.__class__.__bases__[0].terminate_extra_execute(alg)  

    def modify_client(self, client, attack_portion):
        # Preserve clean non-trigger training set, test set, and validation set for forgotten client
        setattr(client, "clean_training_data", copy.deepcopy(client.local_training_data))
        setattr(client, "clean_training_number", client.local_training_number)
        setattr(client, "clean_test_data", copy.deepcopy(client.local_test_data))
        setattr(client, "clean_test_number", client.local_test_number)
        setattr(client, "clean_val_data", copy.deepcopy(client.local_val_data))
        setattr(client, "clean_val_number", client.local_val_number)

        setattr(client, "local_backdoor_test_data", copy.deepcopy(client.local_training_data))
        setattr(client, "local_backdoor_test_number", client.local_training_number)
        
        backdoor = fp.FigRandBackdoor(dataloader=self.algorithm.data_loader, save_folder=self.algorithm.pretrained_model_folder + 'backdoors/', save_name=f'client_{client.id}_backdoor')
        backdoor.add_backdoor(client.local_training_data, attack_portion=attack_portion)
        backdoor.add_backdoor(client.local_backdoor_test_data, attack_portion=attack_portion)
        setattr(client, "backdoor_setting", backdoor)  
        
        # Test accuracy evaluates on backdoor test set
        client.local_test_data = copy.deepcopy(client.local_backdoor_test_data)
        client.local_test_number = client.local_backdoor_test_number
        
        client.test = self.ClientTest(self.algorithm.train_setting, self.algorithm.device)

    @staticmethod
    def compute_membership_inference(client, device):
        """
        Loss-threshold Membership Inference Attack (MIA):
        Evaluates cross-entropy loss on clean training samples of the unlearned client (member=1)
        vs. clean held-out test samples of the same client/classes (non-member=0).
        Compares like with like on clean samples.
        """
        clean_train = getattr(client, "clean_training_data", None)
        clean_test = getattr(client, "clean_test_data", None)
        if clean_train is None or clean_test is None:
            return 0.5
        client.test_module.model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

        member_scores = []
        with torch.no_grad():
            for [batch_x, batch_y] in clean_train:
                bx = fp.Module.change_data_device(batch_x, device)
                by = fp.Module.change_data_device(batch_y, device)
                logits = client.test_module.model(bx)
                probs = torch.softmax(logits, dim=1)
                p_true = torch.gather(probs, 1, by.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
                s = torch.log(p_true / (1.0 - p_true))
                member_scores.extend(s.cpu().numpy().tolist())

        non_member_scores = []
        with torch.no_grad():
            for [batch_x, batch_y] in clean_test:
                bx = fp.Module.change_data_device(batch_x, device)
                by = fp.Module.change_data_device(batch_y, device)
                logits = client.test_module.model(bx)
                probs = torch.softmax(logits, dim=1)
                p_true = torch.gather(probs, 1, by.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0 - 1e-6)
                s = torch.log(p_true / (1.0 - p_true))
                non_member_scores.extend(s.cpu().numpy().tolist())

        if len(member_scores) == 0 or len(non_member_scores) == 0:
            return 0.5, 0.0

        y_true = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])
        scores = np.concatenate([np.array(member_scores), np.array(non_member_scores)])

        n_total = len(scores)
        n_unique = len(np.unique(scores.round(decimals=5)))
        tied_frac = float(1.0 - (n_unique / n_total))

        if np.all(np.isclose(scores, scores[0], atol=1e-5)):
            return 0.5, 1.0

        rank_sum = stats.rankdata(scores)
        u1 = np.sum(rank_sum[y_true == 1]) - len(member_scores) * (len(member_scores) + 1) / 2.0
        auc = u1 / (len(member_scores) * len(non_member_scores) + 1e-12)
        return float(np.clip(auc, 0.0, 1.0)), tied_frac

    @staticmethod
    class ClientTest:
        def __init__(self, train_setting, device):
            self.train_setting = train_setting
            self.device = device
            
            self.metric_history = {
                'training_loss': [],
                'test_loss': [],
                'local_test_number': 0,
                'test_accuracy': [],
                'val_accuracy': [],
                'pre_acc': None,
                'backdoor_test_loss': [],
                'backdoor_test_accuracy': [],
                'clean_test_accuracy': [],
                'forget_acc': [],
                'target_class_recall': [],
                'mia_auc': []
            }

        def run(self, client):
            client.test_module.model.eval()
            criterion = self.train_setting['criterion'].to(self.device)
            
            self.metric_history['training_loss'].append(float(client.upload_loss) if client.upload_loss is not None else None)
            
            metric_dict = {'test_loss': 0, 'correct': 0}
            correct_metric = fp.Correct()
            
            with torch.no_grad():
                self.metric_history['local_test_number'] = client.local_test_number
                for [batch_x, batch_y] in client.local_test_data:
                    batch_x = fp.Module.change_data_device(batch_x, self.device)
                    batch_y = fp.Module.change_data_device(batch_y, self.device)
                    out = client.test_module.model(batch_x)  
                    loss = criterion(out, batch_y).item()
                    metric_dict['test_loss'] += float(loss) * batch_y.shape[0]  
                    metric_dict['correct'] += correct_metric.calc(out, batch_y)
                
                curr_acc = 100 * metric_dict['correct'] / max(1, client.local_test_number)
                self.metric_history['test_loss'].append(round(metric_dict['test_loss'] / max(1, client.local_test_number), 4))
                self.metric_history['test_accuracy'].append(curr_acc)
                if self.metric_history['pre_acc'] is None:
                    self.metric_history['pre_acc'] = curr_acc

                # Evaluate on held-out validation split (used strictly for parameter/weight adaptation)
                if hasattr(client, "local_val_data") and client.local_val_data is not None:
                    val_metric_dict = {'test_loss': 0, 'correct': 0}
                    for [batch_x, batch_y] in client.local_val_data:
                        batch_x = fp.Module.change_data_device(batch_x, self.device)
                        batch_y = fp.Module.change_data_device(batch_y, self.device)
                        out = client.test_module.model(batch_x)
                        val_loss = criterion(out, batch_y).item()
                        val_metric_dict['test_loss'] += float(val_loss) * batch_y.shape[0]
                        val_metric_dict['correct'] += correct_metric.calc(out, batch_y)
                    val_acc = 100 * val_metric_dict['correct'] / max(1, client.local_val_number)
                    self.metric_history['val_accuracy'].append(val_acc)
                else:
                    self.metric_history['val_accuracy'].append(curr_acc)
                
                if hasattr(client, 'local_backdoor_test_data') and client.local_backdoor_test_data is not None:
                    metric_dict = {'test_loss': 0, 'correct': 0}
                    for [batch_x, batch_y] in client.local_backdoor_test_data:
                        batch_x = fp.Module.change_data_device(batch_x, self.device)
                        batch_y = fp.Module.change_data_device(batch_y, self.device)
                        out = client.test_module.model(batch_x)  
                        loss = criterion(out, batch_y).item()
                        metric_dict['test_loss'] += float(loss) * batch_y.shape[0]  
                        metric_dict['correct'] += correct_metric.calc(out, batch_y)
                    self.metric_history['backdoor_test_loss'].append(round(metric_dict['test_loss'] / max(1, client.local_backdoor_test_number), 4))
                    self.metric_history['backdoor_test_accuracy'].append(100 * metric_dict['correct'] / max(1, client.local_backdoor_test_number))
                else:
                    self.metric_history['backdoor_test_loss'].append(None)
                    self.metric_history['backdoor_test_accuracy'].append(None)

                # Clean test accuracy and forget-set utility
                if hasattr(client, "clean_test_data") and client.clean_test_data is not None:
                    clean_correct = 0
                    target_correct = 0
                    target_total = 0
                    for [batch_x, batch_y] in client.clean_test_data:
                        batch_x = fp.Module.change_data_device(batch_x, self.device)
                        batch_y = fp.Module.change_data_device(batch_y, self.device)
                        out = client.test_module.model(batch_x)
                        clean_correct += correct_metric.calc(out, batch_y)
                        preds = torch.argmax(out, dim=1)
                        # Backdoor target class shift is +5 % 10
                        target_class = (batch_y + 5) % 10
                        target_mask = (batch_y == target_class)
                        if target_mask.sum() > 0:
                            target_total += target_mask.sum().item()
                            target_correct += (preds[target_mask] == batch_y[target_mask]).sum().item()

                    f_acc = 100 * clean_correct / max(1, client.clean_test_number)
                    self.metric_history['clean_test_accuracy'].append(f_acc)
                    self.metric_history['forget_acc'].append(f_acc)
                    recall = (100 * target_correct / target_total) if target_total > 0 else f_acc
                    self.metric_history['target_class_recall'].append(recall)

                    # Compute membership inference attack AUC
                    mia, tied_frac = UnlearningTask.compute_membership_inference(client, self.device)
                    self.metric_history['mia_auc'].append(mia)
                    if 'mia_tied_frac' not in self.metric_history:
                        self.metric_history['mia_tied_frac'] = []
                    self.metric_history['mia_tied_frac'].append(tied_frac)

    @staticmethod
    def outFunc(alg):
        # Record pre-unlearning baseline validation accuracy at round 0
        if 'pre_acc' not in alg.metric_log:
            alg.metric_log['pre_acc'] = {}
        if alg.current_comm_round == 0:
            for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
                val_accs = metric_history.get('val_accuracy', [])
                if len(val_accs) > 0:
                    alg.metric_log['pre_acc'][str(i)] = val_accs[0]
                elif len(metric_history['test_accuracy']) > 0:
                    alg.metric_log['pre_acc'][str(i)] = metric_history['test_accuracy'][0]

        # Log peak GPU memory allocation
        if torch.cuda.is_available():
            peak_gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_gpu_mem = 0.0
        alg.algorithm_log['max_memory_allocated'] = peak_gpu_mem

        unlearned_client_loss_list = []
        retained_client_loss_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            training_loss = metric_history['training_loss'][-1]
            if training_loss is None:
                continue
            if alg.client_list[i].unlearn_flag:
                unlearned_client_loss_list.append(training_loss)
            else:
                retained_client_loss_list.append(training_loss)
        
        unlearned_client_local_acc_list = []
        retained_client_local_acc_list = []
        unlearned_clean_acc_list = []
        target_recalls = []
        mia_aucs = []

        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            test_acc = metric_history['test_accuracy'][-1]
            if alg.client_list[i].unlearn_flag:
                unlearned_client_local_acc_list.append(test_acc)
                if 'forget_acc' in metric_history and metric_history['forget_acc']:
                    unlearned_clean_acc_list.append(metric_history['forget_acc'][-1])
                if 'target_class_recall' in metric_history and metric_history['target_class_recall']:
                    target_recalls.append(metric_history['target_class_recall'][-1])
                if 'mia_auc' in metric_history and metric_history['mia_auc']:
                    mia_aucs.append(metric_history['mia_auc'][-1])
            else:
                retained_client_local_acc_list.append(test_acc)
        
        unlearned_client_local_backdoor_acc_list = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            if alg.client_list[i].unlearn_flag:
                test_acc = metric_history['backdoor_test_accuracy'][-1]
                unlearned_client_local_backdoor_acc_list.append(test_acc)

        unlearned_client_local_acc_list = np.array(unlearned_client_local_acc_list)
        retained_client_local_acc_list = np.array(retained_client_local_acc_list)
        unlearned_client_local_backdoor_acc_list = np.array(unlearned_client_local_backdoor_acc_list)
        
        def cal_fairness(values):
            if len(values) == 0:
                return 0.0
            p = np.ones(len(values))
            norm_val = np.linalg.norm(values) * np.linalg.norm(p)
            if norm_val < 1e-12:
                return 0.0
            cos_val = np.clip(values @ p / norm_val, -1.0, 1.0)
            return float(np.arccos(cos_val))

        unlearned_client_fairness = cal_fairness(unlearned_client_local_acc_list)
        retained_client_fairness = cal_fairness(retained_client_local_acc_list)

        # Accuracy drops from pre-unlearning baseline for retained clients
        retained_drops = []
        for i, metric_history in enumerate(alg.metric_log['client_metric_history']):
            if not alg.client_list[i].unlearn_flag:
                test_accs = metric_history['test_accuracy']
                if len(test_accs) > 0:
                    init_acc = alg.metric_log['pre_acc'].get(str(i), test_accs[0])
                    curr_acc = test_accs[-1]
                    retained_drops.append((i, init_acc - curr_acc))

        out_log = ""
        out_log += alg.save_name + ' ' + alg.data_loader.nickname + '\n'
        out_log += 'Lr: ' + str(alg.lr) + '\n'
        out_log += f'round {alg.current_comm_round} training_num {alg.current_training_num}\n'
        if len(unlearned_client_loss_list) > 0:
            out_log += f'Unlearned Client Mean Global Test loss: {np.mean(unlearned_client_loss_list):.6f}\n'
        if len(unlearned_client_local_acc_list) > 0:
            out_log += f'Unlearned Client Local Test Acc: {np.mean(unlearned_client_local_acc_list)/100:.3f}({np.std(unlearned_client_local_acc_list)/100:.3f}), angle: {unlearned_client_fairness:.6f}, min: {np.min(unlearned_client_local_acc_list):.2f}%, max: {np.max(unlearned_client_local_acc_list):.2f}%\n'
        if len(unlearned_clean_acc_list) > 0:
            out_log += f'Forget-Set Clean Acc (forget_acc): {np.mean(unlearned_clean_acc_list):.2f}%\n'
        if len(target_recalls) > 0:
            out_log += f'Target-Class Clean Recall (target_class_recall): {np.mean(target_recalls):.2f}%\n'
        if len(mia_aucs) > 0:
            out_log += f'MIA ROC AUC (mia_auc): {np.mean(mia_aucs):.4f}\n'
        if len(unlearned_client_local_backdoor_acc_list) > 0:
            out_log += f'ASR: {np.mean(unlearned_client_local_backdoor_acc_list)/100:.3f}({np.std(unlearned_client_local_backdoor_acc_list)/100:.3f}), min: {np.min(unlearned_client_local_backdoor_acc_list):.2f}%, max: {np.max(unlearned_client_local_backdoor_acc_list):.2f}%\n'
        if len(retained_client_loss_list) > 0:
            out_log += f'Retained Client Mean Global Test loss: {np.mean(retained_client_loss_list):.6f}\n'
        if len(retained_client_local_acc_list) > 0:
            out_log += f'Retained Client Local Test Acc: {np.mean(retained_client_local_acc_list)/100:.3f}({np.std(retained_client_local_acc_list)/100:.3f}), angle: {retained_client_fairness:.6f}, min: {np.min(retained_client_local_acc_list):.2f}%, max: {np.max(retained_client_local_acc_list):.2f}%\n'
        
        if retained_drops:
            drop_vals = [d[1] for d in retained_drops]
            mean_drop = float(np.mean(drop_vals))
            max_drop_idx = int(np.argmax(drop_vals))
            max_drop = float(drop_vals[max_drop_idx])
            worst_client_id = retained_drops[max_drop_idx][0]
            out_log += f'Retained Accuracy Drop: mean={mean_drop:.2f}%, max={max_drop:.2f}% (Client {worst_client_id})\n'
            if 'retained_drops_history' not in alg.algorithm_log:
                alg.algorithm_log['retained_drops_history'] = []
            alg.algorithm_log['retained_drops_history'].append({'round': alg.current_comm_round, 'mean_drop': mean_drop, 'max_drop': max_drop, 'worst_client': worst_client_id})

        out_log += f'peak_gpu_mem_mb: {peak_gpu_mem:.2f} MB, communication_time: {alg.communication_time:.2f}s, computation_time: {alg.computation_time:.4f}s\n\n'
        alg.out_log = out_log + alg.out_log
        print(str(alg.name))
        print(out_log)

    def read_params(self, return_parser=False):
        parser = super().read_params(return_parser=True)

        parser.add_argument('--unlearn_cn', help='unlearn client num', type=int, default=0)
        parser.add_argument('--unlearn_pretrain', help='pretrain the model before unlearning', type=str, default=False)
        parser.add_argument('--retrain', help='Train from scratch on retained clients only', type=str, default='False')
        parser.add_argument('--UR', help='Unlearning round, must be smaller than R', type=int, default=100)
        parser.add_argument('--r_lr', help='Learning rate in the post-training', type=float, default=-1)  
        parser.add_argument('--adap_lam', help='Soft-constraint strength for adaptive FedOSD', type=float, default=20.0)
        parser.add_argument('--adap_weight_eta', help='Weight-update rate for adaptive FedOSD', type=float, default=0.4)
        parser.add_argument('--w_ema', help='Weight EMA rate for adapting client weights (0.05, 0.2, 0.5)', type=float, default=0.2)
        parser.add_argument('--proj_mode', help='Projection mode: hard, two_sided, one_sided, hybrid', type=str, default='one_sided')
        parser.add_argument('--weight_mode', help='Weight mode: uniform, deficit, drop, conflict', type=str, default='conflict')
        parser.add_argument('--hybrid_k', help='Number of hard-constrained clients in hybrid mode', type=int, default=2)
        parser.add_argument('--d_norm_cap', help='Step-norm control cap for one-sided projection', type=float, default=-1.0)
        parser.add_argument('--risk_signal', help='Risk signal for interpolation: drop or conflict', type=str, default='drop')
        parser.add_argument('--risk_ref', help='Risk reference threshold for hard protection', type=float, default=10.0)
        parser.add_argument('--lam_soft', help='Soft lambda floor for interpolation', type=float, default=20.0)
        parser.add_argument('--lam_hard', help='Hard lambda ceiling for interpolation', type=float, default=100000.0)
        parser.add_argument('--interp_alpha', help='EMA smoothing alpha for interpolation score', type=float, default=0.3)
        parser.add_argument('--cap_hard_ratio', help='Ratio of hard step norm to cap projection step (E3)', type=float, default=-1.0)
        parser.add_argument('--step_scale', help='Scale factor for unlearning update step d in (0, 1]', type=float, default=1.0)
        parser.add_argument('--results_dir', help='Custom output directory for result files', type=str, default='')

        try:
            if return_parser:
                return parser
            else:
                params = vars(parser.parse_args())
                if params['UR'] > params['R']:
                    raise RuntimeError('The parameter of UR must not be bigger than R.')
                return params
        except IOError as msg:
            parser.error(str(msg))


if __name__ == '__main__':
    my_task = UnlearningTask()
    my_task.run()
