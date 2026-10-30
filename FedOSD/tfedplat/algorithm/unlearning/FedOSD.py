import tfedplat as fp
import time
import torch
import torch.nn.functional as F


class FedOSD(fp.UnlearnAlgorithm):
    def __init__(self,
                 name='FedOSD',
                 data_loader=None,
                 module=None,
                 device=None,
                 train_setting=None,
                 client_num=None,
                 client_list=None,
                 online_client_num=None,
                 save_model=False,
                 max_comm_round=0,
                 max_training_num=0,
                 epochs=1,
                 save_name=None,
                 outFunc=None,
                 write_log=True,
                 dishonest=None,
                 test_conflicts=False,
                 params=None,
                 *args,
                 **kwargs):

        super().__init__(name, data_loader, module, device, train_setting, client_num, client_list, online_client_num, save_model, max_comm_round, max_training_num, epochs, save_name, outFunc, write_log, dishonest, test_conflicts, params)
        self.initial_g_norm = None

        self.client_weights = None
        self.lam = params['adap_lam'] if params is not None and 'adap_lam' in params else 1.0
        self.weight_eta = params['adap_weight_eta'] if params is not None and 'adap_weight_eta' in params else 0.05
        print(f"[DEBUG] Resolved lam={self.lam}, weight_eta={self.weight_eta}")
        self.save_name += f" lam{self.lam} eta{self.weight_eta}"

    def get_weighted_soft_d(self, gr_locals, gu, weights, lam):
        """
            Solves: min_d ||d - gu||^2 + lam * sum_i w_i * (g_i^T d)^2
            via Woodbury identity: d = gu - A^T (W^{-1}/lam + AA^T)^{-1} A gu
            Normalized gradient directions ensure diag(AA^T) == 1.0, making
            lambda and per-client weights scale-invariant across any neural architecture.
            As lam -> inf, this converges exactly to FedOSD's original hard
            orthogonality projection: d = gu - A^T(AA^T)^{-1} A gu.
        """
    
        start_time = time.time()

        A = gr_locals                                   # [n, p]
        row_norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / row_norms

        AAT = A_norm @ A_norm.T                          # [n, n] with unit diagonal
        inv_lam_w = 1.0 / (lam * weights)                 # [n]
        M = torch.diag(inv_lam_w) + AAT                    # [n, n]

        Agu = A_norm @ gu                                 # [n]
        inner = torch.linalg.solve(M, Agu.reshape(-1, 1)).reshape(-1)  # [n]

        correction = A_norm.T @ inner                     # [p]
        d = gu - correction

        cal_time = time.time() - start_time
        return d, cal_time
    
    
    def update_client_weights(self, retained_clients):
        if self.client_weights is None:
            self.client_weights = {c.id: 1.0 for c in retained_clients}
        if not hasattr(self, 'ema_acc'):
            self.ema_acc = {}

        beta = 0.8  # EMA smoothing factor
        for client in retained_clients:
            cid = client.id
            if cid not in self.client_weights:
                self.client_weights[cid] = 1.0
            acc_history = self.metric_log['client_metric_history'][cid]['test_accuracy']
            if len(acc_history) == 0:
                continue
            current_acc = acc_history[-1]
            if cid not in self.ema_acc:
                self.ema_acc[cid] = current_acc
            else:
                self.ema_acc[cid] = beta * self.ema_acc[cid] + (1 - beta) * current_acc

        valid_cids = [c.id for c in retained_clients if c.id in self.ema_acc]
        if len(valid_cids) > 0 and self.weight_eta > 0.0:
            mean_group_acc = sum(self.ema_acc[cid] for cid in valid_cids) / len(valid_cids)
            for cid in valid_cids:
                deficit = max(mean_group_acc - self.ema_acc[cid], 0.0)
                target_w = 1.0 + self.weight_eta * (deficit / 10.0)
                # Smooth update toward target_w and decay excess
                self.client_weights[cid] = 1.0 + (self.client_weights[cid] - 1.0) * 0.95 + (target_w - 1.0) * 0.05
                self.client_weights[cid] = min(max(self.client_weights[cid], 0.2), 5.0)

        if self.current_comm_round % 10 == 0:
            print(f"[weights] round {self.current_comm_round}: {self.client_weights}")            

    def train_a_round(self):
        com_time_start = time.time()

        m_locals, l_locals, g_locals = self.train()

        com_time_end = time.time()

        if not self.recovery_stage:
            gu_locals = []
            gr_locals = []
            retained_clients = []
            for idx, client in enumerate(self.online_client_list):
                if client.unlearn_flag:
                    gu_locals.append(g_locals[idx])
                else:
                    gr_locals.append(g_locals[idx])
                    retained_clients.append(client)
            gu_locals = torch.vstack(gu_locals)
            gr_locals = torch.vstack(gr_locals)

            weights = torch.Tensor([1 / len(gu_locals)] * len(gu_locals)).float().to(self.device)
            gu = weights @ gu_locals
            gu_norm = torch.norm(gu)
            g_norm = gu_norm

            self.update_client_weights(retained_clients)
            w_vec = torch.tensor([self.client_weights[c.id] for c in retained_clients],
                                  device=self.device, dtype=gr_locals.dtype)
            w_vec = w_vec / w_vec.sum() * len(w_vec)

            d, cal_time = self.get_weighted_soft_d(gr_locals, gu, w_vec, self.lam)

            d = d / torch.norm(d) * g_norm

            if self.test_conflicts:
                self.stat_update_conflict(d, gr_locals)

            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)
        else:

            model_params = self.module.span_model_params_to_vec()
            ga = model_params - self.init_model_params
            ga_norm = torch.norm(ga)

            for idx in range(len(g_locals)):
                grad = g_locals[idx]
                g_norm = torch.norm(grad)
                g_locals[idx] = grad - grad @ ga / ga_norm**2 * ga
                g_locals[idx] = g_locals[idx] / torch.norm(g_locals[idx]) * g_norm

            cal_time_start = time.time()

            weights = torch.Tensor([1 / len(l_locals)] * len(l_locals)).float().to(self.device)
            d = weights @ g_locals

            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)

            cal_time = time.time() - cal_time_start

        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time

    def run(self):

        for client in self.client_list:
            if client.unlearn_flag:
                client.criterion = self.UnLearningCELoss()

        while not self.terminated():
            self.train_a_round()