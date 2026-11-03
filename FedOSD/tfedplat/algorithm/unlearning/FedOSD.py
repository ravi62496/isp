import tfedplat as fp
import time
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import nnls


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
        self.pre_acc = {}
        self.ema_acc = {}

        self.proj_mode = params.get('proj_mode', 'one_sided') if params is not None else 'one_sided'
        self.weight_mode = params.get('weight_mode', 'conflict') if params is not None else 'conflict'
        self.hybrid_k = int(params.get('hybrid_k', 2)) if params is not None else 2
        self.lam = float(params.get('adap_lam', 20.0)) if params is not None else 20.0
        self.weight_eta = float(params.get('adap_weight_eta', 0.4)) if params is not None else 0.4
        self.w_ema = float(params.get('w_ema', 0.2)) if params is not None else 0.2
        self.d_norm_cap = float(params.get('d_norm_cap', -1.0)) if params is not None else -1.0
        self.risk_signal = params.get('risk_signal', 'drop') if params is not None else 'drop'
        self.risk_ref = float(params.get('risk_ref', 10.0)) if params is not None else 10.0
        self.lam_soft = float(params.get('lam_soft', 20.0)) if params is not None else 20.0
        self.lam_hard = float(params.get('lam_hard', 100000.0)) if params is not None else 100000.0
        self.interp_alpha = float(params.get('interp_alpha', 0.3)) if params is not None else 0.3
        self.cap_hard_ratio = float(params.get('cap_hard_ratio', -1.0)) if params is not None else -1.0
        self.step_scale = float(params.get('step_scale', 1.0)) if params is not None else 1.0
        self.s_prev = {}

        print(f"[FedOSD INIT] proj_mode={self.proj_mode}, weight_mode={self.weight_mode}, lam={self.lam}, eta={self.weight_eta}, w_ema={self.w_ema}, hybrid_k={self.hybrid_k}, d_norm_cap={self.d_norm_cap}, cap_hard_ratio={self.cap_hard_ratio}")

        if self.proj_mode == 'hard':
            self.save_name += " proj_hard lam100000.0 eta0.0"
        elif self.proj_mode == 'interp':
            self.save_name += f" proj_interp_{self.risk_signal}_ref{self.risk_ref}_soft{self.lam_soft}"
        elif self.proj_mode == 'two_sided':
            self.save_name += f" proj_two_sided wm_{self.weight_mode} lam{self.lam} eta{self.weight_eta}"
        elif self.proj_mode == 'hybrid':
            self.save_name += f" proj_hybrid_k{self.hybrid_k} wm_{self.weight_mode} lam{self.lam} eta{self.weight_eta}"
        elif self.proj_mode == 'one_sided' and self.cap_hard_ratio > 0:
            self.save_name += f" proj_one_sided_cap{self.cap_hard_ratio} wm_{self.weight_mode} lam{self.lam}"
        else:
            self.save_name += f" proj_{self.proj_mode} wm_{self.weight_mode} lam{self.lam} eta{self.weight_eta}"

        if self.step_scale != 1.0:
            self.save_name += f" scale{self.step_scale}"
        if hasattr(self, 'r_lr') and self.r_lr != 1e-6 and self.r_lr > 0:
            self.save_name += f" rlr{self.r_lr}"

        if 'client_weights_history' not in self.algorithm_log:
            self.algorithm_log['client_weights_history'] = []
        if 'd_norm_history' not in self.algorithm_log:
            self.algorithm_log['d_norm_history'] = []

    def get_hard_d(self, gr_locals, gu):
        """
        Original FedOSD hard projection:
        min_d ||d - gu||^2  s.t.  g_i^T d = 0 for all i
        d = gu - A_hat^T (A_hat A_hat^T)^{-1} A_hat gu
        """
        start_time = time.time()
        A = gr_locals
        row_norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / row_norms

        AAT = A_norm @ A_norm.T
        jitter = 1e-6 * torch.eye(AAT.shape[0], device=AAT.device, dtype=AAT.dtype)
        Agu = A_norm @ gu
        inner = torch.linalg.solve(AAT + jitter, Agu.unsqueeze(-1)).squeeze(-1)
        correction = A_norm.T @ inner
        d = gu - correction
        cal_time = time.time() - start_time
        return d, cal_time

    def get_two_sided_soft_d(self, gr_locals, gu, weights, lam):
        """
        Two-sided soft constraint solve via Woodbury identity:
        min_d ||d - gu||^2 + lam * sum_i w_i * (g_hat_i^T d)^2
        d = gu - A_norm^T (diag(1/(lam * w)) + A_norm A_norm^T)^{-1} A_norm gu
        """
        start_time = time.time()
        A = gr_locals
        row_norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / row_norms

        AAT = A_norm @ A_norm.T
        inv_lam_w = 1.0 / (lam * weights + 1e-12)
        M = torch.diag(inv_lam_w) + AAT

        Agu = A_norm @ gu
        inner = torch.linalg.solve(M, Agu.unsqueeze(-1)).squeeze(-1)
        correction = A_norm.T @ inner
        d = gu - correction

        cal_time = time.time() - start_time
        return d, cal_time

    def get_one_sided_project_d(self, gr_locals, gu, weights, lam):
        """
        One-sided constraint solve via dual NNLS:
        min_d ||d - gu||^2 + sum_i (lam_i * w_i) * max(0, -g_hat_i^T d)^2
        Dual problem: minimize over nu >= 0:
            nu^T (G + D) nu + 2 nu^T b
        where G = A_norm A_norm^T, b = A_norm gu, D = diag(1 / (lam_i * w_i))
        Via Cholesky M = G + D + 1e-10 I = L L^T, y = -L^{-1} b:
            minimize over nu >= 0: ||L^T nu - y||^2  (solved by SciPy NNLS)
        d = gu + A_norm^T nu
        """
        start_time = time.time()
        A = gr_locals
        row_norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / row_norms

        G = (A_norm @ A_norm.T).double().cpu().numpy()
        b = (A_norm @ gu).double().cpu().numpy()
        n = len(b)

        lam_arr = np.broadcast_to(np.asarray(lam, dtype=np.float64), (n,))
        if isinstance(weights, torch.Tensor):
            w_arr = weights.double().cpu().numpy()
        else:
            w_arr = np.broadcast_to(np.asarray(weights, dtype=np.float64), (n,))

        inv_lam_w = 1.0 / (lam_arr * w_arr + 1e-12)
        M = G + np.diag(inv_lam_w) + 1e-10 * np.eye(n)

        try:
            L = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            M += 1e-6 * np.eye(n)
            L = np.linalg.cholesky(M)

        y = -np.linalg.solve(L, b)
        nu, _ = nnls(L.T, y)
        nu_t = torch.as_tensor(nu, dtype=A_norm.dtype, device=A_norm.device)
        d = gu + A_norm.T @ nu_t

        cal_time = time.time() - start_time
        return d, cal_time

    def get_hybrid_d(self, gr_locals, gu, weights, lam_soft, k):
        """
        Hybrid Hard + Soft Constraint:
        Identifies top-k most conflicted clients (g_hat_i^T gu < 0).
        Assigns hard constraint (lam=1e6) to top-k clients to ensure zero collapse,
        and soft constraint (lam_soft) to all remaining clients.
        """
        start_time = time.time()
        A = gr_locals
        row_norms = torch.norm(A, dim=1, keepdim=True) + 1e-8
        A_norm = A / row_norms

        gu_norm = torch.norm(gu) + 1e-12
        conf = torch.clamp(-(A_norm @ gu) / gu_norm, min=0.0).cpu().numpy()
        n = len(conf)
        k_val = max(1, min(int(k), n))
        top_k = np.argsort(-conf)[:k_val]

        lam_vec = np.full(n, float(lam_soft), dtype=np.float64)
        lam_vec[top_k] = 1e6  # effectively hard constraint

        d, _ = self.get_one_sided_project_d(gr_locals, gu, weights, lam_vec)
        cal_time = time.time() - start_time
        return d, cal_time

    def update_client_weights(self, retained_clients, gr_locals=None, gu=None):
        if self.client_weights is None:
            self.client_weights = {c.id: 1.0 for c in retained_clients}

        # Initialize pre-unlearning baseline validation accuracies at round 0 (no test leakage)
        if len(self.pre_acc) == 0:
            for client in retained_clients:
                cid = client.id
                h = self.metric_log['client_metric_history'][cid]
                val_hist = h.get('val_accuracy', [])
                if len(val_hist) > 0:
                    self.pre_acc[cid] = val_hist[0]
                elif len(h.get('test_accuracy', [])) > 0:
                    self.pre_acc[cid] = h['test_accuracy'][0]

        # Update EMA validation accuracy per client (strictly on held-out validation split)
        beta = 0.8
        for client in retained_clients:
            cid = client.id
            if cid not in self.client_weights:
                self.client_weights[cid] = 1.0
            h = self.metric_log['client_metric_history'][cid]
            val_history = h.get('val_accuracy', [])
            if len(val_history) == 0:
                val_history = h.get('test_accuracy', [])
            if len(val_history) == 0:
                continue
            current_acc = val_history[-1]
            if cid not in self.ema_acc:
                self.ema_acc[cid] = current_acc
            else:
                self.ema_acc[cid] = beta * self.ema_acc[cid] + (1 - beta) * current_acc

        valid_cids = [c.id for c in retained_clients if c.id in self.ema_acc]

        if self.weight_mode == 'uniform':
            for cid in self.client_weights:
                self.client_weights[cid] = 1.0

        elif self.weight_mode == 'deficit':
            if len(valid_cids) > 0 and self.weight_eta > 0.0:
                mean_group_acc = sum(self.ema_acc[cid] for cid in valid_cids) / len(valid_cids)
                for cid in valid_cids:
                    deficit = max(mean_group_acc - self.ema_acc[cid], 0.0)
                    target_w = 1.0 + self.weight_eta * (deficit / 10.0)
                    self.client_weights[cid] = 1.0 + (self.client_weights[cid] - 1.0) * (1.0 - self.w_ema) + (target_w - 1.0) * self.w_ema
                    self.client_weights[cid] = min(max(self.client_weights[cid], 0.2), 5.0)

        elif self.weight_mode == 'drop':
            # Improvement B1: accuracy drop relative to pre-unlearning baseline
            if len(valid_cids) > 0 and self.weight_eta > 0.0:
                for cid in valid_cids:
                    pre = self.pre_acc.get(cid, self.ema_acc[cid])
                    drop = max(pre - self.ema_acc[cid], 0.0)
                    target_w = 1.0 + self.weight_eta * (drop / 10.0)
                    self.client_weights[cid] = 1.0 + (self.client_weights[cid] - 1.0) * (1.0 - self.w_ema) + (target_w - 1.0) * self.w_ema
                    self.client_weights[cid] = min(max(self.client_weights[cid], 0.2), 5.0)

        elif self.weight_mode == 'conflict':
            # Improvement B2: instantaneous gradient conflict g_hat_i @ gu
            if gr_locals is not None and gu is not None and self.weight_eta > 0.0:
                row_norms = torch.norm(gr_locals, dim=1, keepdim=True) + 1e-8
                A_norm = gr_locals / row_norms
                gu_norm = torch.norm(gu) + 1e-12
                conf_scores = torch.clamp(-(A_norm @ gu) / gu_norm, min=0.0).cpu().numpy()
                mean_conf = float(np.mean(conf_scores))
                for idx, client in enumerate(retained_clients):
                    cid = client.id
                    conf_i = float(conf_scores[idx])
                    if mean_conf > 1e-6:
                        target_w = 1.0 + self.weight_eta * (conf_i / mean_conf)
                    else:
                        target_w = 1.0
                    self.client_weights[cid] = 1.0 + (self.client_weights[cid] - 1.0) * (1.0 - self.w_ema) + (target_w - 1.0) * self.w_ema
                    self.client_weights[cid] = min(max(self.client_weights[cid], 0.2), 5.0)

        # Log per-round weights
        self.algorithm_log['client_weights_history'].append({
            'round': self.current_comm_round,
            'weights': {c.id: round(self.client_weights[c.id], 3) for c in retained_clients}
        })

        if self.current_comm_round % 10 == 0:
            print(f"[weights round {self.current_comm_round}] mode={self.weight_mode}, w_ema={self.w_ema}: {self.client_weights}")

    def get_interp_project_d(self, gr_locals, gu, retained_clients):
        start_time = time.time()
        row_norms = torch.norm(gr_locals, dim=1, keepdim=True) + 1e-8
        A_hat = gr_locals / row_norms

        gu_norm = torch.norm(gu) + 1e-12
        conf_scores = torch.clamp(-(A_hat @ gu) / gu_norm, min=0.0).cpu().numpy()
        risk_vals = []
        for idx, client in enumerate(retained_clients):
            cid = client.id
            if self.risk_signal == 'conflict':
                r = float(conf_scores[idx])
            else:
                pre = self.pre_acc.get(cid, self.ema_acc.get(cid, 50.0))
                cur = self.ema_acc.get(cid, pre)
                r = max(pre - cur, 0.0)
            risk_vals.append(r)

        lam_vec = []
        for idx, client in enumerate(retained_clients):
            cid = client.id
            r = risk_vals[idx]
            s_prev_val = self.s_prev.get(cid, 0.0)
            s_new = float(np.clip(r / self.risk_ref, 0.0, 1.0))
            s = (1.0 - self.interp_alpha) * s_prev_val + self.interp_alpha * s_new
            self.s_prev[cid] = s
            lam_i = float(np.exp((1.0 - s) * np.log(self.lam_soft) + s * np.log(self.lam_hard)))
            lam_vec.append(lam_i)

        G = (A_hat @ A_hat.T).double().cpu().numpy()
        M = G + np.diag(1.0 / np.asarray(lam_vec, dtype=np.float64))
        b = (A_hat @ gu).double().cpu().numpy()
        nu = np.linalg.solve(M, b)
        nu_t = torch.as_tensor(nu, dtype=A_hat.dtype, device=A_hat.device)
        d = gu - A_hat.T @ nu_t
        cal_time = time.time() - start_time
        return d, cal_time, lam_vec

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

            # Update client protection weights
            self.update_client_weights(retained_clients, gr_locals=gr_locals, gu=gu)
            w_vec = torch.tensor([self.client_weights[c.id] for c in retained_clients],
                                 device=self.device, dtype=gr_locals.dtype)

            # Shadow hard projection for diagnostics and step-norm control
            row_norms = torch.norm(gr_locals, dim=1, keepdim=True) + 1e-8
            A_hat = gr_locals / row_norms
            G_mat = (A_hat @ A_hat.T).double()
            nu_hard = torch.linalg.lstsq(G_mat, (A_hat @ gu).double()).solution
            d_hard = gu - A_hat.T @ nu_hard.to(A_hat.dtype)
            d_hard_norm = float(torch.norm(d_hard).item())

            # Route to selected projection mode
            if self.proj_mode == 'hard':
                d, cal_time = self.get_hard_d(gr_locals, gu)
                lam_eff_list = [100000.0] * len(retained_clients)
            elif self.proj_mode == 'interp':
                d, cal_time, lam_eff_list = self.get_interp_project_d(gr_locals, gu, retained_clients)
            elif self.proj_mode == 'two_sided':
                w_norm = w_vec / (w_vec.sum() + 1e-12) * len(w_vec)
                d, cal_time = self.get_two_sided_soft_d(gr_locals, gu, w_norm, self.lam)
                lam_eff_list = [float(self.lam * float(w_i)) for w_i in w_norm.cpu().numpy()]
            elif self.proj_mode == 'hybrid':
                d, cal_time = self.get_hybrid_d(gr_locals, gu, w_vec, self.lam, self.hybrid_k)
                lam_eff_list = [float(self.lam * float(w_i)) for w_i in w_vec.cpu().numpy()]
            else:  # default 'one_sided'
                d, cal_time = self.get_one_sided_project_d(gr_locals, gu, w_vec, self.lam)
                lam_eff_list = [float(self.lam * float(w_i)) for w_i in w_vec.cpu().numpy()]

            # Step-norm control and descent step preservation
            d_raw_norm = float(torch.norm(d).item())
            self.algorithm_log['d_norm_history'].append(d_raw_norm)

            if self.cap_hard_ratio > 0:
                cap = self.cap_hard_ratio * d_hard_norm
                if d_raw_norm > cap:
                    d = d * (cap / (d_raw_norm + 1e-12))
            elif self.d_norm_cap > 0 and d_raw_norm > self.d_norm_cap:
                d = d / d_raw_norm * self.d_norm_cap
            elif d_raw_norm > 1e-8:
                d = d / d_raw_norm * g_norm
            else:
                d = gu

            # Apply step_scale before logging and updating
            if self.step_scale != 1.0:
                d = self.step_scale * d

            # Shadow diagnostics (logs the exact descent vector applied to the model)
            d_used_norm = float(torch.norm(d).item())
            capped_flag = bool(d_used_norm < (d_raw_norm * self.step_scale) - 1e-12)
            current_disp = float(torch.norm(self.module.span_model_params_to_vec() - self.init_model_params).item())
            cos_client = ((A_hat @ d) / (torch.norm(d) + 1e-12)).cpu().numpy().tolist()
            conf_client = torch.clamp(-(A_hat @ gu) / (gu_norm + 1e-12), min=0.0).cpu().numpy().tolist()
            if 'shadow_diagnostics' not in self.algorithm_log:
                self.algorithm_log['shadow_diagnostics'] = []
            self.algorithm_log['shadow_diagnostics'].append({
                'round': self.current_comm_round,
                'g_u_norm': float(gu_norm.item()),
                'd_hard_norm': d_hard_norm,
                'd_raw_norm': d_raw_norm,
                'd_used_norm': d_used_norm,
                'raw_over_hard': float(d_raw_norm / (d_hard_norm + 1e-12)),
                'used_over_hard': float(d_used_norm / (d_hard_norm + 1e-12)),
                'capped': capped_flag,
                'displacement': current_disp,
                'd_ratio': round(d_raw_norm / (d_hard_norm + 1e-12), 4),
                'min_cos_d_client': float(min(cos_client)) if cos_client else 0.0,
                'cos_d_client': cos_client,
                'conflict': conf_client,
                'w': [float(w_i) for w_i in w_vec.cpu().numpy()],
                'lam_eff': lam_eff_list
            })

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
                if ga_norm > 1e-8:
                    g_locals[idx] = grad - grad @ ga / (ga_norm**2) * ga
                g_locals[idx] = g_locals[idx] / (torch.norm(g_locals[idx]) + 1e-8) * g_norm

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
