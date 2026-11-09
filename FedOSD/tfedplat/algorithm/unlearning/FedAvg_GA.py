import tfedplat as fp
import time
import torch


class FedAvg_GA(fp.UnlearnAlgorithm):
    """
    FedAvg-GA (Gradient Ascent) literature baseline:
    Applies gradient ascent directly on the unlearned client's data
    without orthogonal projection or conflict mitigation.
    """
    def __init__(self,
                 name='FedAvg_GA',
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

    def train_a_round(self):
        com_time_start = time.time()
        m_locals, l_locals, g_locals = self.train()
        com_time_end = time.time()

        if not self.recovery_stage:
            gu_locals = []
            for idx, client in enumerate(self.online_client_list):
                if client.unlearn_flag:
                    gu_locals.append(g_locals[idx])

            if len(gu_locals) > 0:
                gu_locals = torch.vstack(gu_locals)
                weights = torch.Tensor([1 / len(gu_locals)] * len(gu_locals)).float().to(self.device)
                gu = weights @ gu_locals
                d = gu  # unprojected gradient ascent on unlearning loss
            else:
                d = torch.zeros_like(g_locals[0])

            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)
            cal_time = 0.001
        else:
            weights = torch.Tensor([1 / len(l_locals)] * len(l_locals)).float().to(self.device)
            d = weights @ g_locals
            lr = self.lr
            self.update_module(self.module, self.optimizer, lr, d)
            cal_time = 0.001

        self.communication_time += com_time_end - com_time_start
        self.computation_time += cal_time

    def run(self):
        for client in self.client_list:
            if client.unlearn_flag:
                client.criterion = self.UnLearningCELoss()

        while not self.terminated():
            self.train_a_round()

