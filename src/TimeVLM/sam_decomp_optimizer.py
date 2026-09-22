import torch
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0
    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum
    model.apply(_enable)


class SAMDecompOptimizer(Optimizer):
    """
    APS + MDPS 融合优化器
    - APS: 自适应扰动（adaptive=True时生效）
    - MDPS: 多模态梯度分解扰动
    """
    def __init__(self, params, base_optimizer, model, rho=0.05, 
                 adaptive=True, perturb_eps=1e-12, **kwargs):
        # 初始化参数组
        for group in params:
            group.setdefault("rho", rho)
            group.setdefault("name", "other")
        
        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAMDecompOptimizer, self).__init__(params, defaults)
        
        self.model = model
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.perturb_eps = perturb_eps
        
        # 存储梯度和参数
        self.original_params = {}
        self.multi_gradients = {}
        self.uni_gradients = {}
        self.forward_backward_func = None

    def set_closure(self, loss_fn, inputs, targets):
        """
        设置损失闭包
        loss_fn 必须返回: (融合损失, 单模态1损失, 单模态2损失)
        """
        self.multi_gradients = {}
        self.uni_gradients = {}

        def get_grad(only_multi=False):
            self.base_optimizer.zero_grad()
            outputs = self.model(*inputs)
            loss_multi, loss_modal1, loss_modal2 = loss_fn(outputs, targets)
            
            if not only_multi:
                # 计算融合损失梯度
                loss_multi.backward(retain_graph=True)
                self.multi_gradients['modal1'] = self._store_module_gradients('modal1')
                self.multi_gradients['modal2'] = self._store_module_gradients('modal2')
                self.base_optimizer.zero_grad()
                
                # 计算单模态1损失梯度
                loss_modal1.backward(retain_graph=True)
                self.uni_gradients['modal1'] = self._store_module_gradients('modal1')
                self.base_optimizer.zero_grad()
                
                # 计算单模态2损失梯度
                loss_modal2.backward(retain_graph=True)
                self.uni_gradients['modal2'] = self._store_module_gradients('modal2')
                self.base_optimizer.zero_grad()
            
            total_loss = loss_multi + loss_modal1 + loss_modal2
            total_loss.backward()
            return total_loss.item(), loss_modal1.item(), loss_modal2.item()

        self.forward_backward_func = get_grad

    def _store_module_gradients(self, module_name):
        gradients = {}
        for group in self.param_groups:
            if group['name'] == module_name:
                for p in group['params']:
                    if p.grad is not None:
                        gradients[p] = p.grad.clone().detach()
        return gradients

    def _get_decomposed_gradients(self, g_u, g_m):
        """MDPS核心：梯度分解"""
        # 如果 g_u 或 g_m 是 float，说明这个参数不在这个模态里，返回 0
        if isinstance(g_u, float) or isinstance(g_m, float):
            return {
                'uni_parallel_multi': 0.0,
                'uni_perpendicular_multi': 0.0,
            }

dot_product = torch.dot(g_u.view(-1), g_m.view(-1))
        norm_m_squared = torch.norm(g_m) ** 2
        
        if dot_product < 0:
            g_u_parallel_m = g_m.clone()
        else:
            g_u_parallel_m = (dot_product / norm_m_squared) * g_m
        
        return {'uni_parallel_multi': g_u_parallel_m}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """第一步：施加扰动"""
        # 保存原始参数
        for group in self.param_groups:
            name = group['name']
            if name not in self.original_params:
                self.original_params[name] = {}
            for p in group['params']:
                if p.grad is not None:
                    self.original_params[name][p] = p.data.clone().detach()

        # 对各模态分别扰动
        self._perturb_specific_modality('modal1')
        self._perturb_specific_modality('modal2')
        # 对其他参数普通扰动
        self._perturb_other_params()

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def _perturb_specific_modality(self, modality_name):
        """MDPS：对指定模态用分解后的梯度扰动"""
        for group in self.param_groups:
            if group['name'] != modality_name:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                g_u = self.uni_gradients.get(modality_name, {}).get(p, 0.0)
                g_m = self.multi_gradients.get(modality_name, {}).get(p, 0.0)
                decomposed = self._get_decomposed_gradients(g_u, g_m)
                p.grad = decomposed['uni_parallel_multi'].clone()

        specific_norm = self._grad_specific_norm(modality_name)

        for group in self.param_groups:
            if group['name'] != modality_name:
                continue
            scale = group["rho"] / (specific_norm + self.perturb_eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                # APS核心：自适应扰动
                if group["adaptive"]:
                    e_w *= torch.pow(p, 2)
                p.data.add_(e_w)

    @torch.no_grad()
    def _perturb_other_params(self):
        """对非模态参数普通扰动"""
        total_norm = self._grad_norm()
        for group in self.param_groups:
            if group['name'] in ['modal1', 'modal2']:
                continue
            scale = group["rho"] / (total_norm + self.perturb_eps)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                if group["adaptive"]:
                    e_w *= torch.pow(p, 2)
                p.data.add_(e_w)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """第二步：恢复参数，更新"""
        for group in self.param_groups:
            name = group['name']
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data.copy_(self.original_params[name][p].to(p.device))

        self.base_optimizer.step()

        if zero_grad:
            self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """完整两步更新"""
        get_grad = closure if closure else self.forward_backward_func
        
        with torch.enable_grad():
            losses = get_grad()  # 第一次前向
        self.first_step(zero_grad=True)
        
        disable_running_stats(self.model)
        with torch.enable_grad():
            get_grad(only_multi=True)  # 第二次前向（扰动后）
        enable_running_stats(self.model)
        
        self.second_step(zero_grad=True)
        return losses[0]

    @torch.no_grad()
    def _grad_specific_norm(self, group_name):
        shared_device = self._first_param_device()
        if shared_device is None:
            return torch.tensor(0.0)
        for group in self.param_groups:
            if group["name"] == group_name:
                norms = [
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                    for p in group["params"]
                    if p.grad is not None
                ]
                if len(norms) == 0:
                    return torch.tensor(0.0, device=shared_device)
                return torch.norm(torch.stack(norms), p=2)
        return torch.tensor(0.0, device=shared_device)

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self._first_param_device()
        if shared_device is None:
            return torch.tensor(0.0)
        norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def _first_param_device(self):
        for group in self.param_groups:
            if len(group["params"]) > 0:
                return group["params"][0].device
        return None

