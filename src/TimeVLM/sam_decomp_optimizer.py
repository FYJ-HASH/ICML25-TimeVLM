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
    APS + MDPS 融合优化器（三分支版：vision / text / temporal）
    """
    def __init__(self, params, base_optimizer, model, rho=0.05,
                 adaptive=True, perturb_eps=1e-12, **kwargs):
        for group in params:
            group.setdefault("rho", rho)
            group.setdefault("name", "other")

        defaults = dict(adaptive=adaptive, **kwargs)
        super(SAMDecompOptimizer, self).__init__(params, defaults)

        self.model = model
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.perturb_eps = perturb_eps

        self.original_params = {}
        self.multi_gradients = {}
        self.uni_gradients = {}
        self.forward_backward_func = None

        # === APS：三个分支各自的扰动范数 ===
        self.aps_history = {'vision': [], 'text': [], 'temporal': []}
        self._perturb_norm_sq = {}
        self._current_aps_norm = {}

        # === MDPS：三个分支各自的融合/单模态梯度范数 ===
        self.mdps_history = {
            'vision_multi_norm': [], 'vision_uni_norm': [],
            'text_multi_norm': [], 'text_uni_norm': [],
            'temporal_multi_norm': [], 'temporal_uni_norm': [],
        }

        self.last_losses = {'fusion': 0.0, 'temporal': 0.0, 'multimodal': 0.0}

        def set_closure(self, loss_fn, inputs, targets):
        self.multi_gradients = {}
        self.uni_gradients = {}

        def get_grad(only_multi=False):
            self.base_optimizer.zero_grad()
            outputs = self.model(*inputs)
            loss_fusion, loss_temporal, loss_vision, loss_text = loss_fn(outputs, targets)

            self.last_losses = {
                'fusion': loss_fusion.item(),
                'temporal': loss_temporal.item(),
                'vision': loss_vision.item(),
                'text': loss_text.item()
            }

            if not only_multi:
                # Step 1: Fusion loss → multi_gradients (all three branches)
                loss_fusion.backward(retain_graph=True)
                self.multi_gradients['vision'] = self._store_module_gradients('modal1')
                self.multi_gradients['text'] = self._store_module_gradients('modal2')
                self.multi_gradients['temporal'] = self._store_module_gradients('modal3')
                self.base_optimizer.zero_grad()

                # Step 2: Temporal-only loss → uni_gradients['temporal']
                loss_temporal.backward(retain_graph=True)
                self.uni_gradients['temporal'] = self._store_module_gradients('modal3')
                self.base_optimizer.zero_grad()

                # Step 3: Vision-only loss → uni_gradients['vision']
                loss_vision.backward(retain_graph=True)
                self.uni_gradients['vision'] = self._store_module_gradients('modal1')
                self.base_optimizer.zero_grad()

                # Step 4: Text-only loss → uni_gradients['text']
                loss_text.backward(retain_graph=True)
                self.uni_gradients['text'] = self._store_module_gradients('modal2')
                self.base_optimizer.zero_grad()

            total_loss = loss_fusion + loss_temporal + loss_vision + loss_text
            total_loss.backward()
            return total_loss.item(), loss_temporal.item(), loss_vision.item(), loss_text.item()

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
        if isinstance(g_u, float) or isinstance(g_m, float):
            return {'uni_parallel_multi': 0.0}
        norm_m_squared = torch.norm(g_m) ** 2
        if norm_m_squared < 1e-12:  # 除零保护
            return {'uni_parallel_multi': 0.0}
        dot_product = torch.dot(g_u.view(-1), g_m.view(-1))
        if dot_product < 0:
            g_u_parallel_m = g_m.clone()
        else:
            g_u_parallel_m = (dot_product / norm_m_squared) * g_m
        return {'uni_parallel_multi': g_u_parallel_m}

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self._perturb_norm_sq = {'modal1': 0.0, 'modal2': 0.0, 'modal3': 0.0}

        for group in self.param_groups:
            name = group['name']
            if name not in self.original_params:
                self.original_params[name] = {}
            for p in group['params']:
                if p.grad is not None:
                    self.original_params[name][p] = p.data.clone().detach()

        self._perturb_specific_modality('modal1')
        self._perturb_specific_modality('modal2')
        self._perturb_specific_modality('modal3')
        self._perturb_other_params()

        self._current_aps_norm = {
            'vision': self._perturb_norm_sq['modal1'] ** 0.5,
            'text': self._perturb_norm_sq['modal2'] ** 0.5,
            'temporal': self._perturb_norm_sq['modal3'] ** 0.5,
        }

        if zero_grad:
            self.base_optimizer.zero_grad()

    @torch.no_grad()
    def _perturb_specific_modality(self, modality_name):
        branch_key = {'modal1': 'vision', 'modal2': 'text', 'modal3': 'temporal'}[modality_name]

        for group in self.param_groups:
            if group['name'] != modality_name:
                continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                g_u = self.uni_gradients.get(branch_key, {}).get(p, 0.0)
                g_m = self.multi_gradients.get(branch_key, {}).get(p, 0.0)
                decomposed = self._get_decomposed_gradients(g_u, g_m)
                if isinstance(decomposed['uni_parallel_multi'], float):
                    continue
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
                if group["adaptive"]:
                    e_w *= torch.pow(p, 2)
                p.data.add_(e_w)
                self._perturb_norm_sq[modality_name] += e_w.norm().item() ** 2

    @torch.no_grad()
    def _perturb_other_params(self):
        total_norm = self._grad_norm()
        for group in self.param_groups:
            if group['name'] in ['modal1', 'modal2', 'modal3']:
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
        for group in self.param_groups:
            name = group['name']
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p not in self.original_params.get(name, {}):
                    continue
                p.data.copy_(self.original_params[name][p].to(p.device))
        self.base_optimizer.step()
        if zero_grad:
            self.base_optimizer.zero_grad()

    def step(self, closure=None):
        get_grad = closure if closure else self.forward_backward_func

        with torch.enable_grad():
            losses = get_grad()

        self.mdps_history['vision_multi_norm'].append(self._grad_specific_norm('modal1').item())
        self.mdps_history['vision_uni_norm'].append(self._compute_uni_grad_norm('vision'))
        self.mdps_history['text_multi_norm'].append(self._grad_specific_norm('modal2').item())
        self.mdps_history['text_uni_norm'].append(self._compute_uni_grad_norm('text'))
        self.mdps_history['temporal_multi_norm'].append(self._grad_specific_norm('modal3').item())
        self.mdps_history['temporal_uni_norm'].append(self._compute_uni_grad_norm('temporal'))

        self.first_step(zero_grad=True)

        self.aps_history['vision'].append(self._current_aps_norm['vision'])
        self.aps_history['text'].append(self._current_aps_norm['text'])
        self.aps_history['temporal'].append(self._current_aps_norm['temporal'])

        disable_running_stats(self.model)
        with torch.enable_grad():
            get_grad(only_multi=True)
        enable_running_stats(self.model)

        self.second_step(zero_grad=True)
        return losses[0]

    def _compute_uni_grad_norm(self, branch_key):
        gradients = self.uni_gradients.get(branch_key, {})
        if not gradients:
            return 0.0
        norms = [g.norm().item() for g in gradients.values()]
        return sum(n ** 2 for n in norms) ** 0.5

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

