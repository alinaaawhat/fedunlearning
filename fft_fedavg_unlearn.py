from __future__ import annotations
import os
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import CNN_Cifar100
from dataset.generate_data import data_init
TARGET_LAYER_PATHS: Tuple[str, ...] = ('layer4.0.conv2', 'layer4.1.conv1', 'layer4.1.conv2', 'fc')

def _get_module_by_path(resnet: nn.Module, path: str) -> nn.Module:
    cur: nn.Module = resnet
    for part in path.split('.'):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur

def _set_module_by_path(resnet: nn.Module, path: str, module: nn.Module) -> None:
    parts = path.split('.')
    parent = resnet
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], module)

def allocate_supports(resnet: nn.Module, layer_paths: Sequence[str], num_clients: int, n_freq: int, seed: int, device: torch.device) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]]:
    rng = np.random.RandomState(seed)
    supports: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for path in layer_paths:
        mod = _get_module_by_path(resnet, path)
        if isinstance(mod, nn.Conv2d):
            o, i, kh, kw = mod.weight.shape
            h, w2 = (o, i * kh * kw)
        elif isinstance(mod, nn.Linear):
            h, w2 = mod.weight.shape
        else:
            raise TypeError(f'{path}: expected Conv2d or Linear, got {type(mod)}')
        total = h * w2
        need = num_clients * n_freq
        if need > total:
            raise ValueError(f'{path}: need {need} bins, have {total}')
        perm = rng.permutation(total)
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for c in range(num_clients):
            idx = perm[c * n_freq:(c + 1) * n_freq]
            u = torch.tensor(idx // w2, dtype=torch.long, device=device)
            v = torch.tensor(idx % w2, dtype=torch.long, device=device)
            pairs.append((u, v))
        supports[path] = pairs
    return supports

def spectral_grad_step(grad_2d: torch.Tensor, u: torch.Tensor, v: torch.Tensor, lr: float, ascent: bool) -> torch.Tensor:
    g = grad_2d.float()
    G = torch.fft.fft2(g.to(torch.complex64))
    dB = torch.zeros_like(G)
    sign = 1.0 if ascent else -1.0
    uu, vv = (u.to(device=G.device, dtype=torch.long), v.to(device=G.device, dtype=torch.long))
    dB[uu, vv] = sign * lr * G[uu, vv]
    dW = torch.fft.ifft2(dB).real
    return torch.fft.fft2(dW.to(torch.complex64))

class SpectralConv2dFedSOUL(nn.Module):

    def __init__(self, conv: nn.Conv2d, supports_list: List[Tuple[torch.Tensor, torch.Tensor]], fft_scale: float, num_clients: int):
        super().__init__()
        self.register_buffer('weight_base', conv.weight.data.clone())
        self.stride = conv.stride
        self.padding = conv.padding
        self.weight_shape = conv.weight.shape
        o, i, kh, kw = self.weight_shape
        self.h2d, self.w2d = (o, i * kh * kw)
        self.fft_scale = fft_scale
        self.num_clients = num_clients
        z = torch.zeros(self.h2d, self.w2d, dtype=torch.complex64)
        self.register_buffer('B_work', z.clone())
        self.register_buffer('B_cumulative', z.clone())
        for c in range(num_clients):
            u, v = supports_list[c]
            self.register_buffer(f'su_{c}', u.clone())
            self.register_buffer(f'sv_{c}', v.clone())
        self._w_eff_last: Optional[torch.Tensor] = None

    def reset_working(self) -> None:
        self.B_work.zero_()

    def commit_working_to_cumulative(self) -> None:
        self.B_cumulative.add_(self.B_work)
        self.B_work.zero_()

    def _delta_spatial(self, B: torch.Tensor) -> torch.Tensor:
        return (torch.fft.ifft2(B).real * self.fft_scale).view(self.weight_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Btot = self.B_cumulative + self.B_work
        w_eff = self.weight_base + self._delta_spatial(Btot)
        w_eff = w_eff.detach().clone().requires_grad_(True)
        self._w_eff_last = w_eff
        return F.conv2d(x, w_eff, stride=self.stride, padding=self.padding)

    def apply_grad_step(self, client_id: int, lr: float, ascent: bool, max_norm: Optional[float]) -> None:
        if self._w_eff_last is None or self._w_eff_last.grad is None:
            return
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_([self._w_eff_last], max_norm)
        g2d = self._w_eff_last.grad.reshape(self.h2d, self.w2d)
        u = getattr(self, f'su_{client_id}')
        v = getattr(self, f'sv_{client_id}')
        dB = spectral_grad_step(g2d, u, v, lr, ascent)
        self.B_work.add_(dB.to(self.B_work.dtype))

class SpectralLinearFedSOUL(nn.Module):

    def __init__(self, linear: nn.Linear, supports_list: List[Tuple[torch.Tensor, torch.Tensor]], fft_scale: float, num_clients: int):
        super().__init__()
        self.register_buffer('weight_base', linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer('_bias_buf', linear.bias.data.clone())
        else:
            self.register_buffer('_bias_buf', torch.zeros(1))
        self.has_bias = linear.bias is not None
        o, i = linear.weight.shape
        self.h2d, self.w2d = (o, i)
        self.fft_scale = fft_scale
        self.num_clients = num_clients
        z = torch.zeros(self.h2d, self.w2d, dtype=torch.complex64)
        self.register_buffer('B_work', z.clone())
        self.register_buffer('B_cumulative', z.clone())
        for c in range(num_clients):
            u, v = supports_list[c]
            self.register_buffer(f'su_{c}', u.clone())
            self.register_buffer(f'sv_{c}', v.clone())
        self._w_eff_last: Optional[torch.Tensor] = None

    def reset_working(self) -> None:
        self.B_work.zero_()

    def commit_working_to_cumulative(self) -> None:
        self.B_cumulative.add_(self.B_work)
        self.B_work.zero_()

    def _delta_spatial(self, B: torch.Tensor) -> torch.Tensor:
        return (torch.fft.ifft2(B).real * self.fft_scale).view(self.h2d, self.w2d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Btot = self.B_cumulative + self.B_work
        w_eff = self.weight_base + self._delta_spatial(Btot)
        w_eff = w_eff.detach().clone().requires_grad_(True)
        self._w_eff_last = w_eff
        b = self._bias_buf if self.has_bias else None
        return F.linear(x, w_eff, b)

    def apply_grad_step(self, client_id: int, lr: float, ascent: bool, max_norm: Optional[float]) -> None:
        if self._w_eff_last is None or self._w_eff_last.grad is None:
            return
        if max_norm is not None and max_norm > 0:
            torch.nn.utils.clip_grad_norm_([self._w_eff_last], max_norm)
        g2d = self._w_eff_last.grad.reshape(self.h2d, self.w2d)
        u = getattr(self, f'su_{client_id}')
        v = getattr(self, f'sv_{client_id}')
        dB = spectral_grad_step(g2d, u, v, lr, ascent)
        self.B_work.add_(dB.to(self.B_work.dtype))
SpectralModule = Union[SpectralConv2dFedSOUL, SpectralLinearFedSOUL]

def install_fedsoul_layers(resnet: nn.Module, supports: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]], fft_scale: float, num_clients: int) -> Dict[str, SpectralModule]:
    spectral: Dict[str, SpectralModule] = {}
    for path in TARGET_LAYER_PATHS:
        old = _get_module_by_path(resnet, path)
        pairs = supports[path]
        if isinstance(old, nn.Conv2d):
            new: SpectralModule = SpectralConv2dFedSOUL(old, pairs, fft_scale, num_clients)
        elif isinstance(old, nn.Linear):
            new = SpectralLinearFedSOUL(old, pairs, fft_scale, num_clients)
        else:
            raise TypeError(path)
        _set_module_by_path(resnet, path, new)
        spectral[path] = new
    return spectral

def _zero_spectral_working(spectral: Dict[str, SpectralModule]) -> None:
    for m in spectral.values():
        m.reset_working()

def stage1_retention_aggregation(net: CNN_Cifar100.Model, spectral: Dict[str, SpectralModule], client_loaders: List[DataLoader], num_clients: int, device: torch.device, args, criterion: nn.Module) -> None:
    B_server = {p: torch.zeros_like(spectral[p].B_work) for p in TARGET_LAYER_PATHS}
    use_cuda = device.type == 'cuda'
    for k in range(num_clients):
        _zero_spectral_working(spectral)
        for _ in range(args.stage1_epochs):
            for data, target in client_loaders[k]:
                data = data.to(device, non_blocking=use_cuda)
                target = target.to(device, non_blocking=use_cuda)
                net.zero_grad(set_to_none=True)
                loss = criterion(net(data), target)
                loss.backward()
                for mod in spectral.values():
                    mod.apply_grad_step(k, args.stage1_lr, ascent=False, max_norm=args.stage1_max_norm)
        for p in TARGET_LAYER_PATHS:
            B_server[p].add_(spectral[p].B_work)
    for p, mod in spectral.items():
        d_w = args.fft_scaling * torch.fft.ifft2(B_server[p]).real
        if isinstance(mod, SpectralConv2dFedSOUL):
            mod.weight_base.add_(d_w.view(mod.weight_shape))
        else:
            mod.weight_base.add_(d_w.view(mod.h2d, mod.w2d))
        mod.B_work.zero_()
        mod.B_cumulative.zero_()

def stage2_forget_chunk(net: CNN_Cifar100.Model, spectral: Dict[str, SpectralModule], client_loaders: List[DataLoader], forget_ids: List[int], device: torch.device, args, criterion: nn.Module) -> None:
    use_cuda = device.type == 'cuda'
    for c in forget_ids:
        _zero_spectral_working(spectral)
        for _ in range(args.stage2_epochs):
            for data, target in client_loaders[c]:
                data = data.to(device, non_blocking=use_cuda)
                target = target.to(device, non_blocking=use_cuda)
                net.zero_grad(set_to_none=True)
                loss = criterion(net(data), target)
                loss.backward()
                for mod in spectral.values():
                    mod.apply_grad_step(c, args.stage2_lr, ascent=True, max_norm=args.stage2_max_norm)
        for mod in spectral.values():
            mod.commit_working_to_cumulative()

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _enable_fast_cuda() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')

def _rebuild_client_loaders(client_loaders: List[DataLoader], device: torch.device) -> None:
    if device.type != 'cuda':
        return
    _cnw = min(4, os.cpu_count() or 2)
    for i, ld in enumerate(client_loaders):
        client_loaders[i] = DataLoader(ld.dataset, batch_size=ld.batch_size, shuffle=True, num_workers=_cnw, pin_memory=True, drop_last=getattr(ld, 'drop_last', False), persistent_workers=False)

def _rebuild_test_loaders(test_loaders: List[DataLoader], batch_size: int, device: torch.device) -> List[DataLoader]:
    if device.type != 'cuda':
        return test_loaders
    _cnw = min(4, os.cpu_count() or 2)
    out = []
    for ld in test_loaders:
        out.append(DataLoader(ld.dataset, batch_size=batch_size, shuffle=False, num_workers=_cnw, pin_memory=True, drop_last=getattr(ld, 'drop_last', False), persistent_workers=False))
    return out

@torch.no_grad()
def evaluate(model: nn.Module, loaders: Sequence[DataLoader], device: torch.device) -> float:
    if not loaders:
        return float('nan')
    model.eval()
    correct = total = 0
    use_cuda = device.type == 'cuda'
    for loader in loaders:
        for data, target in loader:
            data = data.to(device, non_blocking=use_cuda)
            target = target.to(device, non_blocking=use_cuda)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    return correct / total if total > 0 else 0.0

def get_args():
    p = argparse.ArgumentParser(description='FedSOUL: spectral continual unlearning (CIFAR-100 ResNet-18)')
    _base = os.path.dirname(os.path.abspath(__file__))
    _default_ckpt = os.path.join(_base, 'results', 'fedavg_cifar100_alpha0.01_le10.pth')
    p.add_argument('--ckpt_pth', type=str, default=_default_ckpt)
    p.add_argument('--data_name', default='cifar100', type=str)
    p.add_argument('--num_user', default=50, type=int, help='Number of FL clients (continual chunks use forget 1–3 then 4–6).')
    p.add_argument('--local_batch_size', default=128, type=int)
    p.add_argument('--test_batch_size', default=64, type=int)
    p.add_argument('--alpha', default=0.01, type=float)
    p.add_argument('--seed', default=50, type=int)
    p.add_argument('--partition', default='dir', type=str)
    p.add_argument('--niid', default=True, type=bool)
    p.add_argument('--balance', default=True, type=bool)
    p.add_argument('--proxy_frac', default=0.2, type=float)
    p.add_argument('--forget_paradigm', default='client', type=str)
    p.add_argument('--forget_client_idx', type=list, default=[])
    p.add_argument('--forget_class_idx', type=list, default=[])
    p.add_argument('--n_freq_per_client', type=int, default=1500)
    p.add_argument('--fft_scaling', type=float, default=100.0)
    p.add_argument('--support_seed', type=int, default=42, help='RNG seed for disjoint supports.')
    p.add_argument('--stage1_lr', type=float, default=0.0001, help='Stage I retention spectral step lr.')
    p.add_argument('--stage1_epochs', type=int, default=1, help='Local epochs per client in Stage I.')
    p.add_argument('--stage1_max_norm', type=float, default=1.0)
    p.add_argument('--stage2_lr', type=float, default=0.001, help='Stage II forgetting spectral step lr (ascent).')
    p.add_argument('--stage2_epochs', type=int, default=5)
    p.add_argument('--stage2_max_norm', type=float, default=1.0)
    p.add_argument('--save_csv', type=str, default='./results/fft_fedavg_unlearn_two_phase.csv')
    p.add_argument('--save_pth', type=str, default='./results/fft_fedavg_unlearn_two_phase.pth')
    p.add_argument('--save_params_jsonl', type=str, default='')
    return p.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = 'cuda' if device.type == 'cuda' else 'cpu'
    args.num_classes = 100
    if device.type == 'cuda':
        _enable_fast_cuda()
    print(f'device: {device}')
    nu = args.num_user
    if nu < 7:
        raise SystemExit('--num_user must be >= 7 for clients 1–6 split')
    phase1_forget = [1, 2, 3]
    phase2_forget = [4, 5, 6]
    phase1_remain = [i for i in range(nu) if i not in phase1_forget]
    phase2_remain = [i for i in range(nu) if i not in phase1_forget + phase2_forget]
    phase2_unlearn_all = phase1_forget + phase2_forget
    if not os.path.isfile(args.ckpt_pth):
        raise FileNotFoundError(args.ckpt_pth)
    client_loaders, test_loaders, _, _ = data_init(args)
    assert len(client_loaders) == nu
    _rebuild_client_loaders(client_loaders, device)
    test_loaders = _rebuild_test_loaders(test_loaders, args.test_batch_size, device)
    net = CNN_Cifar100.Model(args)
    try:
        sd = torch.load(args.ckpt_pth, map_location='cpu', weights_only=True)
    except TypeError:
        sd = torch.load(args.ckpt_pth, map_location='cpu')
    _inq = net.load_state_dict(sd, strict=True)
    if _inq is not None and (getattr(_inq, 'missing_keys', ()) or getattr(_inq, 'unexpected_keys', ())):
        raise RuntimeError(f'load_state_dict mismatch: {_inq}')
    print(f'Loaded FedAvg base weights from: {args.ckpt_pth}', flush=True)
    supports = allocate_supports(net.model, TARGET_LAYER_PATHS, nu, args.n_freq_per_client, args.support_seed, device)
    spectral = install_fedsoul_layers(net.model, supports, args.fft_scaling, nu)
    net.to(device)
    for p in net.parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    rows = []
    t_total = time.time()
    t0 = time.time()
    print(f'FedSOUL Stage I: retention init, all {nu} clients (spectral_descent + aggregate)')
    stage1_retention_aggregation(net, spectral, client_loaders, nu, device, args, criterion)
    print('Stage I applied global spatial update to weight_base.', flush=True)
    print(f'FedSOUL Stage II-A: forget {phase1_forget} (spectral_ascent, cumulative B)')
    stage2_forget_chunk(net, spectral, client_loaders, phase1_forget, device, args, criterion)
    t1 = time.time() - t0
    remain_tr_lds = [client_loaders[i] for i in phase1_remain]
    remain_te_lds = [test_loaders[i] for i in phase1_remain]
    unlearn_tr_lds = [client_loaders[i] for i in phase1_forget]
    unlearn_te_lds = [test_loaders[i] for i in phase1_forget]
    rows.append({'n_freq_per_client': args.n_freq_per_client, 'fft_scaling': args.fft_scaling, 'phase': 1, 'unlearn_spec': '1-3', 'remain_train_acc': round(evaluate(net, remain_tr_lds, device), 6), 'remain_test_acc': round(evaluate(net, remain_te_lds, device), 6), 'unlearn_train_acc': round(evaluate(net, unlearn_tr_lds, device), 6), 'unlearn_test_acc': round(evaluate(net, unlearn_te_lds, device), 6), 'retain_protection': 0, 'time_s': round(t1, 2)})
    print('Phase 1 metrics:', rows[-1])
    t0 = time.time()
    print(f'FedSOUL Stage II-B: forget {phase2_forget}')
    stage2_forget_chunk(net, spectral, client_loaders, phase2_forget, device, args, criterion)
    t2 = time.time() - t0
    remain_tr_lds2 = [client_loaders[i] for i in phase2_remain]
    remain_te_lds2 = [test_loaders[i] for i in phase2_remain]
    unlearn_tr_lds2 = [client_loaders[i] for i in phase2_unlearn_all]
    unlearn_te_lds2 = [test_loaders[i] for i in phase2_unlearn_all]
    rows.append({'n_freq_per_client': args.n_freq_per_client, 'fft_scaling': args.fft_scaling, 'phase': 2, 'unlearn_spec': '1-6', 'remain_train_acc': round(evaluate(net, remain_tr_lds2, device), 6), 'remain_test_acc': round(evaluate(net, remain_te_lds2, device), 6), 'unlearn_train_acc': round(evaluate(net, unlearn_tr_lds2, device), 6), 'unlearn_test_acc': round(evaluate(net, unlearn_te_lds2, device), 6), 'retain_protection': 0, 'time_s': round(t2, 2)})
    print('Phase 2 metrics:', rows[-1])
    df = pd.DataFrame(rows)
    df['total_time_s'] = round(time.time() - t_total, 2)
    os.makedirs(os.path.dirname(args.save_csv) or '.', exist_ok=True)
    _csv_exists = os.path.isfile(args.save_csv) and os.path.getsize(args.save_csv) > 0
    df.to_csv(args.save_csv, mode='a' if _csv_exists else 'w', header=not _csv_exists, index=False)
    os.makedirs(os.path.dirname(args.save_pth) or '.', exist_ok=True)
    torch.save(net.state_dict(), args.save_pth)
    print(f'{('Appended to' if _csv_exists else 'Wrote')} {args.save_csv} | {args.save_pth}')
    if args.save_params_jsonl:
        payload = {'n_freq_per_client': args.n_freq_per_client, 'fft_scaling': args.fft_scaling, 'ckpt_pth': args.ckpt_pth, 'num_user': args.num_user, 'alpha': args.alpha, 'seed': args.seed, 'stage1_lr': args.stage1_lr, 'stage1_epochs': args.stage1_epochs, 'stage1_max_norm': args.stage1_max_norm, 'stage2_lr': args.stage2_lr, 'stage2_epochs': args.stage2_epochs, 'stage2_max_norm': args.stage2_max_norm, 'support_seed': args.support_seed, 'save_csv': args.save_csv, 'save_pth': args.save_pth, 'total_time_s': round(time.time() - t_total, 2), 'phase_results': rows}
        os.makedirs(os.path.dirname(args.save_params_jsonl) or '.', exist_ok=True)
        with open(args.save_params_jsonl, 'a', encoding='utf-8') as jf:
            json.dump(payload, jf, ensure_ascii=False)
            jf.write('\n')
        print(f'Appended run record to {args.save_params_jsonl}')
if __name__ == '__main__':
    main()
