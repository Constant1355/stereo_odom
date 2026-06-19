#!/usr/bin/env python3
"""Quick profile: SuperPoint vs LightGlue latency (no camera, no display)."""
import os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_model import SuperPoint, LightGlue

device = 'cuda'
torch.cuda.synchronize()
torch.backends.cudnn.benchmark = True

sp = SuperPoint().to(device).eval()
sp.load_state_dict(torch.load('superpoint_v1.pth', map_location=device))
sp = torch.compile(sp, mode='reduce-overhead')

sd = torch.load('superpoint_lightglue.pth', map_location=device)
lg = LightGlue(dim=256, num_layers=9, num_heads=4).to(device).eval()
mapped = {}
for k, v in sd.items():
    p = k.split('.')
    k2 = f"transformers.{p[1]}.self_attn." + '.'.join(p[2:]) if p[0]=='self_attn' else \
         f"transformers.{p[1]}.cross_attn." + '.'.join(p[2:]) if p[0]=='cross_attn' else k
    mapped[k2] = v
lg.load_state_dict(mapped, strict=False)

dummy = torch.randn(2, 1, 480, 640, device=device)
# Warm up (more for compiled model)
for _ in range(5):
    with torch.no_grad(): sp(dummy)
torch.cuda.synchronize()

N = 100
t0 = time.time()
for _ in range(N):
    with torch.no_grad(): sp(dummy)
torch.cuda.synchronize()
print(f'SuperPoint compiled (batch=2): {(time.time()-t0)/N*1000:.0f}ms')

# Test without compile for comparison
sp2 = SuperPoint().to(device).eval()
sp2.load_state_dict(torch.load('superpoint_v1.pth', map_location=device))
for _ in range(3):
    with torch.no_grad(): sp2(dummy)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(N):
    with torch.no_grad(): sp2(dummy)
torch.cuda.synchronize()
print(f'SuperPoint no-compile (batch=2): {(time.time()-t0)/N*1000:.0f}ms')

for k in [50, 100, 200, 300]:
    d0 = torch.randn(1, k, 256, device=device)
    d1 = torch.randn(1, k, 256, device=device)
    k0 = torch.rand(1, k, 2, device=device) * 2 - 1
    k1 = torch.rand(1, k, 2, device=device) * 2 - 1
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        with torch.no_grad(): lg(d0, k0, d1, k1)
    torch.cuda.synchronize()
    print(f'LightGlue (k={k:3d}): {(time.time()-t0)/50*1000:.0f}ms')
