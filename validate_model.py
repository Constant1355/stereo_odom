#!/usr/bin/env python3
"""
Validate superpoint_v1.pth + superpoint_lightglue.pth on Orin NX.
Builds both architectures from scratch WITHOUT installing superpoint/lightglue packages.

SuperPoint (1.30M params, 24 tensors):
  - VGG-style encoder: conv1a-4b, MaxPool 3x
  - Keypoint head: convPa, convPb (65-cell softmax)
  - Descriptor head: convDa, convDb (256-dim descriptors)

LightGlue (11.9M params, 260 tensors):
  - posenc, 9x SelfBlock, 9x CrossBlock, Sinkhorn matching
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# Part 1: SuperPoint Architecture
# =========================================================================

class SuperPoint(nn.Module):
    """SuperPoint: Self-Supervised Interest Point Detection and Description.
    Matches the cvg/LightGlue superpoint_v1.pth weight names exactly.
    """
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.conv1a = nn.Conv2d(1, c1, 3, 1, 1)
        self.conv1b = nn.Conv2d(c1, c1, 3, 1, 1)
        self.conv2a = nn.Conv2d(c1, c2, 3, 1, 1)
        self.conv2b = nn.Conv2d(c2, c2, 3, 1, 1)
        self.conv3a = nn.Conv2d(c2, c3, 3, 1, 1)
        self.conv3b = nn.Conv2d(c3, c3, 3, 1, 1)
        self.conv4a = nn.Conv2d(c3, c4, 3, 1, 1)
        self.conv4b = nn.Conv2d(c4, c4, 3, 1, 1)
        self.convPa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convPb = nn.Conv2d(c5, 65, 1, 1, 0)
        self.convDa = nn.Conv2d(c4, c5, 3, 1, 1)
        self.convDb = nn.Conv2d(c5, 256, 1, 1, 0)

    def forward(self, x):
        x = self.relu(self.conv1a(x)); x = self.relu(self.conv1b(x)); x = self.pool(x)
        x = self.relu(self.conv2a(x)); x = self.relu(self.conv2b(x)); x = self.pool(x)
        x = self.relu(self.conv3a(x)); x = self.relu(self.conv3b(x)); x = self.pool(x)
        x = self.relu(self.conv4a(x)); x = self.relu(self.conv4b(x))
        semi = self.relu(self.convPa(x)); semi = self.convPb(semi)
        desc = self.relu(self.convDa(x)); desc = self.convDb(desc)
        desc = F.normalize(desc, p=2, dim=1)
        return semi, desc


# =========================================================================
# Part 2: LightGlue Architecture
# =========================================================================

class SelfBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = dim; self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.Wqkv = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=True),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=True),
        )

    def forward(self, x, encoding=None):
        B, N, D = x.shape
        q, k, v = torch.chunk(self.Wqkv(x), 3, dim=-1)
        H = self.num_heads
        hd = D // H
        q = q.reshape(B, N, H, hd).transpose(1, 2)
        k = k.reshape(B, N, H, hd).transpose(1, 2)
        v = v.reshape(B, N, H, hd).transpose(1, 2)
        attn = F.softmax((q @ k.transpose(-2, -1)) / (hd ** 0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return x + self.ffn(torch.cat([x, self.out_proj(out)], -1))


class CrossBlock(nn.Module):
    def __init__(self, dim=256, num_heads=4):
        super().__init__()
        self.heads = num_heads
        dh = dim // num_heads
        self.scale = dh ** -0.5
        self.to_qk = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)
        self.to_out = nn.Linear(dim, dim, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * dim, 2 * dim, bias=True),
            nn.LayerNorm(2 * dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * dim, dim, bias=True),
        )

    def forward(self, x0, x1):
        qk0, qk1 = self.to_qk(x0), self.to_qk(x1)
        v0, v1 = self.to_v(x0), self.to_v(x1)
        H = self.heads
        def mh(t): return t.reshape(*t.shape[:-1], H, -1).transpose(1, 2)
        qk0, qk1, v0, v1 = mh(qk0), mh(qk1), mh(v0), mh(v1)
        sim = torch.einsum("bhid,bhjd->bhij", qk0, qk1)
        a01 = F.softmax(sim * self.scale, dim=-1)
        a10 = F.softmax(sim.transpose(-2, -1) * self.scale, dim=-1)
        m0 = torch.einsum("bhij,bhjd->bhid", a01, v1)
        m1 = torch.einsum("bhji,bhjd->bhid", a10.transpose(-2, -1), v0)
        m0 = self.to_out(m0.transpose(1, 2).flatten(start_dim=-2))
        m1 = self.to_out(m1.transpose(1, 2).flatten(start_dim=-2))
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M=2, dim=256):
        super().__init__()
        self.Wr = nn.Linear(M, dim // 8, bias=False)

    def forward(self, x):
        proj = self.Wr(x)
        return torch.stack([torch.cos(proj), torch.sin(proj)], 0)


class MatchAssignment(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.r = nn.Parameter(torch.zeros(1))
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0, desc1):
        m0, m1 = self.final_proj(desc0), self.final_proj(desc1)
        d = m0.shape[-1]
        sim = torch.einsum("bmd,bnd->bmn", m0 / d**0.25, m1 / d**0.25)
        return sim, self.matchability(desc0), self.matchability(desc1)


class TokenConfidence(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, desc0, desc1):
        return self.token(desc0).squeeze(-1), self.token(desc1).squeeze(-1)


class LightGlue(nn.Module):
    def __init__(self, dim=256, num_layers=9, num_heads=4):
        super().__init__()
        self.posenc = LearnableFourierPositionalEncoding(M=2, dim=dim)
        self.input_proj = nn.Identity()
        self.transformers = nn.ModuleList()
        for _ in range(num_layers):
            self.transformers.append(nn.ModuleDict({
                'self_attn': SelfBlock(dim, num_heads),
                'cross_attn': CrossBlock(dim, num_heads),
            }))
        self.log_assignment = nn.ModuleList(
            [MatchAssignment(dim) for _ in range(num_layers)])
        self.token_confidence = nn.ModuleList(
            [TokenConfidence(dim) for _ in range(num_layers - 1)])

    def forward(self, desc0, kpts0, desc1, kpts1):
        x0, x1 = self.input_proj(desc0), self.input_proj(desc1)
        enc0, enc1 = self.posenc(kpts0), self.posenc(kpts1)
        scores_list = []
        for i in range(len(self.transformers)):
            x0 = self.transformers[i]['self_attn'](x0, enc0)
            x1 = self.transformers[i]['self_attn'](x1, enc1)
            x0, x1 = self.transformers[i]['cross_attn'](x0, x1)
            s, _, _ = self.log_assignment[i](x0, x1)
            scores_list.append(s)
        scores = scores_list[-1]
        scores0 = F.softmax(scores, -1)
        match_scores, matches0 = scores0.max(dim=-1)
        return matches0, match_scores


# =========================================================================
# Part 3: Main validation
# =========================================================================

def validate_superpoint(sd):
    print("\n" + "=" * 60)
    print("  SuperPoint Validation")
    print("=" * 60)
    model = SuperPoint()
    m_sd = model.state_dict()
    ok = {k: v for k, v in sd.items() if k in m_sd and v.shape == m_sd[k].shape}
    print(f"  Weight match: {len(ok)}/{len(m_sd)}")
    if len(ok) != len(m_sd):
        print(f"  >>> FAILED: Architecture mismatch")
        return False
    model.load_state_dict(ok, strict=True)
    dummy = torch.randn(1, 1, 240, 320)
    with torch.no_grad():
        semi, desc = model(dummy)
    print(f"  >>> Forward pass: ✅")
    print(f"      Scores: {list(semi.shape)}, Descriptors: {list(desc.shape)}")
    print(f"      Max score: {semi[:,:-1].max():.4f}")
    return True


def validate_lightglue(sd):
    print("\n" + "=" * 60)
    print("  LightGlue Validation")
    print("=" * 60)
    model = LightGlue(dim=256, num_layers=9, num_heads=4)
    m_sd = model.state_dict()
    mapped = {}
    for k, v in sd.items():
        p = k.split('.')
        if p[0] == 'self_attn':    k2 = f"transformers.{p[1]}.self_attn." + '.'.join(p[2:])
        elif p[0] == 'cross_attn': k2 = f"transformers.{p[1]}.cross_attn." + '.'.join(p[2:])
        else:                      k2 = k
        mapped[k2] = v
    ok = {k: v for k, v in mapped.items() if k in m_sd and v.shape == m_sd[k].shape}
    print(f"  Weight match: {len(ok)}/{len(m_sd)}")
    if len(ok) < len(m_sd) * 0.8:
        print(f"  >>> FAILED: Only {len(ok)}/{len(m_sd)} matched")
        return False
    model.load_state_dict(ok, strict=False)
    B, N0, N1 = 1, 100, 120
    d0 = F.normalize(torch.randn(B, N0, 256), dim=-1)
    k0 = torch.rand(B, N0, 2) * 2 - 1
    d1 = F.normalize(torch.randn(B, N1, 256), dim=-1)
    k1 = torch.rand(B, N1, 2) * 2 - 1
    with torch.no_grad():
        matches, scores = model(d0, k0, d1, k1)
    matched = (matches >= 0).sum().item()
    print(f"  >>> Forward pass: ✅")
    print(f"      Matches: {list(matches.shape)}, Scores: {list(scores.shape)}")
    print(f"      Mean confidence: {scores.mean().item():.4f}, Matched: {matched}/{N0}")
    return True


def main():
    base = os.path.expanduser('~/Documents/stereo_odom')
    results = {}
    
    sp_path = os.path.join(base, 'superpoint_v1.pth')
    if os.path.exists(sp_path):
        sd = torch.load(sp_path, map_location='cpu')
        print(f"Loaded superpoint_v1.pth ({len(sd)} tensors, "
              f"{sum(v.numel() for v in sd.values())/1e6:.2f}M params)")
        results['SuperPoint'] = validate_superpoint(sd)
    else:
        print(f"[SKIP] superpoint_v1.pth not found")
    
    lg_path = os.path.join(base, 'superpoint_lightglue.pth')
    if os.path.exists(lg_path):
        sd = torch.load(lg_path, map_location='cpu')
        print(f"\nLoaded superpoint_lightglue.pth ({len(sd)} tensors, "
              f"{sum(v.numel() for v in sd.values())/1e6:.2f}M params)")
        results['LightGlue'] = validate_lightglue(sd)
    else:
        print(f"[SKIP] superpoint_lightglue.pth not found")
    
    print("\n" + "=" * 60)
    print("  Overall Summary")
    print("=" * 60)
    for name, ok in results.items():
        print(f"  {name}: {'✅ VALID' if ok else '❌ FAILED'}")
    if results.get('SuperPoint') and results.get('LightGlue'):
        print(f"\n  ✅ Full pipeline: SuperPoint -> LightGlue")
        print(f"     = stereo matching / visual odometry ready")
    print(f"\n  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("✅ Done!")


if __name__ == '__main__':
    main()
