#!/usr/bin/env python3
"""
Real-time stereo matching: SuperPoint (TensorRT) + LightGlue (PyTorch).
First run:  exports ONNX + builds TRT engine (~3 min).
Later runs: loads cached engine, runs live matching.

Usage:
  python3 stereo_matching_onnx.py          # first run: export + build + match
  python3 stereo_matching_onnx.py --build  # force rebuild engine
"""

import os, sys, time, re, argparse, pickle
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# TensorRT
import tensorrt as trt

# Model defs
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_model import SuperPoint, LightGlue

# =========================================================================
# Config
# =========================================================================
CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(MODEL_DIR, 'superpoint.onnx')
ENGINE_PATH = os.path.join(MODEL_DIR, 'superpoint.engine')
CACHE_PATH = os.path.join(MODEL_DIR, 'trt_cache.pkl')

MAX_KP = 300
CONF_THRESH = 0.002
MATCH_THRESH = 0.001
NMS_RADIUS = 4
CAM_W = 1280; CAM_H = 480

trt_logger = trt.Logger(trt.Logger.WARNING)


# =========================================================================
# Step 1: Export ONNX
# =========================================================================

def export_onnx():
    """Export SuperPoint to ONNX with dynamic H/W."""
    print("\n[Export] Exporting SuperPoint to ONNX...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sp = SuperPoint().eval().to(device)
    sp.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'superpoint_v1.pth'), map_location=device))
    
    dummy = torch.randn(1, 1, 480, 640, device=device)
    
    # Export with dynamic axes for variable input size
    torch.onnx.export(
        sp, dummy, ONNX_PATH,
        input_names=['input'],
        output_names=['scores', 'descriptors'],
        dynamic_axes={
            'input': {2: 'H', 3: 'W'},
            'scores': {2: 'Hc', 3: 'Wc'},
            'descriptors': {2: 'Hc', 3: 'Wc'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  ONNX saved: {ONNX_PATH}")
    
    # Verify
    import onnx
    model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(model)
    print(f"  ONNX verified: {len(model.graph.node)} ops")


# =========================================================================
# Step 2: Build TensorRT Engine
# =========================================================================

def build_engine(force=False):
    """Build TensorRT engine from ONNX (FP32)."""
    if os.path.exists(ENGINE_PATH) and not force:
        print(f"\n[TRT] Engine exists: {ENGINE_PATH}")
        print("  Use --build to force rebuild")
        return
    
    print("\n[TRT] Building TensorRT engine (FP32)...")
    print("  This takes ~2-3 minutes on Orin NX...")
    
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, trt_logger)
    
    # Parse ONNX
    with open(ONNX_PATH, 'rb') as f:
        if not parser.parse(f.read()):
            for err in range(parser.num_errors):
                print(f"  ONNX error: {parser.get_error(err)}")
            raise RuntimeError("ONNX parse failed")
    
    # Config - use FP32 for accuracy
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Profile for dynamic shapes
    profile = builder.create_optimization_profile()
    profile.set_shape('input', (1, 1, 240, 320), (1, 1, 480, 640), (1, 1, 960, 1280))
    config.add_optimization_profile(profile)
    
    # Build
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Engine build failed")
    
    with open(ENGINE_PATH, 'wb') as f:
        f.write(serialized)
    
    print(f"  Engine built in {time.time()-t0:.0f}s: {ENGINE_PATH}")
    print(f"  Size: {os.path.getsize(ENGINE_PATH)/1024/1024:.0f} MB")


# =========================================================================
# Step 3: TensorRT Inference Wrapper
# =========================================================================

class SuperPointTRT:
    """SuperPoint inference via TensorRT."""
    
    def __init__(self, engine_path):
        print(f"\n[TRT] Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        
        # Get tensor names
        self.input_name = self.engine.get_tensor_name(0)
        self.score_name = self.engine.get_tensor_name(1)
        self.desc_name = self.engine.get_tensor_name(2)
        
        # Pre-allocate output buffers (max size)
        self.max_h, self.max_w = 480, 640
        self._alloc_buffers(self.max_h, self.max_w)
        
        print(f"  Engine loaded: {os.path.getsize(engine_path)/1024/1024:.0f} MB")
    
    def _alloc_buffers(self, H, W):
        """Allocate torch CUDA buffers for given input size."""
        Hc, Wc = H // 8, W // 8
        # TRT expects FP32 input
        self.input_buf = torch.empty((1, 1, H, W), dtype=torch.float32, device='cuda')
        self.score_buf = torch.empty((1, 65, Hc, Wc), dtype=torch.float32, device='cuda')
        self.desc_buf = torch.empty((1, 256, Hc, Wc), dtype=torch.float32, device='cuda')
    
    def infer(self, gray_tensor):
        """
        Run TRT inference.
        gray_tensor: (1, 1, H, W) float32 on CUDA, values [0,1]
        Returns: scores (1,65,Hc,Wc), descriptors (1,256,Hc,Wc) as float32
        """
        H, W = gray_tensor.shape[2:]
        Hc, Wc = H // 8, W // 8
        
        # Re-allocate if size changed
        if H != self.max_h or W != self.max_w:
            self.max_h, self.max_w = H, W
            self._alloc_buffers(H, W)
        
        # TRT input
        self.input_buf[:] = gray_tensor
        
        # Set dynamic shapes
        self.context.set_input_shape(self.input_name, (1, 1, H, W))
        
        # Bind tensor addresses
        self.context.set_tensor_address(self.input_name, self.input_buf.data_ptr())
        self.context.set_tensor_address(self.score_name, self.score_buf.data_ptr())
        self.context.set_tensor_address(self.desc_name, self.desc_buf.data_ptr())
        
        # Run
        stream = torch.cuda.current_stream()
        self.context.execute_async_v3(stream.cuda_stream)
        stream.synchronize()
        
        # Return as float32 (matching original PyTorch behavior)
        return self.score_buf[:, :, :Hc, :Wc], self.desc_buf[:, :, :Hc, :Wc]


# =========================================================================
# Keypoint Extraction (GPU)
# =========================================================================

def extract_keypoints_gpu(scores, desc_dense, max_kp=MAX_KP, conf_thresh=CONF_THRESH):
    """
    Extract keypoints from SuperPoint score map, fully on GPU.
    
    scores: (1, 65, Hc, Wc) float32 CUDA
    desc_dense: (1, 256, Hc, Wc) float32 CUDA
    Returns: kpts_np (N,2), desc_np (N,256), scores_np (N,)
    """
    scores_soft = F.softmax(scores, dim=1)           # (1,65,Hc,Wc)
    scores_grid = scores_soft[0, :-1]                 # (64,Hc,Wc)
    max_scores, max_idx = scores_grid.max(dim=0)      # (Hc,Wc)
    Hc, Wc = max_scores.shape
    
    # Threshold
    mask = max_scores >= conf_thresh                   # (Hc,Wc) bool
    
    # NMS via maxpool
    pad = NMS_RADIUS // 2
    pool = F.max_pool2d(max_scores.unsqueeze(0).unsqueeze(0),
                         kernel_size=NMS_RADIUS, stride=1, padding=pad)
    pool = pool[:, :, :Hc, :Wc]
    nms_mask = (max_scores == pool[0,0]) & mask
    
    # Get keypoint coordinates
    indices = torch.nonzero(nms_mask)                  # (N, 2) -> [y, x]
    if indices.shape[0] == 0:
        return np.zeros((0,)), np.zeros((0,)), np.zeros((0,256)), np.zeros((0,))
    
    ys, xs = indices[:, 0], indices[:, 1]
    confs = max_scores[ys, xs]
    
    # Sort by confidence, limit
    order = torch.argsort(-confs)[:max_kp]
    ys, xs, confs = ys[order], xs[order], confs[order]
    N = len(ys)
    
    # Map to original resolution (superpoint grid: 8x8 pixel cells)
    cell_ids = max_idx[ys, xs]  # which of 64 sub-cells
    cx = cell_ids % 8
    cy = cell_ids // 8
    
    kpx = (xs.float() * 8 + cx.float()).cpu().numpy()
    kpy = (ys.float() * 8 + cy.float()).cpu().numpy()
    
    # Sample descriptors via grid_sample (GPU)
    grid_x = xs.float() / max(scores.shape[2] - 1, 1) * 2 - 1
    grid_y = ys.float() / max(scores.shape[3] - 1, 1) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=1).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
    
    sampled = F.grid_sample(desc_dense, grid, mode='bilinear', align_corners=True)
    desc = sampled[0, :, 0, :].T  # (N, 256)
    desc = F.normalize(desc, p=2, dim=1)
    
    return kpx, kpy, desc.cpu().numpy(), confs.cpu().numpy()


# =========================================================================
# Calibration
# =========================================================================

def load_yaml(path):
    with open(path) as f:
        text = f.read()
    text = re.sub(r'^%YAML.*\n', '', text); text = re.sub(r'^---\n', '', text)
    result = {}; in_matrix = False; mat_key = None; rows = cols = 0; data_vals = []
    for line in text.splitlines():
        s = line.strip()
        if not s: continue
        if not in_matrix:
            m = re.match(r'(\w+):\s*!!opencv-matrix', s)
            if m:
                mat_key = m.group(1); in_matrix = True; rows = cols = 0; data_vals = []
                continue
            m = re.match(r'(\w+):\s*(.+)', s)
            if m:
                k, v = m.group(1), m.group(2)
                try: result[k] = int(v)
                except ValueError:
                    try: result[k] = float(v)
                    except ValueError: result[k] = v.strip('"\'')
        else:
            if s.startswith('rows:'): rows = int(s.split(':')[1])
            elif s.startswith('cols:'): cols = int(s.split(':')[1])
            elif s.startswith('dt:'): pass
            elif s.startswith('data:'):
                vals = s[5:].strip().replace('...','').replace('[','').replace(']','')
                data_vals.extend([float(x) for x in vals.split(',') if x.strip()])
            elif re.match(r'\w+:\s*!!opencv-matrix', s):
                if mat_key and data_vals:
                    result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
                m = re.match(r'(\w+):\s*!!opencv-matrix', s)
                mat_key = m.group(1); rows = cols = 0; data_vals = []
            elif re.match(r'\w+:\s', s) and '!!opencv-matrix' not in s:
                if mat_key and data_vals:
                    result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
                in_matrix = False; mat_key = None
                m = re.match(r'(\w+):\s*(.+)', s)
                if m:
                    k, v = m.group(1), m.group(2)
                    try: result[k] = int(v)
                    except ValueError:
                        try: result[k] = float(v)
                        except ValueError: result[k] = v.strip('"\'')
            else:
                vals = s.replace('...','').replace('[','').replace(']','')
                data_vals.extend([float(x) for x in vals.split(',') if x.strip()])
    if mat_key and data_vals:
        result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
    return result


def load_calibration():
    left = load_yaml(os.path.join(CALIB_DIR, 'left_intrinsics.yaml'))
    right = load_yaml(os.path.join(CALIB_DIR, 'right_intrinsics.yaml'))
    rect = load_yaml(os.path.join(CALIB_DIR, 'stereo_rectify.yaml'))
    sz = (640, 480)
    m1x, m1y = cv2.initUndistortRectifyMap(
        left['camera_matrix'], left['distortion_coefficients'],
        rect['rectification_matrix_left'], rect['projection_matrix_left'], sz, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(
        right['camera_matrix'], right['distortion_coefficients'],
        rect['rectification_matrix_right'], rect['projection_matrix_right'], sz, cv2.CV_32FC1)
    return m1x, m1y, m2x, m2y


# =========================================================================
# LightGlue Matching
# =========================================================================

def match_lightglue(model, desc0, kpts0, desc1, kpts1, device):
    """LightGlue matching (PyTorch)."""
    if len(kpts0) < 2 or len(kpts1) < 2:
        return np.zeros((0,2), dtype=int), np.zeros((0,))
    
    d0 = torch.from_numpy(desc0).unsqueeze(0).to(device)
    d1 = torch.from_numpy(desc1).unsqueeze(0).to(device)
    k0 = torch.from_numpy(kpts0).unsqueeze(0).to(device)
    k1 = torch.from_numpy(kpts1).unsqueeze(0).to(device)
    k0n = k0 / torch.tensor([320., 240.], device=device) * 2 - 1
    k1n = k1 / torch.tensor([320., 240.], device=device) * 2 - 1
    
    with torch.no_grad():
        matches_idx, scores = model(d0, k0n, d1, k1n)
    
    matches_idx = matches_idx[0].cpu().numpy()
    scores = scores[0].cpu().numpy()
    
    valid = (matches_idx >= 0) & (scores >= MATCH_THRESH)
    si = np.where(valid)[0]
    di = matches_idx[valid].astype(int)
    sc = scores[valid]
    
    if len(si) == 0:
        return np.zeros((0,2), dtype=int), np.zeros((0,))
    if len(si) > 0 and len(kpts0) > 0 and len(kpts1) > 0:
        yd = np.abs(kpts0[si, 1] - kpts1[di, 1])
        ok = yd < 5.0
        si, di, sc = si[ok], di[ok], sc[ok]
    
    return np.stack([si, di], axis=1), sc


# =========================================================================
# Display
# =========================================================================

def draw(left, right, k0, k1, matches, scores, fps, nk):
    h, w = left.shape[:2]
    disp = np.hstack([left, right])
    for (i0, i1), conf in zip(matches, scores):
        p0 = (int(k0[i0,0]), int(k0[i0,1]))
        p1 = (int(k1[i1,0]) + w, int(k1[i1,1]))
        d = abs(k0[i0,0] - k1[i1,0])
        c = (min(255,int(d*2)), 0, max(0,255-int(d*2)))
        cv2.line(disp, p0, p1, c, 1, cv2.LINE_AA)
        cv2.circle(disp, p0, 2, (0,255,0), -1)
        cv2.circle(disp, p1, 2, (0,255,0), -1)
    md = np.mean([abs(k0[m[0],0]-k1[m[1],0]) for m in matches]) if len(matches) > 0 else 0
    for i, t in enumerate([
        f"FPS: {fps:.1f}  |  TRT SuperPoint + PyTorch LightGlue",
        f"Kpts: L={nk[0]} R={nk[1]}  |  Matches: {len(matches)}  Disp: {md:.1f}px",
        "[q] quit  [s] save",
    ]):
        cv2.putText(disp, t, (10, 25+i*22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.line(disp, (w,0), (w,h), (255,255,255), 2)
    return disp


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='Force rebuild TRT engine')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Step 1: Export ONNX (if needed)
    if not os.path.exists(ONNX_PATH):
        export_onnx()
    
    # Step 2: Build TRT engine (if needed)
    if not os.path.exists(ENGINE_PATH) or args.build:
        build_engine(force=args.build)
    
    # Step 3: Load TRT SuperPoint
    print("\n[Load] Loading TensorRT SuperPoint...")
    trt_sp = SuperPointTRT(ENGINE_PATH)
    
    # Step 4: Load PyTorch LightGlue
    print("[Load] Loading LightGlue (PyTorch)...")
    lg = LightGlue(dim=256, num_layers=9, num_heads=4).to(device).eval()
    sd = torch.load(os.path.join(MODEL_DIR, 'superpoint_lightglue.pth'), map_location=device)
    mapped = {}
    for k, v in sd.items():
        p = k.split('.')
        if p[0] == 'self_attn':    k2 = f"transformers.{p[1]}.self_attn." + '.'.join(p[2:])
        elif p[0] == 'cross_attn': k2 = f"transformers.{p[1]}.cross_attn." + '.'.join(p[2:])
        else:                      k2 = k
        mapped[k2] = v
    lg.load_state_dict(mapped, strict=False)
    print("  LightGlue loaded")
    
    # Step 5: Calibration
    print("[Load] Calibration...")
    m1x, m1y, m2x, m2y = load_calibration()
    
    # Step 6: Camera
    print("[Load] Camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(5): cap.read()
    
    cv2.namedWindow('Stereo Matching (TRT)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Matching (TRT)', 1280, 480)
    
    # Step 7: Live loop
    print("\n[Run] Live matching (press 'q' to quit)...\n")
    fc = 0; save_idx = 0; fps_timer = time.time(); fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            fc += 1
            t0 = time.time()
            
            # Split + rectify
            h, w = frame.shape[:2]
            left_bgr = frame[:, :w//2]; right_bgr = frame[:, w//2:]
            left_rect = cv2.remap(left_bgr, m1x, m1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_bgr, m2x, m2y, cv2.INTER_LINEAR)
            
            # To GPU grayscale
            left_g = torch.from_numpy(
                cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0).cuda()
            right_g = torch.from_numpy(
                cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            ).unsqueeze(0).unsqueeze(0).cuda()
            
            # === TRT SuperPoint ===
            scores0, desc0 = trt_sp.infer(left_g)
            scores1, desc1 = trt_sp.infer(right_g)
            
            # === Keypoint extraction (GPU) ===
            kx0, ky0, d0, c0 = extract_keypoints_gpu(scores0, desc0)
            kx1, ky1, d1, c1 = extract_keypoints_gpu(scores1, desc1)
            kpts0 = np.stack([kx0, ky0], axis=1) if len(kx0) > 0 else np.zeros((0,2))
            kpts1 = np.stack([kx1, ky1], axis=1) if len(kx1) > 0 else np.zeros((0,2))
            
            # === LightGlue ===
            matches, match_scores = match_lightglue(lg, d0, kpts0, d1, kpts1, device)
            
            # FPS
            if fc % 5 == 0:
                fps = fc / (time.time() - fps_timer)
            
            # Display
            display = draw(left_rect, right_rect, kpts0, kpts1, matches, match_scores, fps, (len(kpts0), len(kpts1)))
            cv2.imshow('Stereo Matching (TRT)', display)
            
            if fc % 30 == 0:
                print(f"  Frame {fc} | FPS: {fps:.1f} | Kpts: {len(kpts0)}/{len(kpts1)} | Matches: {len(matches)}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            if key == ord('s'):
                save_idx += 1
                ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'trt_match_{ts}_left.png', left_rect)
                cv2.imwrite(f'trt_match_{ts}_right.png', right_rect)
                cv2.imwrite(f'trt_match_{ts}_result.png', display)
    
    except KeyboardInterrupt: pass
    finally:
        cap.release(); cv2.destroyAllWindows()
    
    duration = time.time() - fps_timer
    print(f"\nFrames: {fc} | Avg FPS: {fc/duration:.1f} | Duration: {duration:.0f}s")
    print("✅ Done!")


if __name__ == '__main__':
    main()
