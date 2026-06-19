#!/usr/bin/env python3
"""
Real-time stereo matching with SuperPoint + LightGlue on Orin NX.
Reads USB stereo camera, extracts features, matches, displays results.
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import cv2
import numpy as np

# Add model definitions from validate_model.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate_model import SuperPoint, LightGlue


# =========================================================================
# Configuration
# =========================================================================

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_KEYPOINTS = 300       # max features per image
CONFIDENCE_THRESH = 0.002  # keypoint confidence threshold
MATCH_THRESHOLD = 0.001    # minimum match confidence
NMS_RADIUS = 4            # non-max suppression radius
CAM_WIDTH = 1280
CAM_HEIGHT = 480
FPS_TARGET = 30


# =========================================================================
# Model Loading
# =========================================================================

def load_models(device='cuda'):
    """Load SuperPoint and LightGlue from saved weights."""
    print("[1] Loading models...")
    
    # SuperPoint
    sp = SuperPoint().to(device).eval()
    sp_path = os.path.join(MODEL_DIR, 'superpoint_v1.pth')
    if not os.path.exists(sp_path):
        raise FileNotFoundError(f"Missing {sp_path}")
    sp_sd = torch.load(sp_path, map_location=device)
    sp.load_state_dict(sp_sd, strict=True)
    print(f"  SuperPoint: loaded ({len(sp_sd)} tensors)")
    
    # LightGlue
    lg = LightGlue(dim=256, num_layers=9, num_heads=4).to(device).eval()
    lg_path = os.path.join(MODEL_DIR, 'superpoint_lightglue.pth')
    if not os.path.exists(lg_path):
        raise FileNotFoundError(f"Missing {lg_path}")
    lg_sd = torch.load(lg_path, map_location=device)
    
    # Map keys (self_attn.0 → transformers.0.self_attn)
    mapped = {}
    for k, v in lg_sd.items():
        parts = k.split('.')
        if parts[0] == 'self_attn':
            k2 = f"transformers.{parts[1]}.self_attn." + '.'.join(parts[2:])
        elif parts[0] == 'cross_attn':
            k2 = f"transformers.{parts[1]}.cross_attn." + '.'.join(parts[2:])
        else:
            k2 = k
        mapped[k2] = v
    lg.load_state_dict(mapped, strict=False)
    print(f"  LightGlue: loaded ({len(lg_sd)} tensors)")
    
    return sp, lg


# =========================================================================
# Calibration Loading
# =========================================================================

def load_yaml(path):
    """Parse OpenCV YAML calibration files."""
    import re
    with open(path) as f:
        text = f.read()
    text = re.sub(r'^%YAML.*\n', '', text)
    text = re.sub(r'^---\n', '', text)
    
    result = {}
    in_matrix = False
    mat_key = None
    rows = cols = 0
    data_vals = []
    
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if not in_matrix:
            m = re.match(r'(\w+):\s*!!opencv-matrix', s)
            if m:
                mat_key = m.group(1)
                in_matrix = True
                rows = cols = 0
                data_vals = []
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
                data_vals.extend(_parse_data(s[5:].strip()))
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
                data_vals.extend(_parse_data(s))
    
    if mat_key and data_vals:
        result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
    return result


def _parse_data(s):
    s = s.replace('...', '').replace('[', '').replace(']', '')
    vals = []
    for part in s.split(','):
        part = part.strip()
        if part:
            try: vals.append(float(part))
            except ValueError: pass
    return vals


def load_calibration():
    """Load stereo calibration and compute rectification maps."""
    print("[2] Loading calibration...")
    
    left = load_yaml(os.path.join(CALIB_DIR, 'left_intrinsics.yaml'))
    right = load_yaml(os.path.join(CALIB_DIR, 'right_intrinsics.yaml'))
    rectify = load_yaml(os.path.join(CALIB_DIR, 'stereo_rectify.yaml'))
    
    K1, D1 = left['camera_matrix'], left['distortion_coefficients']
    K2, D2 = right['camera_matrix'], right['distortion_coefficients']
    R1, R2 = rectify['rectification_matrix_left'], rectify['rectification_matrix_right']
    P1, P2 = rectify['projection_matrix_left'], rectify['projection_matrix_right']
    
    size = (640, 480)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    
    print(f"  Calibration loaded (baseline from stereo_extrinsics)")
    return map1x, map1y, map2x, map2y


# =========================================================================
# Camera
# =========================================================================

def open_camera(device=0):
    """Open USB stereo camera."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise IOError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)
    # Warm up
    for _ in range(5):
        cap.read()
    return cap


# =========================================================================
# SuperPoint Inference
# =========================================================================

def superpoint_inference(model, gray_tensor, device):
    """
    Run SuperPoint and extract keypoints + descriptors.
    
    Args:
        model: SuperPoint model
        gray_tensor: (1, 1, H, W) normalized to [0,1], on device
    Returns:
        kpts: (N, 2) keypoint coordinates (x, y)
        desc: (N, 256) descriptors
        scores: (N,) confidence scores
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            semi, desc_dense = model(gray_tensor)  # (1,65,Hc,Wc), (1,256,Hc,Wc)
    
    Hc, Wc = semi.shape[2:]
    
    # Keypoint extraction
    scores_soft = F.softmax(semi, dim=1)               # (1,65,Hc,Wc)
    scores_grid = scores_soft[0, :-1, :, :]             # (64,Hc,Wc) - remove dustbin
    max_scores, max_idx = scores_grid.max(dim=0)        # (Hc,Wc) each
    
    # NMS
    nms_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (NMS_RADIUS, NMS_RADIUS))
    nms_mask = (max_scores.cpu().numpy() >= CONFIDENCE_THRESH).astype(np.uint8)
    nms_dilated = cv2.dilate(nms_mask, nms_kernel)
    
    # Keypoint candidates
    ys, xs = np.where((nms_mask == 1) & (nms_dilated == nms_mask))
    
    # Limit keypoints by confidence
    confs = max_scores[ys, xs].cpu().numpy()
    order = np.argsort(-confs)[:MAX_KEYPOINTS]
    ys, xs, confs = ys[order], xs[order], confs[order]
    
    if len(xs) == 0:
        return np.zeros((0, 2)), np.zeros((0, 256)), np.zeros((0,))
    
    # Map to original resolution (each cell is 8x8 pixels)
    cell_indices = max_idx[ys, xs].cpu().numpy()  # which of the 64 sub-cells
    cx, cy = cell_indices % 8, cell_indices // 8
    kpx = (xs * 8 + cx).astype(np.float32)
    kpy = (ys * 8 + cy).astype(np.float32)
    kpts = np.stack([kpx, kpy], axis=1)
    
    # Sample descriptors at keypoint locations (bilinear interpolation)
    # desc_dense shape: (1, 256, Hc, Wc)
    desc = desc_dense[0]  # (256, Hc, Wc)
    # Grid sample coordinates [-1, 1]
    grid_x = xs.astype(np.float32) / max(Wc - 1, 1) * 2 - 1
    grid_y = ys.astype(np.float32) / max(Hc - 1, 1) * 2 - 1
    grid = torch.from_numpy(np.stack([grid_x, grid_y], axis=1)).unsqueeze(0).unsqueeze(0)  # (1,1,N,2)
    grid = grid.to(device)
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            sampled = F.grid_sample(
            desc.unsqueeze(0), grid, mode='bilinear', align_corners=True
        )  # (1,256,1,N)
    sampled_desc = sampled[0, :, 0, :].T.cpu().numpy()  # (N,256)
    
    # Normalize descriptors
    sampled_desc = sampled_desc / (np.linalg.norm(sampled_desc, axis=1, keepdims=True) + 1e-12)
    
    return kpts, sampled_desc, confs


# =========================================================================
# LightGlue Inference
# =========================================================================

def lightglue_inference(model, desc0, kpts0, desc1, kpts1, device):
    """
    Match keypoints between two images using LightGlue.
    
    Returns:
        matches: (M, 2) matched pairs as [idx0, idx1]
        match_conf: (M,) confidence scores
    """
    if len(kpts0) < 2 or len(kpts1) < 2:
        return np.zeros((0, 2), dtype=int), np.zeros((0,))
    
    # Prepare tensors
    desc0_t = torch.from_numpy(desc0).unsqueeze(0).to(device)
    desc1_t = torch.from_numpy(desc1).unsqueeze(0).to(device)
    
    # Normalize keypoints to [-1, 1]
    kpts0_t = torch.from_numpy(kpts0).unsqueeze(0).to(device)
    kpts1_t = torch.from_numpy(kpts1).unsqueeze(0).to(device)
    kpts0_norm = kpts0_t / torch.tensor([320., 240.], device=device) * 2 - 1
    kpts1_norm = kpts1_t / torch.tensor([320., 240.], device=device) * 2 - 1
    
    # Run LightGlue
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            matches_idx, scores = model(desc0_t, kpts0_norm, desc1_t, kpts1_norm)
    
    matches_idx = matches_idx[0].cpu().numpy()  # (N0,)
    scores = scores[0].cpu().numpy()             # (N0,)
    
    # Filter valid matches
    valid = (matches_idx >= 0) & (scores >= MATCH_THRESHOLD)
    src_idx = np.where(valid)[0]
    dst_idx = matches_idx[valid].astype(int)
    match_conf = scores[valid]
    
    # Epipolar filter: for rectified images, y-coordinates should match
    if len(src_idx) > 0:
        y_diff = np.abs(kpts0[src_idx, 1] - kpts1[dst_idx, 1])
        epipolar_ok = y_diff < 5.0  # max 5px vertical disparity (rectified)
        src_idx = src_idx[epipolar_ok]
        dst_idx = dst_idx[epipolar_ok]
        match_conf = match_conf[epipolar_ok]
    
    # Debug: print score stats periodically
    if np.random.random() < 0.01:
        print(f"    Match stats: scores min={scores.min():.6f} max={scores.max():.6f} "
              f"mean={scores.mean():.6f}, matched_before_filter={np.sum(matches_idx>=0)}")
    
    matches = np.stack([src_idx, dst_idx], axis=1)
    return matches, match_conf


# =========================================================================
# Visualization
# =========================================================================

def draw_matches(left_img, right_img, kpts0, kpts1, matches, match_conf, fps, n_kpts):
    """
    Draw matching result between left and right images.
    """
    h, w = left_img.shape[:2]
    display = np.hstack([left_img, right_img])
    
    # Draw matches with depth-colored lines
    for i, (i0, i1) in enumerate(matches):
        pt0 = (int(kpts0[i0, 0]), int(kpts0[i0, 1]))
        pt1 = (int(kpts1[i1, 0]) + w, int(kpts1[i1, 1]))
        disp = abs(kpts0[i0, 0] - kpts1[i1, 0])
        # Color: red (close) → blue (far)
        color = (min(255, int(disp * 2)), 0, max(0, 255 - int(disp * 2)))
        cv2.line(display, pt0, pt1, color, 1, cv2.LINE_AA)
        cv2.circle(display, pt0, 2, (0, 255, 0), -1)
        cv2.circle(display, pt1, 2, (0, 255, 0), -1)
    
    # Overlay info
    num_matches = len(matches)
    disparities = kpts0[matches[:, 0], 0] - kpts1[matches[:, 1], 0] if num_matches > 0 else []
    mean_disp = np.mean(disparities) if len(disparities) > 0 else 0
    info = [
        f"FPS: {fps:.1f}",
        f"Keypoints: L={n_kpts[0]}  R={n_kpts[1]}",
        f"Matches: {num_matches}  Disp: {mean_disp:.1f}px",
        f"q: quit  s: save",
    ]
    for i, txt in enumerate(info):
        cv2.putText(display, txt, (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Draw separator
    cv2.line(display, (w, 0), (w, h), (255, 255, 255), 2)
    
    return display


# =========================================================================
# Main Loop
# =========================================================================

def main():
    print("=" * 60)
    print("  Real-Time Stereo Matching: SuperPoint + LightGlue")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n  Device: {device}")
    
    # Load models
    sp, lg = load_models(device)
    
    # Load calibration
    map1x, map1y, map2x, map2y = load_calibration()
    
    # Open camera
    print("\n[3] Opening camera...")
    cap = open_camera()
    print(f"  Camera ready: {CAM_WIDTH}x{CAM_HEIGHT}")
    
    # Create windows
    cv2.namedWindow('Stereo Matching', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Stereo Matching', 1280, 480)
    
    # Main loop
    print("\n[4] Running (press 'q' to quit, 's' to save pair)...\n")
    
    frame_count = 0
    fps_timer = time.time()
    save_idx = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed")
                break
            
            frame_count += 1
            t0 = time.time()
            
            # Split stereo
            h, w = frame.shape[:2]
            left_bgr = frame[:, :w//2]
            right_bgr = frame[:, w//2:]
            
            # Rectify
            left_rect = cv2.remap(left_bgr, map1x, map1y, cv2.INTER_LINEAR)
            right_rect = cv2.remap(right_bgr, map2x, map2y, cv2.INTER_LINEAR)
            
            # Convert to grayscale + normalize for SuperPoint
            left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            
            # To tensor
            left_t = torch.from_numpy(left_gray).unsqueeze(0).unsqueeze(0).to(device)
            right_t = torch.from_numpy(right_gray).unsqueeze(0).unsqueeze(0).to(device)
            
            # SuperPoint inference
            kpts0, desc0, _ = superpoint_inference(sp, left_t, device)
            kpts1, desc1, _ = superpoint_inference(sp, right_t, device)
            
            # LightGlue matching
            matches, match_conf = lightglue_inference(lg, desc0, kpts0, desc1, kpts1, device)
            
            # FPS
            t1 = time.time()
            elapsed = t1 - t0
            display_fps = frame_count / (t1 - fps_timer) if frame_count % 30 == 0 else \
                          frame_count / (t1 - fps_timer)
            
            # Draw
            display = draw_matches(
                left_rect, right_rect, kpts0, kpts1,
                matches, match_conf, display_fps,
                (len(kpts0), len(kpts1))
            )
            cv2.imshow('Stereo Matching', display)
            
            # Periodic status
            if frame_count % 30 == 0:
                avg_fps = frame_count / (time.time() - fps_timer)
                print(f"  Frame {frame_count} | FPS: {avg_fps:.1f} | "
                      f"Kpts: {len(kpts0)}/{len(kpts1)} | Matches: {len(matches)}")
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("  User quit")
                break
            elif key == ord('s'):
                save_idx += 1
                ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'match_{ts}_left.png', left_rect)
                cv2.imwrite(f'match_{ts}_right.png', right_rect)
                cv2.imwrite(f'match_{ts}_result.png', display)
                print(f"  Saved match #{save_idx} ({ts})")
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    # Summary
    duration = time.time() - fps_timer
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Average FPS: {frame_count / max(duration, 0.01):.1f}")
    print(f"  Saved: {save_idx} pairs")
    print("✅ Done!")


if __name__ == '__main__':
    main()
