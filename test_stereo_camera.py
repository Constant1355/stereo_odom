#!/usr/bin/env python3
"""
Stereo camera test on Orin NX.
Reads USB stereo camera (1280x480 → 2× 640x480), loads calibration,
displays rectified video, saves sample.
"""

import os
import sys
import cv2
import numpy as np
import time

CALIB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calibration')

# =========================================================================
# Load calibration
# =========================================================================

def read_opencv_matrix(node):
    """Parse OpenCV YAML matrix node."""
    data = node['data']
    rows = node['rows']
    cols = node['cols']
    return np.array(data, dtype=np.float64).reshape(rows, cols)


def load_yaml(path):
    """Robust parser for OpenCV YAML calibration files."""
    import re
    
    with open(path) as f:
        text = f.read()
    
    # Remove header and separator
    text = re.sub(r'^%YAML.*\n', '', text)
    text = re.sub(r'^---\n', '', text)
    
    result = {}
    buf = ''
    in_matrix = False
    mat_key = None
    rows = cols = 0
    data_vals = []
    
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        
        if not in_matrix:
            m = re.match(r'(\w+):\s*!!opencv-matrix', stripped)
            if m:
                mat_key = m.group(1)
                in_matrix = True
                rows = cols = 0
                data_vals = []
                continue
            
            # Plain key-value
            m = re.match(r'(\w+):\s*(.+)', stripped)
            if m:
                k, v = m.group(1), m.group(2)
                try:
                    result[k] = int(v)
                except ValueError:
                    try:
                        result[k] = float(v)
                    except ValueError:
                        result[k] = v.strip('"\'')
        else:
            # Inside a matrix block
            if stripped.startswith('rows:'):
                rows = int(stripped.split(':')[1])
            elif stripped.startswith('cols:'):
                cols = int(stripped.split(':')[1])
            elif stripped.startswith('dt:'):
                pass
            elif stripped.startswith('data:'):
                buf = stripped[5:].strip()
                vals = _parse_data(buf)
                data_vals.extend(vals)
            elif re.match(r'\w+:\s*!!opencv-matrix', stripped):
                # Next matrix starts - save current
                if mat_key and data_vals:
                    result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
                # Start new matrix
                m = re.match(r'(\w+):\s*!!opencv-matrix', stripped)
                mat_key = m.group(1)
                rows = cols = 0
                data_vals = []
            elif re.match(r'\w+:\s', stripped) and '!!opencv-matrix' not in stripped:
                # New key-value after matrix - save matrix first
                if mat_key and data_vals:
                    result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
                in_matrix = False
                mat_key = None
                # Process this line as plain kv
                m = re.match(r'(\w+):\s*(.+)', stripped)
                if m:
                    k, v = m.group(1), m.group(2)
                    try:
                        result[k] = int(v)
                    except ValueError:
                        try:
                            result[k] = float(v)
                        except ValueError:
                            result[k] = v.strip('"\'')
            else:
                # Data continuation line
                vals = _parse_data(stripped)
                data_vals.extend(vals)
    
    # Save last matrix if any
    if mat_key and data_vals:
        result[mat_key] = np.array(data_vals, dtype=np.float64).reshape(rows, cols)
    
    return result


def _parse_data(s):
    """Extract float values from '[1.0, 2.0 ...]' string."""
    s = s.replace('...', '').replace('[', '').replace(']', '')
    vals = []
    for part in s.split(','):
        part = part.strip()
        if part:
            try:
                vals.append(float(part))
            except ValueError:
                pass
    return vals


def load_calibration():
    """Load all stereo calibration parameters."""
    calib = {}
    
    # Intrinsics
    left = load_yaml(os.path.join(CALIB_DIR, 'left_intrinsics.yaml'))
    right = load_yaml(os.path.join(CALIB_DIR, 'right_intrinsics.yaml'))
    calib['K1'] = left['camera_matrix']
    calib['D1'] = left['distortion_coefficients']
    calib['K2'] = right['camera_matrix']
    calib['D2'] = right['distortion_coefficients']
    
    # Rectification
    rectify = load_yaml(os.path.join(CALIB_DIR, 'stereo_rectify.yaml'))
    calib['R1'] = rectify['rectification_matrix_left']
    calib['R2'] = rectify['rectification_matrix_right']
    calib['P1'] = rectify['projection_matrix_left']
    calib['P2'] = rectify['projection_matrix_right']
    
    # Extrinsics
    extr = load_yaml(os.path.join(CALIB_DIR, 'stereo_extrinsics.yaml'))
    calib['T'] = extr['translation_vector_mm']
    calib['R'] = extr['rotation_matrix']
    
    calib['image_size'] = (640, 480)
    
    return calib


# =========================================================================
# Camera
# =========================================================================

def open_camera(device=0, width=1280, height=480, fps=30):
    """Open USB stereo camera."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise IOError(f"Cannot open camera /dev/video{device}")
    
    # Set properties
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Read one frame to verify
    ret, frame = cap.read()
    if not ret:
        raise IOError("Cannot read from camera")
    
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Camera opened: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS")
    
    return cap


def split_stereo(frame):
    """Split 1280x480 frame into left (640x480) and right (640x480)."""
    h, w = frame.shape[:2]
    mid = w // 2
    left = frame[:, :mid]
    right = frame[:, mid:]
    return left, right


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("  Stereo Camera Test")
    print("=" * 60)
    
    # Load calibration
    print("\n[1] Loading calibration...")
    try:
        calib = load_calibration()
        print(f"  Left camera matrix:\n{calib['K1']}")
        print(f"  Right camera matrix:\n{calib['K2']}")
        print(f"  Baseline: {float(calib['T'].flatten()[0]):.1f} mm")
        print(f"  Image size: {calib['image_size']}")
    except Exception as e:
        print(f"  ⚠️  Cannot load calibration: {e}")
        import traceback; traceback.print_exc()
        calib = None
    
    # Open camera
    print("\n[2] Opening camera...")
    try:
        cap = open_camera(device=0)
    except Exception as e:
        print(f"  ❌ {e}")
        sys.exit(1)
    
    # Prepare rectification maps if calibration available
    map1x, map1y, map2x, map2y = None, None, None, None
    if calib:
        print("\n[3] Computing rectification maps...")
        size = calib['image_size']
        map1x, map1y = cv2.initUndistortRectifyMap(
            calib['K1'], calib['D1'], calib['R1'], calib['P1'], size, cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(
            calib['K2'], calib['D2'], calib['R2'], calib['P2'], size, cv2.CV_32FC1)
        print("  Done")
    
    # Capture loop
    print("\n[4] Capturing 100 frames for validation...")
    print(f"  Running headless - samples saved to stereo_samples/")
    print(f"  Press Ctrl+C in terminal to stop early\n")
    
    os.makedirs('stereo_samples', exist_ok=True)
    frame_count = 0
    saved_count = 0
    save_interval = 10  # save every 10th frame
    max_frames = 100
    t_start = time.time()
    
    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                print("  [WARN] Failed to grab frame")
                break
            
            frame_count += 1
            left, right = split_stereo(frame)
            
            # Rectify
            if map1x is not None:
                rect_left = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
                rect_right = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
            else:
                rect_left, rect_right = left, right
            
            # Save sample frames
            if frame_count % save_interval == 0:
                saved_count += 1
                ts = time.strftime('%Y%m%d_%H%M%S')
                cv2.imwrite(f'stereo_samples/{ts}_frame{frame_count:04d}_raw.png', frame)
                cv2.imwrite(f'stereo_samples/{ts}_frame{frame_count:04d}_left.png', rect_left)
                cv2.imwrite(f'stereo_samples/{ts}_frame{frame_count:04d}_right.png', rect_right)
            
            # Progress
            fps = frame_count / (time.time() - t_start)
            if frame_count % 10 == 0:
                print(f"  Frame {frame_count}/{max_frames} | FPS: {fps:.1f}", end='\r')
    
    except KeyboardInterrupt:
        print("\n  Interrupted")
    
    finally:
        cap.release()
    
    # Summary
    duration = time.time() - t_start
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total frames: {frame_count}")
    print(f"  Duration: {duration:.1f} s")
    print(f"  Average FPS: {frame_count / duration:.1f}")
    print(f"  Samples saved: {saved_count}")
    print(f"  Camera: {'✅ Working' if frame_count > 0 else '❌ Failed'}")
    print("✅ Done!")


if __name__ == '__main__':
    main()
