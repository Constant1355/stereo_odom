"""Headless benchmark for stereo matching pipeline."""
import sys, os, time, cv2, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stereo_matching import load_models, load_calibration, open_camera, superpoint_inference_batch, lightglue_inference

N_WARMUP = 10
N_BENCH = 50

device = 'cuda'
torch.backends.cudnn.benchmark = True
sp, lg = load_models(device)
m1x,m1y,m2x,m2y = load_calibration()
cap = open_camera()

# Warmup (discard these frames)
for _ in range(N_WARMUP):
    ret, frame = cap.read()
    if not ret: break
    h,w = frame.shape[:2]
    left = cv2.remap(frame[:,:w//2], m1x, m1y, cv2.INTER_LINEAR)
    right = cv2.remap(frame[:,w//2:], m2x, m2y, cv2.INTER_LINEAR)
    lgray = torch.from_numpy(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).cuda()
    rgray = torch.from_numpy(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).cuda()
    k0,d0,k1,d1 = superpoint_inference_batch(sp, lgray, rgray, device)
    matches, _ = lightglue_inference(lg, d0, k0, d1, k1, device)

torch.cuda.synchronize()

sp_tot, lg_tot, fc = 0, 0, 0
for _ in range(N_BENCH):
    ret, frame = cap.read()
    if not ret: break
    fc += 1
    h,w = frame.shape[:2]
    left = cv2.remap(frame[:,:w//2], m1x, m1y, cv2.INTER_LINEAR)
    right = cv2.remap(frame[:,w//2:], m2x, m2y, cv2.INTER_LINEAR)
    lgray = torch.from_numpy(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).cuda()
    rgray = torch.from_numpy(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY).astype(np.float32)/255).unsqueeze(0).unsqueeze(0).cuda()
    
    t0 = time.time()
    k0,d0,k1,d1 = superpoint_inference_batch(sp, lgray, rgray, device)
    torch.cuda.synchronize()
    sp_tot += time.time() - t0
    t0 = time.time()
    matches, _ = lightglue_inference(lg, d0, k0, d1, k1, device)
    torch.cuda.synchronize()
    lg_tot += time.time() - t0

cap.release()
print(f'Frames: {fc} (after {N_WARMUP} warmup)')
print(f'SP avg: {sp_tot/fc*1000:.0f}ms  LG avg: {lg_tot/fc*1000:.0f}ms')
print(f'Pipeline: {(sp_tot+lg_tot)/fc*1000:.0f}ms -> {fc/(sp_tot+lg_tot):.1f} FPS')
