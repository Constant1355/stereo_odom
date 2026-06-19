import torch, os
f = os.path.expanduser('~/Documents/stereo_odom/superpoint_v1.pth')
print('File:', f)
print('Size:', round(os.path.getsize(f)/1024, 1), 'KB')
ckpt = torch.load(f, map_location='cpu')
if isinstance(ckpt, dict):
    keys = list(ckpt.keys())
    print('Type: dict, keys:', len(keys))
    for k in keys:
        v = ckpt[k]
        if hasattr(v, 'shape'):
            print(f'  {k}: {list(v.shape)}')
        else:
            print(f'  {k}: {type(v).__name__}')
    total = sum(v.numel() for v in ckpt.values() if hasattr(v, 'numel'))
    print(f'Total params: {total/1e6:.2f}M')
