import pickle, json, numpy as np
import os

root = r'D:\waymo_4d\val.tar.gz.part\val.tar\val'
meta = os.listdir(os.path.join(root, 'meta_infos'))[0].replace('.pkl', '')
print('segment:', meta)

with open(os.path.join(root, 'meta_infos', meta + '.pkl'), 'rb') as f:
    meta_info = pickle.load(f)
print('meta_info keys:', meta_info.keys())

pcd_path = os.path.join(root, 'pcds', meta, '000.npz')
label_path = os.path.join(root, 'sam4d_labels', meta, '000.json')
img_path = os.path.join(root, 'undistort_images', meta, '000', 'FRONT.jpg')

print('pcd_path:', pcd_path, 'exists:', os.path.exists(pcd_path))
print('label_path:', label_path, 'exists:', os.path.exists(label_path))
print('img_path:', img_path, 'exists:', os.path.exists(img_path))

pcd = np.load(pcd_path)
print('pcd keys:', pcd.files)
print('pcd shape:', {k: v.shape for k, v in pcd.items()})

with open(label_path, 'r') as f:
    label = json.load(f)
print('label keys:', label.keys())
print('label example:', str(label)[:500])