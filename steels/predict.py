from module import SpacingModule, OverlapModule, HookAngleModule
import numpy as np
import os, cv2, argparse

'''
arguemnt
'''

parser = argparse.ArgumentParser()
parser.add_argument('dataset_dir', type=str, help='Path to dataset includes color, depth and coord folders')
args = parser.parse_args()


HookAngle = HookAngleModule(threshold=0.7, device='cuda', model_path='./mrcnn_model/model_angle_ccc.pth')

prefixes = [os.path.splitext(name)[0] for name in os.listdir(os.path.join(args.dataset_dir, 'color'))]

os.makedirs(os.path.join(args.dataset_dir, 'angle_mask'), exist_ok=True)
os.makedirs(os.path.join(args.dataset_dir, 'angle_parameter'), exist_ok=True)

for n, prefix in enumerate(prefixes):
	color = cv2.imread(os.path.join(args.dataset_dir, 'color/{}.png'.format(prefix)))
	
	mask_angle, instances_angle = HookAngle.predict_mask(color, None)
	
	cv2.imwrite(os.path.join(args.dataset_dir, 'angle_mask/{}.png'.format(prefix)), mask_angle)
	
	info_angle = HookAngle.predict_angle(instances_angle)
	parameter_angle = HookAngle.draw_angle(color, info_angle)
	
	cv2.imwrite(os.path.join(args.dataset_dir, 'angle_parameter/{}.png'.format(prefix)), parameter_angle)
	print('{} / {}'.format(n+1, len(prefixes)), end='\r')
