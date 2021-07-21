import os, cv2, datetime
import numpy as np

class PicViewer(object):
	def __init__(self, cover=np.zeros((45, 80, 3))):
		self.cover = cover
		self.reset()
	
	def reset(self):
		self.img = self.cover.copy()
		self.name = None
		self.state = 'mask'
		self.pic_path = {}
	
	def run(self):		
		#cv2.namedWindow('PicViewer', cv2.WND_PROP_FULLSCREEN)
		#cv2.setWindowProperty('PicViewer', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
		while True:
			cv2.namedWindow('PicViewer', cv2.WINDOW_NORMAL)
			cv2.imshow('PicViewer', self.img)
			presskey = cv2.waitKey(1)			
			
			if presskey == 97:
				self.back()
				
			elif presskey == 100:
				self.next()
				
			elif presskey == 32:
				self.switch()
				
			elif presskey == 27:
				cv2.destroyAllWindows()
				break
			elif presskey == 49:
				self.img = cv2.imread('dataset/1.png')
				self.state = 'mask'
			
			elif presskey == 50:
				self.img = cv2.imread('dataset/2.png')
				self.state = 'mask'
				
			elif presskey == 51:
				self.img = cv2.imread('dataset/cover.png')
				self.state = 'mask'
						
				
		exit()
			
	def update_mask(self, mask):
		self.img = mask
		self.name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		mask_path = 'predictions/'+self.name+'_mask.png'
		cv2.imwrite(mask_path, mask)
		self.pic_path[self.name] = {'mask':mask_path, 'parameter':False}
		return self.name
	
	def update_parameter(self, parameter, name):
		parameter_path = 'predictions/'+name+'_parameter.png'
		self.pic_path[name]['parameter'] = parameter_path
		cv2.imwrite(parameter_path, parameter)
		
	def switch(self):
		if len(self.pic_path) == 0: return
		self.state = 'parameter' if self.state == 'mask' else 'mask'
		if self.pic_path[self.name][self.state]:
			self.img = cv2.imread(self.pic_path[self.name][self.state])
	
	def back(self):
		if len(self.pic_path) == 0: return
		index = list(self.pic_path.keys()).index(self.name)
		if index > 0 and self.pic_path[self.name][self.state] is not False:
			self.name = list(self.pic_path.keys())[index-1]
			self.img = cv2.imread(self.pic_path[self.name][self.state])
	
	def next(self):
		if len(self.pic_path) == 0: return
		index = list(self.pic_path.keys()).index(self.name)
		if index < len(self.pic_path)-1 and self.pic_path[self.name][self.state] is not False:
			self.name = list(self.pic_path.keys())[index+1]
			self.img = cv2.imread(self.pic_path[self.name][self.state])
			
	def quit(self):
		pass
		
if __name__ == '__main__':
	pv = PicViewer()
	pv.run()

