from torchvision.transforms import Compose,RandomResizedCrop,RandomHorizontalFlip, RandomAffine,Resize,CenterCrop
import numpy as np
import cv2

import torch.utils.data as data
import os
import torch
from torchvision.transforms import ToTensor, Normalize
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.utils import shuffle
from PIL import Image


def crop_image_from_gray(image,tol=7):
	if image.ndim==2:
		mask=image>tol
		return image[np.ix_(mask.any(1),mask.any(0))]

	elif image.ndim==3:
		gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
		mask=gray_image>tol

		check_shape=image[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
		if (check_shape==0):
			return image
		else:
			image1=image[:,:,2][np.ix_(mask.any(1),mask.any(0))]
			image2=image[:,:,1][np.ix_(mask.any(1),mask.any(0))]
			image3=image[:,:,0][np.ix_(mask.any(1),mask.any(0))]
			image=np.stack([image1,image2,image3],axis=-1)
	return image


def load_ben_color(path,sigmax=10):
	image=cv2.imread(path)
	image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	image=crop_image_from_gray(image)
	image=cv2.resize(image,(512,512))
	image=cv2.addWeighted(image,4,cv2.GaussianBlur(image,(0,0),sigmax),-4,128)
	return image

class ImageDataset(torch.utils.data.Dataset):
	def __init__(self,root,path_list,target=None,transform=None,extension='.png',with_id=False):
		super(ImageDataset,self).__init__()
		self.root=root
		self.path_list=path_list
		self.target=target
		self.transform=transform
		self.extension=extension
		self.with_id=with_id
		if self.target is not None:
			assert  len(self.path_list)==len(self.target)
			self.target=torch.LongTensor(target)
	def __getitem__(self,index):
		path=os.path.join(self.root,  self.path_list[index]+self.extension)
		sample=load_ben_color(path)
		sample=Image.fromarray(sample)
		if self.transform is not None:
			sample=self.transform(sample)
		if self.target is not None:
			return sample,self.target[index]
		elif self.with_id:
			return sample,torch.LongTensor([]),self.path_list[index]
		else:
			return sample,torch.LongTensor([])
	def __len__(self):
		return len(self.path_list)



SEED=510
df_train = pd.read_csv('/data/wen/data/aptos/train.csv')
df_test = pd.read_csv('/data/wen/data/aptos/test.csv')
x = df_train['id_code']
y = df_train['diagnosis']
x, y = shuffle(x, y,random_state=SEED)
img_states=[[0.485,0.456,0.406],[0.229,0.224,0.225]]
train_x, valid_x, train_y, valid_y = train_test_split(x.values, y.values, test_size=0.10, stratify=y, random_state=SEED)
test_x = df_test.id_code.values


train_transform=Compose([
	Resize(250),
	RandomResizedCrop(224),
	RandomAffine(degrees=2,translate=(0.02,0.02),scale=(0.98,1.02),shear=2,fillcolor=(0,0,0)),
	RandomHorizontalFlip(),
	ToTensor(),
	Normalize(*img_states)
])

test_transform=Compose([

	Resize(250),
	CenterCrop(224),
	ToTensor(),
	Normalize(*img_states)
])


def get_data():
	train_set=ImageDataset(root='/data/wen/data/aptos/train_images/',path_list=train_x,target=train_y,transform=train_transform)
	train_eval_set=ImageDataset(root='/data/wen/data/aptos/train_images/',path_list=valid_x,target=valid_y,transform=test_transform)
	test_set=ImageDataset(root='/data/wen/data/aptos/test_images/',path_list=test_x,transform=test_transform,with_id=True)
	train_batch_size=64
	eval_batch_size=64
	num_workers=os.cpu_count()
	train_loader=data.DataLoader(train_set,batch_size=train_batch_size,num_workers=num_workers,shuffle=True,drop_last=True,pin_memory=True)
	eval_loader=data.DataLoader(train_eval_set,batch_size=eval_batch_size,num_workers=num_workers,shuffle=False,drop_last=False,pin_memory=True)
	test_loader=data.DataLoader(test_set,batch_size=eval_batch_size,num_workers=num_workers,shuffle=False,drop_last=False,pin_memory=True)
	return  train_loader,eval_loader,test_loader



