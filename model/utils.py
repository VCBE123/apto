import torch
import torch.nn as nn

class Swish(nn.Module):
	def forward(self, x):
		return x * torch.sigmoid(x)

class Flatten(nn.Module):
	def forward(self, x):
		return x.reshape(x.shape[0], -1)

class SqueezeExcitation(nn.Module):
	def __init__(self, inplane, se_plane):
		super(SqueezeExcitation, self).__init__()
		self.reduce_expand = nn.Sequential(
			nn.Conv2d(inplane, se_plane, kernel_size=1, stride=1, padding=0, bias=True),
			Swish(),
			nn.Conv2d(se_plane, inplane, kernel_size=1, stride=1, padding=0, bias=True),
			nn.Sigmoid()
		)

	def forward(self, x):
		x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
		x_se = self.reduce_expand(x_se)
		return x_se * x
class MBConv(nn.Module):
	def __init__(self, inplane, plane, kernel_size, stride, expand_ratio=1.0, se_rate=0.25, drop_connect_rate=0.2):
		super(MBConv, self).__init__()
		expand_plane = int(inplane * expand_ratio)

		se_plane = max(1, int(inplane * se_rate))
		self.expand_conv = None
		if expand_ratio > 1.0:
			self.expand_conv = nn.Sequential(
				nn.Conv2d(inplane, expand_plane, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(expand_plane, momentum=0.01, eps=1e-3),
				Swish()
			)
			inplane = expand_plane
		self.depthconv = nn.Sequential(
			nn.Conv2d(inplane, expand_plane, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
					  groups=expand_plane, bias=False),
			nn.BatchNorm2d(expand_plane,momentum=0.01,eps=1e-3),
			Swish()
		)
		self.squeeze_excitation=SqueezeExcitation(expand_plane,se_plane)
		self.project_conv=nn.Sequential(
			nn.Conv2d(expand_plane,plane,kernel_size=1,stride=1,padding=0,bias=False),
			nn.BatchNorm2d(plane,momentum=0.01,eps=1e-3)
		)
		self.with_skip=stride==1
		self.drop_connect_rate=torch.tensor(drop_connect_rate,requires_grad=False)

	def __drop_connect(self,x):
		keep_prob=1-self.drop_connect_rate
		drop_mask=torch.rand(x.shape[0],1,1,1)+keep_prob
		drop_mask=drop_mask.type_as(x)
		drop_mask.floor_()
		return drop_mask*x/keep_prob
	def forward(self, x):
		z=x
		if self.expand_conv is not None:
			x=self.expand_conv(x)
		x=self.depthconv(x)
		x=self.squeeze_excitation(x)
		x=self.project_conv(x)

		if x.shape==z.shape and self.with_skip:
			if self.training and self.drop_connect_rate is not None:
				self.__drop_connect(x)
			x+=z
		return x



