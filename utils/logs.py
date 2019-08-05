import os
import time
import sys
import torch
import tensorboardX

class Logger(object):
	def __init__(self,args):
		"""
		Create a summary writer logging to log_dir
		"""
		if not os.path.exists(args.logdir):
			os.mkdir(args.logdir)
		time_str=time.strftime('%Y-%m-%d-%H-%M')
		logdir=os.path.join(args.logdir,time_str)
		if not os.path.exists(logdir):
			os.mkdir(logdir)
		self.writer=tensorboardX.SummaryWriter(log_dir=logdir)
		self.log=open(os.path.join(logdir,'log.txt'),'w')
	def write(self,txt):
		self.log.write(txt)
	def close(self):
		self.log.close()
	def scale_summary(self,tag,value,step):
		"""
		log a scalar variable
		:param value:
		:param step:
		:return:
		"""
		self.writer.add_scalar(tag,value,step)

