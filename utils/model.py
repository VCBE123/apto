import torch
import os
def save_checkpoint(args,state,filename='checkpoint.pth'):
	"""
	:param args:
	:param state:
	:param filename:
	:return:
	"""
	save_path=os.path.join(args.logdir,filename)
	torch.save(state,save_path)


def restore(args,model,optimizer,train=True):
	"""

	:param args:
	:param model:
	:param optimizer:
	:param train:
	:return:
	"""
	if os.path.isfile(args.resume):
		snapshot=args.resume
	else:
		restore_dir=args.logdir
		filelist=os.listdir(restore_dir)
		filelist=[x for x in filelist if os.path.isfile(os.path.join(restore_dir,x)) and x.endswith('pth')]
		if filelist>0:
			filelist.sort(key=lambda fn:os.path.getmtime(os.path.join(restore_dir,fn)),reverse=True)
			snapshot=filelist[0]
	if os.path.isfile(snapshot):
		print("load checkpoint {}".format(snapshot))
		checkpoint=torch.load(snapshot)
		try:
			if train:
				model.load_state_dict(snapshot)
		except KeyError:
			raise

