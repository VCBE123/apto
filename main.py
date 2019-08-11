import torchvision
import pandas
import torchvision.models
import torch
from utils import load_pretrained_weights
import os
from model import Effnet
import torch.optim as optim
from dataset import get_data
from utils import AverageMeter, Logger, save_checkpoint, restore
import argparse
import torch.nn as nn
from torch.optim import lr_scheduler
import tqdm
from sklearn.metrics import cohen_kappa_score

def get_arguments():
	argument = argparse.ArgumentParser('transfer apto')
	argument.add_argument('--lr', default=0.001)
	argument.add_argument('--bach_size', default=64)
	argument.add_argument('--input', default=224)
	argument.add_argument('--ckpdir', default='checkpoint')
	argument.add_argument('--logdir', default='logs')
	argument.add_argument('--gpus', default='0,1,2,3')
	argument.add_argument('--epochs', default=30)
	argument.add_argument('--resume', default=False)
	return argument.parse_args()


def main():
	args = get_arguments()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	logger = Logger(args)
	model=Effnet(num_classes=5,width_coeffficient=1.2,depth_coefficient=1.4,drop_out=0.3)
	model=load_pretrained_weights(model,'efficientnet-b3')
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
	criterion = nn.CrossEntropyLoss()
	train_loader, val_loader, test_loader = get_data()
	scheduler = lr_scheduler.StepLR(optimizer, step_size=20,gamma=0.1)
	# scheduler = lr_scheduler.CosineAnnealingLr(optimizer,len(train_loader),eta_min=1e-6)
	model.cuda()
	model = torch.nn.DataParallel(model, [0, 1,2,3])
	best_val_acc=0.0
	for epoch in range(1,args.epochs+1):
		train(args, model, train_loader, optimizer, criterion, epoch, scheduler,logger)
		val_acc=val(args, model, val_loader, optimizer, criterion, epoch,logger)
		if val_acc>best_val_acc:
			best_val_acc=val_acc
			save_checkpoint(args, model.state_dict(), filename='epoch_{}_best_{}.pth'.format(epoch,val_acc))
		# if epoch == args.epochs:
		# 	test(args, model, test_loader)


def train(args, model, loader, opt, criterion, epoch, scheduler,logger):
	scheduler.step()
	model.train()
	running_loss = 0.0
	for input, target in tqdm.tqdm(loader):
		input, target = input.cuda(), target.cuda()
		opt.zero_grad()
		output = model(input)
		loss = criterion(output, target)
		loss.backward()
		opt.step()
		running_loss += loss.item()
	epoch_loss = running_loss / len(loader)
	print("\nepoch:{} train loss {:.4f} ".format(epoch,epoch_loss))
	logger.scale_summary('train_loss', epoch_loss, epoch)
	return epoch_loss

def val(args, model, loader, opt, criterion, epoch,logger):
	val_loss = 0.0
	val_acc = 0.0
	val_kappa=0.0
	if epoch:
		model.eval()
		for input, target in loader:
			input, target = input.cuda(), target.cuda()
			output = model(input)
			loss = criterion(output, target)
			pred = torch.argmax(output, dim=1)
			val_loss += loss.item()
			val_acc += torch.sum(pred == target).item()*1.0/ input.size(0)
			val_kappa+=cohen_kappa_score(pred.cpu(),target.cpu(),weights='quadratic')
		val_loss /= len(loader)
		val_acc/=len(loader)
		val_kappa/=len(loader)
		print('epoch val: {}'.format(epoch))
		print('\nval_loss:{:.4f}'.format(val_loss))
		print('\nval_acc:{:.2f}'.format(val_acc))
		print('\nval_kappa:{:.2f}'.format(val_kappa))
		logger.scale_summary('val_loss',val_loss,epoch)
		logger.scale_summary('val_acc',val_acc,epoch)
		logger.scale_summary('val_kappa',val_kappa,epoch)
	return val_acc


def test(args, model, loader):
	model.eval()
	save_checkpoint(args, model.state_dict())
	submission = pandas.DataFrame({'id_code': [], 'diagnosis': []})
	for input, target, id in loader:
		input = input.cuda()
		out = model(input)
		pred = torch.argmax(out, dim=1).cpu().numpy()
		pred=list(map(lambda x:int(x),pred))
		submission=submission.append(pandas.DataFrame({'id_code':id,'diagnosis':pred}),ignore_index=True)
	submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
	main()
