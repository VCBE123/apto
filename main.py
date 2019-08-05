import torchvision
import pandas
import torchvision.models
import torch
import os
import torch.optim as optim
from dataset import get_data
from utils import AverageMeter, Logger, save_checkpoint, restore
import argparse
import torch.nn as nn
from torch.optim import lr_scheduler
import tqdm


def get_arguments():
	argument = argparse.ArgumentParser('transfer apto')
	argument.add_argument('--lr', default=0.003)
	argument.add_argument('--bach_size', default=64)
	argument.add_argument('--input', default=224)
	argument.add_argument('--ckpdir', default='checkpoint')
	argument.add_argument('--logdir', default='logs')
	argument.add_argument('--gpus', default='2,3')
	argument.add_argument('--epochs', default=2)
	argument.add_argument('--resume', default=False)
	return argument.parse_args()


def main():
	args = get_arguments()
	os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
	logger = Logger(args)
	model = torchvision.models.resnet50(pretrained=True)
	model.fc = nn.Linear(2048, 5)
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
	criterion = nn.CrossEntropyLoss()
	train_loader, val_loader, test_loader = get_data()
	scheduler = lr_scheduler.StepLR(optimizer, step_size=30)
	model.cuda()
	model = torch.nn.DataParallel(model, [0, 1])
	for epoch in range(1,args.epochs+1):
		train_loss = train(args, model, train_loader, optimizer, criterion, epoch, scheduler)
		val_loss, val_acc = val(args, model, val_loader, optimizer, criterion, epoch)
		logger.scale_summary('train_loss',train_loss,epoch)
		logger.scale_summary('val_loss',val_loss,epoch)
		logger.scale_summary('val_acc',val_acc,epoch)
		if epoch == args.epochs:
			test(args, model, test_loader)


def train(args, model, loader, opt, criterion, epoch, scheduler):
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
	print("\ntrain loss {:.4f}".format(epoch_loss))
	return epoch_loss


def val(args, model, loader, opt, criterion, epoch):
	val_loss = 0.0
	val_acc = 0.0
	if epoch % 5 == 0:
		model.eval()
		for input, target in loader:
			input, target = input.cuda(), target.cuda()
			output = model(input)
			loss = criterion(output, target)
			pred = torch.argmax(output, dim=1)
			val_loss += loss.item()
			val_acc += len(pred == target) / input.size(0)
		val_loss /= len(loader)

		print('\nval_loss:{:.4f}'.format(val_loss))
		print('\nval_acc:{:.2f}'.format(val_acc))
	return val_loss, val_acc


def test(args, model, loader):
	model.eval()
	save_checkpoint(args, model.state_dict)
	submission = pandas.DataFrame({'id_code': [], 'diagnosis': []})
	for input, target, id in loader:
		input = input.cuda()
		out = model(input)
		pred = torch.argmax(out, dim=1)
		submission.append({'id_code':id},ignore_index=False)
		submission.append({'diagnosis':pred},ignore_index=False)
	submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
	main()
