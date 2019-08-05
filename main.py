import torchvision
import torchvision.models
import torch
import os
import torch.optim as optim
from dataset import get_data
import argparse
import torch.nn as nn
from torch.optim import lr_scheduler
import tqdm
def args_():
	argument=argparse.ArgumentParser('transfer apto')
	argument.add_argument('--lr',default=0.003)
	argument.add_argument('--bach_size',default=64)
	argument.add_argument('--input',default=224)
	argument.add_argument('--logdir',default='.\logs')
	argument.add_argument('--gpus',default='2,3')
	argument.add_argument('--epochs',default=80)
	return argument.parse_args()


def main():
	args=args_()
	os.environ['CUDA_VISIBLE_DEVICES']=args.gpus

	model=torchvision.models.resnet50(pretrained=True)
	model.fc=nn.Linear(2048,5)
	optimizer=optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=1e-4,nesterov=True)
	loss=nn.CrossEntropyLoss()
	train_loader,val_loader,test_loader=get_data()
	steps=len(train_loader)//args.bach_size
	scheduler=lr_scheduler.StepLR(optimizer,step_size=30)
	model=model.cuda()
	for epoch in range(args.epochs):
		train(model,train_loader,optimizer,loss,epoch,scheduler)
		val(model,val_loader,optimizer,loss,epoch)
		if epoch==args.epochs-1:
			test(model,test_loader)



def train(model,loader,opt,loss,epoch,scheduler):
	scheduler.step()
	model.train()
	running_loss=0.0
	for input,target in tqdm.tqdm(loader):
		input,target=input.cuda(),target.cuda()
		opt.zero_grad()
		loss.zero_grad()
		output=model(input)
		loss_=loss(output,target)
		loss_.backward()
		opt.step()
		running_loss+=loss_.item()*input.size(0)

	epoch_loss=running_loss/len(loader)
	print("train loss {:.4f}".format(epoch_loss))
	return epoch_loss




def val(model,loader,opt,loss,epoch):
	if epoch%5==0:
		model.eval()
		val_loss=0.0
		for input,target in loader:
			input,target=input.cuda(),target.cuda()
			output=model(input)
			loss_=loss(output,target)
			pred=torch.argmax(output,dim=1)
			val_loss+=loss_.item()
			acc=len(pred==target)/len(loader)
		print('val_loss:{:.4f}'.format(loss_))
		print('val_acc:{:.2f}'.format(acc))
		return loss
def test(model,loader):
	model.eval()
	for input,target in loader:
		input =input.cuda()
		out=model(input)
		pred=torch.argmax(out,dim=1)


if __name__ == '__main__':
    main()




