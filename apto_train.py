import argparse
import os
import torch
import torch.optim as optim
import torch.utils.model_zoo
import torch.nn as nn
import torchvision.models
import torch.nn.functional as F
from ray.tune import Trainable
import ray
from ray import tune
from ray.tune.schedulers import HyperBandScheduler
from utils import print_num_params,load_pretrained_weights
from dataset import get_data
classes=("NO DR","Mild","Moderate","servere","Proliferative DR")


# model=Effnet(num_classes=1000,width_coeffficient=1.4,depth_coefficient=1.8,drop_out=0.35)
model=torchvision.models.inception_v3(pretrained=True)
print_num_params(model)
# model=load_pretrained_weights(model,'efficientnet-b4')
model.fc=nn.Linear(2048,4)
EPOCH_SIZE = 125
TEST_SIZE = 125

parser = argparse.ArgumentParser(description='Pytorch MNIST ')
parser.add_argument('--batch-size', type=int, default=64, metavar="N", help="input batch size for training")
parser.add_argument('--test-batch-size', type=int, default=1000, metavar="N", help="input batchsize for testing")
parser.add_argument('--epochs', type=int, default=1, metavar="N", help="num of epochs to train")
parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar="M", help='sgd momentum')
parser.add_argument('--no-cuda', action="store_true", default=False)
parser.add_argument('--redis-address', default=None, type=str, help='the redis address of the cluster')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--smoke-test', action="store_true", help='finish quickly testing')




class TrainAPTO(Trainable):
	def _setup(self, config):
		args = config.pop("args", parser.parse_args([]))
		vars(args).update(config)
		args.cuda = not args.no_cuda and torch.cuda.is_available()

		torch.manual_seed(args.seed)
		if args.cuda:
			torch.cuda.manual_seed(args.seed)
		self.train_loader,self.eval_loader,self.test_loader=get_data()
		self.model = model
		if args.cuda:
			self.model.cuda()
		self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=args.momentum)
		self.args = args

	def _train_iteration(self):
		self.model.train()
		for batch_idx, (data, target) in enumerate(self.train_loader):
			if batch_idx * len(data) > EPOCH_SIZE:
				return
			if self.args.cuda:
				data, target = data.cuda(), target.cuda()
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = F.nll_loss(output, target)
			loss.backward()
			self.optimizer.step()

	def _test(self):
		self.model.eval()
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(self.eval_loader):
				if batch_idx * len(data)>TEST_SIZE:
					break
				if self.args.cuda:
					data, target = data.cuda(), target.cuda()
				output = self.model(data)
				test_loss += F.nll_loss(output, target, reduction="sum").item()
				pred = output.argmax(dim=1, keepdim=True)
				correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
		test_loss = test_loss / len(self.test_loader.dataset)
		accuracy = correct.item() / len(self.test_loader.dataset)
		return {"mean_loss": test_loss, "mean_accuracy": accuracy}

	def _train(self):
		self._train_iteration()
		return self._test()
	def _save(self, checkpoint_dir):
		checkpoint_path=os.path.join(checkpoint_dir,"model.pth")
		torch.save(self.model.state_dict(),checkpoint_path)
		return checkpoint_path

	def _restore(self, checkpointpath):
		self.model.load_state_dict(torch.load(checkpointpath))


if __name__ == '__main__':
	args = parser.parse_args()
	ray.init(redis_address=args.redis_address)
	sched = HyperBandScheduler(time_attr="training_iteration", metric="mean_loss", mode="min")
	tune.run(TrainAPTO, scheduler=sched,
			 **{"stop": {"mean_accuracy": 0.65, "training_iteration": 1 if args.smoke_test else 5, },
				"resources_per_trial": {"cpu": 6, "gpu": int(not args.no_cuda)},
				"num_samples": 1 if args.smoke_test else 4,
				"checkpoint_at_end": True,
				"config": {"args": args, "lr": tune.uniform(0.002, 0.0002), "momentum": tune.uniform(0.8, 0.95), }},reuse_actors=False)
