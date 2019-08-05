import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune
from ray.tune import track
from ray.tune.schedulers import AsyncHyperBandScheduler

EPOCH_SIZE = 120
TEST_SIZE = 256


class Net(nn.Module):
	def __init__(self, config):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
		self.fc = nn.Linear(192, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 3))
		x = x.view(-1, 192)
		x = self.fc(x)
		return F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader, device):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if batch_idx * len(data) > EPOCH_SIZE:
			return
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()


def test(model, data_loader, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(data_loader):
			if batch_idx * len(data) > TEST_SIZE:
				break
			data, target = data.to(device), target.to(device)
			output = model(data)
			_, predicted = torch.max(output.data, 1)
			total += target.size(0)
			correct += (predicted == target).sum().item()
	return correct / total


def get_data_loaders():
	mnist_trans = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.1307,), (0.3081,))])
	train_loader = torch.utils.data.DataLoader(
		datasets.MNIST(root='/home/wen/Desktop/apto/data/', train=True, transform=mnist_trans, download=False), batch_size=64,
		shuffle=True)
	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST(root='/home/wen/Desktop/apto/data/', train=False, transform=mnist_trans, download=False), batch_size=64,
		shuffle=False)
	return train_loader, test_loader


def train_mnist(config):
	use_cuda = config.get('use_gpu') and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	train_loader, test_loader = get_data_loaders()
	model = Net(config).to(device)

	optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config['momentum'])

	while True:
		train(model, optimizer, train_loader, device)
		acc = test(model, test_loader, device)
		track.log(mean_accuracy=acc)


if __name__ == "__main__":
	parse = argparse.ArgumentParser(description="Pytorch tune")
	parse.add_argument("--cuda", action="store_true", default=True)
	parse.add_argument("--smoke-test", action="store_true", help="Finish quickly for testing")
	parse.add_argument("--ray-redis-address", help='address of ray cluster for seamless distributed execution')
	args = parse.parse_args()
	if args.ray_redis_address:
		ray.init(redis_address=args.ray_redis_address)
	sched = AsyncHyperBandScheduler(time_attr="training_iteration", metric="mean_accuracy")
	tune.run(train_mnist, name="exp", scheduler=sched,reuse_actors=False,
			 stop={"mean_accuracy": 0.99, "training_iteration": 5 if args.smoke_test else 100},
			 resources_per_trial={"cpu": 2, "gpu": int(args.cuda)},
			 num_samples=1 if args.smoke_test else 50,
			 config={'lr': tune.sample_from([1e-3,1e-4,1e-5,1e-2,1e-1]),
					 'momentum': tune.uniform(0.5, 0.99), "use_gpu": int(args.cuda)})
