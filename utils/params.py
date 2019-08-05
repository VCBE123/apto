from torch.utils import model_zoo
import torch
from collections import OrderedDict

def print_num_params(model, display_all_modules=False):
	total_num_params = 0
	for n, p in model.named_parameters():
		num_params = 1
		for s in p.shape:
			num_params *= s
		if display_all_modules: print("{}:{}".format(n, num_params))
		total_num_params += num_params
	print("*" * 50)
	print("Total number of parameters: {:.2e}".format(total_num_params))


url_map = {
	'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
	'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
	'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
	'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
	'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth',
	'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth',
}


def load_pretrained_weights(model, model_name):
	model_state= torch.load('/home/wen/efficientnet-b4-e116e8b3.pth')
	mapping={k:v for k,v in zip(model_state.keys(),model.state_dict().keys())}
	mapped_model_state=OrderedDict( [(mapping[k],v) for k ,v in model_state.items()])
	model.load_state_dict(mapped_model_state,strict=False)
	print('Loaded pretrained weights for {}'.format(model_name))

	return model