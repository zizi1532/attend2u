# extract features using Torch

https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/README.md#extracting-image-features

https://github.com/Cadene/pretrained-models.pytorch


~~~python
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import os
from tqdm import tqdm
import numpy as np

model = pretrainedmodels.__dict__['resnet101'](num_classes=1000, pretrained='imagenet')
model.cuda()
load_img = utils.LoadImage()
# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

def extract_features(paths):
	batch = None
	for path in paths:
		input_img = load_img(path)
		input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
		input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
		input = torch.autograd.Variable(input_tensor,
		    requires_grad=False).cuda()
		batch = input if (batch is None) else torch.cat([batch, input], dim=0)
	output_features = model.features(batch) # 1x14x14x2048 size may differ
	return output_features





data_dir = './images'
img_names = os.listdir(data_dir)
result_dir='./image_features'
if not os.path.exists(result_dir): os.mkdir(result_dir)

batch_size = 32
# for img_name in tqdm(img_names):
for i in tqdm(range(0, len(img_names), batch_size)):
	batch_img_names = img_names[i:i+batch_size]
	
	paths = [os.path.join(data_dir, x) for x in batch_img_names]
	output_features=extract_features(paths=paths)
	output_features=output_features.cpu().data.numpy()
	for j, img_name in enumerate(batch_img_names):
		np.save(os.path.join(result_dir, img_name), output_features[j])

~~~
