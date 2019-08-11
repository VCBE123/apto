import glob
import cv2
import tqdm
import numpy as np
import ray
def crop_image_from_gray(image, tol=7):
	if image.ndim == 2:
		mask = image > tol
		return image[np.ix_(mask.any(1), mask.any(0))]

	elif image.ndim == 3:
		gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		mask = gray_image > tol

		check_shape = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
		if (check_shape == 0):
			return image
		else:
			image1 = image[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
			image2 = image[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
			image3 = image[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
			image = np.stack([image1, image2, image3], axis=-1)
	return image


def load_ben_color(path, sigmax=10):

	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = crop_image_from_gray(image)
	image = cv2.resize(image, (512, 512))
	image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmax), -4, 128)
	return image

@ray.remote(num_return_vals=1,num_cpus=32)
def reformat(path):
	for image in tqdm.tqdm(path):
		# img_mat=cv2.imread(image)
		img_mat=load_ben_color(image)
		image_name=image.split('/')[-1]
		image_name=image_name.split('.')[0]
		cv2.imwrite('/data/wen/data/aptos/fusion/preprocess_test_images/'+image_name+'.png',img_mat)
	return 0

images=glob.glob('/data/wen/data/aptos/test_images/*.*')
print(images)
n=len(images)//32+1
final = [images[i * n:(i + 1) * n] for i in range((len(images) + n - 1) // n )]
print(len(final))
# ray.init()
# result=[reformat.remote(final[i]) for i in range(32)]
# ray.get(result)
for image in tqdm.tqdm(images):
	# img_mat=cv2.imread(image)
	img_mat = load_ben_color(image)
	image_name = image.split('/')[-1]
	image_name = image_name.split('.')[0]
	cv2.imwrite('/data/wen/data/aptos/fusion/test/' + image_name + '.png', img_mat)