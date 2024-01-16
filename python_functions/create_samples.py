import matplotlib.pyplot as plt
import numpy as np

array = [0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x3ad48080, 0xa4bc1820, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x7f5f8080, 0x727f7f7f, 0x47474747, 0x47474747, 0x8080b42b, 0x80808080, 0x80808080, \
  0xf2c38080, 0x6424f2c8, 0x7f7f627f, 0x7f667b7f, 0x80800d7f, 0x80808080, 0x80808080, 0x80808080, \
  0x91808080, 0xc3c38ec2, 0x6d95bbc3, 0x8080ea7f, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x7ed38080, 0x80809252, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x7f6a9680, 0x808080d3, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x6f7f0280, \
  0x808080ac, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xbe7f7abb, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x853c7f06, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x89808080, 0x80ba794e, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0xfe808080, 0x8080377f, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x7ccb8080, 0x8080b971, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x7f5e9380, 0x80808027, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x5c7f4c83, \
  0x808080a3, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0xcd7f7fa6, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x9f808080, 0x81f37f61, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x06808080, 0x80b47f7f, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x73bd8080, 0x80b47f7f, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x7ff98080, 0x80a85c7f, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x7ff98080, 0x80809250, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, \
  0x80808080, 0x80808080, 0x80808080, 0x808080808]



import torch
from torchvision import datasets, transforms
import numpy as np


class normalize:

    def __init__(self):
        print("OK")
        # self.args = args

    def __call__(self, img):
        # if self.args.act_mode_8bit:
        #     return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)
        return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

def mod2(val):
    # Convert the hexadecimal value to an integer
    value = int(val, 16)

    # Perform the bitwise operations and sum
    return -((value >> 7) & 0b1) * (2 ** 7) + ((value >> 6) & 0b1) * (2 ** 6) + \
           ((value >> 5) & 0b1) * (2 ** 5) + ((value >> 4) & 0b1) * (2 ** 4) + \
           ((value >> 3) & 0b1) * (2 ** 3) + ((value >> 2) & 0b1) * (2 ** 2) + \
           ((value >> 1) & 0b1) * (2 ** 1) + ((value >> 0) & 0b1) * (2 ** 0)

def reverse_mod2(mod2_output):
    result = 0

    if mod2_output < 0:
        result |= 1 << 7
        mod2_output += 2 ** 7

    for i in range(6, -1, -1):
        if mod2_output >= 2 ** i:
            result |= 1 << i
            mod2_output -= 2 ** i

    return format(result, '02x')

train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5),
            transforms.ToTensor(),
            normalize()
        ])

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,
                                                   transform=train_transform)


arr = []
indexes = np.zeros((10, 10), dtype=int)
ind = np.zeros(10, dtype=int)

for i, (train_image_zero, train_target_zero) in enumerate(mnist_trainset):
    if ind[train_target_zero] < 10:
        indexes[train_target_zero][ind[train_target_zero]] = i
        ind[train_target_zero] += 1

    # Check if all digits have 10 samples each
    if np.all(ind == 10):
        break

print(indexes)


train_image_zero, train_target_zero = mnist_trainset[0]

print(train_target_zero)


train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            lambda img: transforms.functional.rotate(img, -90),
            lambda img: transforms.functional.hflip(img),
            transforms.ToTensor(),
            normalize()
        ])


trainset = datasets.EMNIST(root='./data', split='letters', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True)



arr = []
indexes2 = np.zeros((1, 100), dtype=int)
ind = np.zeros(1, dtype=int)
ind_to = 16
for i, (train_image_zero, train_target_zero) in enumerate(trainset):
    if train_target_zero == ind_to:
        if ind[0] < 100:
            indexes2[0][ind[0]] = i
            ind[0] += 1
            
    if np.all(ind == 100):
        break

print(indexes2)

# plt.imshow((trainset[indexes2[0][1]][0])[0], cmap='gray')  # 'gray' colormap for grayscale image
# plt.title("MNIST Digit")
# plt.axis('off')  # Turn off axis numbers and labels
# plt.show()



dataiter = iter(trainloader)
images, labels = dataiter.next()

images = []

for i in range(10):
    images.append(trainset[indexes2[0][i]][0])

# for j in range(1, 10):
#     for i in range(10):
#         images.append(mnist_trainset[indexes[j][i]][0])

print(len(images))
# images = [trainset[indexes[0][1]][0], trainset[indexes[0][2]][0]]

hex_images = []
for image in images:
    np_image = image.numpy().squeeze()

    np_image = np_image.astype(np.uint8)
    hex_image = np_image.flatten().tolist()
    hex_image = ["{:02x}".format(pixel) for pixel in hex_image]
    hex_images.append(hex_image)

print(hex_images[0])

def format_hex_data(hex_images):
    formatted_data = "int samples_training[40][196]= { \\\n"
    for img in hex_images:
        formatted_data += "{ \\\n"
        for i in range(0, len(img), 4):
            combined_hex = "0x" + "".join(img[i:i+4])
            if i + 4 < len(img):
                combined_hex += ", "
            formatted_data += combined_hex
        formatted_data += "}, \\\n"
    formatted_data += "};"
    return formatted_data

formatted_hex_images = format_hex_data(hex_images)

with open('formatted_hex_images.h', 'w') as file:
    file.write(formatted_hex_images)
