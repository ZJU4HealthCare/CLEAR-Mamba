# import os
# from medmnist import RetinaMNIST,OCTMNIST
#
# # 指定保存路径
# save_dir = "./datasets/RetinaMNIST"
# os.makedirs(save_dir, exist_ok=True)
#
# # 下载 RetinaMNIST，
# train_dataset = RetinaMNIST(split="train", download=True, root=save_dir, size=224)
# val_dataset   = RetinaMNIST(split="val",   download=True, root=save_dir, size=224)
# test_dataset  = RetinaMNIST(split="test",  download=True, root=save_dir, size=224)
#
# print("下载完成！文件保存在:", save_dir)
#
# # 指定保存路径
# save_dir = "./datasets/octmnist"
# os.makedirs(save_dir, exist_ok=True)
#
# # 下载 OCTMNIST，
# train_dataset = OCTMNIST(split="train", download=True, root=save_dir, size=224)
# val_dataset   = OCTMNIST(split="val",   download=True, root=save_dir, size=224)
# test_dataset  = OCTMNIST(split="test",  download=True, root=save_dir, size=224)
#
# print("下载完成！文件保存在:", save_dir)

import numpy as np

# 你的 .npz 文件路径
data = np.load("./datasets/octmnist/octmnist_224.npz")

# 查看里面有哪些 key
print(data.files)
# 输出类似: ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']

# 取出数据
X_train, y_train = data["train_images"], data["train_labels"]
X_val,   y_val   = data["val_images"],   data["val_labels"]
X_test,  y_test  = data["test_images"],  data["test_labels"]

print("训练集:", X_train.shape, y_train.shape)
print("验证集:", X_val.shape, y_val.shape)
print("测试集:", X_test.shape, y_test.shape)

data = np.load("./datasets/RetinaMNIST/retinamnist.npz")
# 查看里面有哪些 key
print(data.files)
# 输出类似: ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']

# 取出数据
X_train, y_train = data["train_images"], data["train_labels"]
X_val,   y_val   = data["val_images"],   data["val_labels"]
X_test,  y_test  = data["test_images"],  data["test_labels"]

print("训练集:", X_train.shape, y_train.shape)
print("验证集:", X_val.shape, y_val.shape)
print("测试集:", X_test.shape, y_test.shape)
