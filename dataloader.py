from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class LabelMe(Dataset):
    def __init__(self, path):
        self.view1 = scipy.io.loadmat(path + 'LabelMe.mat')['X1'].astype(np.float32)
        self.view2 = scipy.io.loadmat(path + 'LabelMe.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'LabelMe.mat')['Y']  # .transpose()
        self.y = labels

    def __len__(self):
        return 2688

    def __getitem__(self, idx):
        return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
            self.y[idx]), torch.from_numpy(np.array(idx)).long()


class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path + 'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path + 'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path + 'MFCC.npy').astype(np.float32)
        self.labels = np.load(path + 'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
            x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path + 'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path + 'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path + 'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
            self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(
                self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class COIL20(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'COIL20.mat')['Y'].astype(np.int32).reshape(1440, )
        self.V1 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'COIL20.mat')['X'][0, 2].astype(np.float32)

    def __len__(self):
        return 1440

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(1024)
        x2 = self.V2[idx].reshape(3304)
        x3 = self.V3[idx].reshape(6750)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class MSRCv1(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MSRCv1.mat')['Y'].astype(np.int32).reshape(210, )
        self.V1 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'MSRCv1.mat')['X'][0, 4].astype(np.float32)

    def __len__(self):
        return 210

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(24)
        x2 = self.V2[idx].reshape(576)
        x3 = self.V3[idx].reshape(512)
        x4 = self.V4[idx].reshape(256)
        x5 = self.V5[idx].reshape(254)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech101(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Caltech101-20.mat')['Y'].astype(np.int32).reshape(2386, )
        self.V1 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 4].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'Caltech101-20.mat')['X'][0, 5].astype(np.float32)

    def __len__(self):
        return 2386

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(48)
        x2 = self.V2[idx].reshape(40)
        x3 = self.V3[idx].reshape(254)
        x4 = self.V4[idx].reshape(1984)
        x5 = self.V5[idx].reshape(512)
        x6 = self.V6[idx].reshape(928)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5),torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class YouTubeFace(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'YouTubeFaces.mat')['Y'].astype(np.int32).reshape(101499, )
        self.V1 = scipy.io.loadmat(path + 'YouTubeFaces.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'YouTubeFaces.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'YouTubeFaces.mat')['X'][0, 2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'YouTubeFaces.mat')['X'][0, 3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'YouTubeFaces.mat')['X'][0, 4].astype(np.float32)


    def __len__(self):
        return 101499

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(64)
        x2 = self.V2[idx].reshape(512)
        x3 = self.V3[idx].reshape(64)
        x4 = self.V4[idx].reshape(647)
        x5 = self.V5[idx].reshape(838)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3),torch.from_numpy(x4),torch.from_numpy(x5)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class DHA(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'DHA.mat')['Y'].astype(np.int32).reshape(483, )
        self.V1 = scipy.io.loadmat(path + 'DHA.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'DHA.mat')['X'][0, 1].astype(np.float32)


    def __len__(self):
        return 483

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(110)
        x2 = self.V2[idx].reshape(6144)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Scene15(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Scene15.mat')['Y'].astype(np.int32).reshape(4485, )
        self.V1 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0, 0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0, 1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0, 2].astype(np.float32)


    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(20)
        x2 = self.V2[idx].reshape(59)
        x3 = self.V3[idx].reshape(40)

        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class nus_wide(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/nus-wide/L.mat')['L'].reshape(20000, )
        self.V1 = scipy.io.loadmat('./data/nus-wide/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/nus-wide/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class flickr(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/flickr30k/L.mat')['L'].reshape(12154, )
        self.V1 = scipy.io.loadmat('./data/flickr30k/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/flickr30k/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 12154

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()



class ESP_Game(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat('./data/esp-game/L.mat')['L'].reshape(11032, )
        self.V1 = scipy.io.loadmat('./data/esp-game/img.mat')['img'].astype(np.float32)
        self.V2 = scipy.io.loadmat('./data/esp-game/txt.mat')['txt'].astype(np.float32)


    def __len__(self):
        return 11032

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(100)
        x2 = self.V2[idx].reshape(100)

        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


def load_data(dataset):
    if dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "LabelMe":
        dataset = LabelMe('./data/')
        dims = [512, 245]
        view = 2
        data_size = 2688
        class_num = 8
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5

    elif dataset == "MNIST-USPS":  # 数据格式与本地存储的数据集不一致，可以添加到数据集中
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000

    elif dataset == "Caltech-2V":  # 本地数据集无此数据
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "COIL20":
        dataset = COIL20('./data/')
        dims = [1024, 3304, 6750]
        view = 3
        data_size = 1440
        class_num = 20
    elif dataset == 'MSRCv1':
        dataset = MSRCv1('./data/')
        dims =[24,576,512,256,254]
        view = 5
        data_size = 210
        class_num = 7
    elif dataset == 'Caltech101-20':
        dataset = Caltech101('./data/')
        dims = [48,40,254,1984,512,928]
        view = 6
        data_size = 2386
        class_num =20
    elif dataset =='YouTubeFaces':
        dataset = YouTubeFace('./data/')
        dims = [64, 512, 64, 647, 838]
        view = 5
        data_size = 101499
        class_num = 31
    elif dataset == 'DHA':
        dataset = DHA('./data/')
        dims = [110, 6144]
        view = 2
        data_size = 483
        class_num = 23
    elif dataset == 'Scene15':
        dataset = Scene15('./data/')
        dims = [20,59,40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == 'nus-wide':
        dataset = nus_wide('./data/')
        dims = [100, 100]
        view = 2
        data_size = 20000
        class_num = 8
    elif dataset == 'flickr':
        dataset = flickr('./data/')
        dims = [100,100]
        view = 2
        data_size = 12154
        class_num = 6

    elif dataset == 'ESP-Game':
        dataset = ESP_Game('./data/')
        dims = [100, 100]
        view = 2
        data_size = 11032
        class_num = 7
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
