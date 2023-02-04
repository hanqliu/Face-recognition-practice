from lenet import LeNet5
from lenet import FaceData
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# import visdom

# viz = visdom.Visdom()

## 加载人脸数据，形成可迭代对象
data_train = FaceData("C:/Users/Lenovo/Desktop/pics/data/train",
                      transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))

data_test = FaceData("C:/Users/Lenovo/Desktop/pics/data/test",transforms.Compose([
                       transforms.Resize((32, 32)),
                       transforms.ToTensor()]))
## 将数据按照batch格式进行组合
data_train_loader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4)
data_test_loader = DataLoader(data_test, batch_size=32, num_workers=4)
## 定义神经网络、定义损失函数、优化器等
net = LeNet5()
# criterion = nn.CrossEntropyLoss()#nn.LogSoftmax+NLLLoss=CrossEntropyLoss
criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

# cur_batch_win = None
# cur_batch_win_opts = {
#     'title': 'Epoch Loss Trace',
#     'xlabel': 'Batch Number',
#     'ylabel': 'Loss',
#     'width': 1200,
#     'height': 600,
# }


def train(epoch):
    # global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # # Update Visualization
        # if viz.check_connection():
        #     cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
        #                              win=cur_batch_win, name='current_batch_loss',
        #                              update=(None if cur_batch_win is None else 'replace'),
        #                              opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()


def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1, 16):
        train_and_test(e)


if __name__ == '__main__':
    main()
