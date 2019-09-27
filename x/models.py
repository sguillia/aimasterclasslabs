import torch.nn as nn
import torch.nn.functional as F

"""
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc0 = nn.Linear(28*28, 27)

	def forward(self, x):
		x = self.fc0(x.view(x.size(0), -1))

		return F.log_softmax(x)
"""
"""
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv0 = nn.Conv2d(1, 2, 5)
		self.conv1 = nn.Conv2d(2, 4, 5)
		self.conv2 = nn.Conv2d(4, 8, 5)
		self.conv3 = nn.MaxPool2d(2)
		self.conv4 = nn.Linear(512, 27) # 27 = alphabet + 1 for unknown, 512 = 8*8*8

	def forward(self, x):
		#x = F.relu(self.conv1(x))

		x = self.conv0(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x.view(x.size(0), -1))
		
		return F.log_softmax(x)
   #return F.relu(self.conv2(x))
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 27)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
