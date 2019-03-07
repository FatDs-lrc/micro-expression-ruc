from .resnet import resnet50
from .base_model import BaseModel
import torch.optim as optim
import torch.nn.functional as F_nn
import torch

class BaselineModel(BaseModel):
    ''' for simple resnet model '''

    def __init__(self, opt):
        super().__init__(opt)
        self.model = resnet50(num_classes=5)
        self.init_net()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.momentum, 0.999))
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=opt.lr)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=opt.momentum)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, opt.schedule, opt.gamma)
        if opt.resume_epoch != 'None':
            self.resume(opt.resume_epoch)
        self.criterion = F_nn.nll_loss

    def set_input(self, data):
        self.x = data['img'].cuda()
        self.label = data['label'].cuda()
        self.path = data['path']

    def forward(self):
        self.prob = self.model(self.x)
        self.prob = F_nn.log_softmax(self.prob, dim=1)
        _, self.pred = torch.max(self.prob, dim=1)

    def backward(self):
        self.loss = self.criterion(self.prob, self.label)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def run_one_batch(self, data, isTrain=True):
        # self.scheduler.step()
        self.set_input(data)
        self.forward()
        if isTrain:
            self.backward()
        return self.pred

    def get_current_loss(self):
        return self.loss
