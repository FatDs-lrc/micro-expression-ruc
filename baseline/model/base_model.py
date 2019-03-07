import torch
import os
import torch.backends.cudnn as cudnn
import torch.nn.init as init

class BaseModel(object):
    ''' Base model with basic save and load method'''

    def __init__(self, opt):
        self.isTrain = opt.isTrain
        self.save_dir = opt.save_dir

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def save(self, epoch):
        path = os.path.join(self.save_dir, 'epoch_{}.pth'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict,
            'optimizer_state_dict': self.optimizer.state_dict
        }, path)

    def resume(self, epoch):
        path = os.path.join(self.save_dir, 'epoch_{}.pth'.format(epoch))
        checkpoint = torch.load(path)
        assert int(epoch) == int(checkpoint['epoch']), 'Error epochs don\'t match between checkpoint and its name, may be sth wrong in save method'
        print("--------------------------------------------------")
        print("-------------Epoch:{} state resumed-------------".format(checkpoint['epoch']))
        print("--------------------------------------------------\n")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.isTrain:
            self.train()
        else:
            self.eval()

    def init_net(self, init_type='normal', gain=1, *args):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        cudnn.benchmark = True
        self.model = self.model.to(self.device)
        for arg in args:
            arg.cuda()
        self.init_weights(self.model, init_type=init_type, gain=gain)

    def init_weights(self, net, init_type='normal', gain=1):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
