from torch import nn

class MarioNet(nn.Module):
    # 构造简单的CNN结构 要求输入H*W=84*84
    def __build_cnn(self, c, output_dim):
        return nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            # 输出 20*20*32
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # 输出 9*9*64
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # 输出 7*7*64
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )
        
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
        
        if h != 84:
            raise ValueError(f"Expecting input height:84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width:84, got: {w}")
        
        self.online = self.__build_cnn(c, output_dim)
        
        self.target = self.__build_cnn(c, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
