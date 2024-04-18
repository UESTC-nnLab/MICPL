import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class MotionCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        

        super(MotionCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias) 
        

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state 
        
        combined = torch.cat([input_tensor, h_cur], dim=1)  
        
        combined_conv = self.conv(combined) 
   
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g       
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class Motion(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(Motion, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(MotionCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)
        
        self.cell_list2 = nn.ModuleList(cell_list) 
        self.maxpool = nn.AdaptiveMaxPool2d(input_dim)
        self.w0 = nn.Linear(input_dim*input_dim, input_dim)
        self.w1 = nn.Linear(input_dim, input_dim)
        self.w2 = nn.Linear(input_dim, input_dim)   
        self.se1 = SEAttention(channel=input_dim,reduction=2)
        self.se2 = SEAttention(channel=input_dim,reduction=2)
        self.conv1 = nn.Sequential(
                nn.Conv2d(input_dim*10, input_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(input_dim),
                nn.ReLU())
        self.mapping = nn.Sequential(
            nn.Linear(input_dim*2, input_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim, bias=False))
        
    def forward(self, input_tensor, hidden_state=None):
       
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            
            #-----------------Motion Pattern Mining-----------------#

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len): 
                
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                
                h_t = []
                batch_size = h.shape[0]
                channal = h.shape[1]
                width = h.shape[2]
                height = h.shape[3]
                
                node_feat = self.maxpool(h)
                node_feat = self.se1(node_feat).view(batch_size,-1, channal) 
                
                h_tt = h 
                cc = c 
                
                for tt in range(seq_len): 
                    h_tt, cc = self.cell_list2[layer_idx](input_tensor=cur_layer_input[:, tt, :, :, :],
                                                 cur_state=[h_tt, cc])  
                    ht = torch.cat((h, h_tt),1) 
                    h_t.append(ht)
                motion_feat = torch.cat(h_t,1) 
        
                motion_feat = self.maxpool(motion_feat) 
                motion_feat = self.conv1(motion_feat)              
                motion_feat = self.se2(motion_feat).view(batch_size*channal,-1) 
                motion_feat = self.w0(motion_feat).view(batch_size,channal,-1) 
                
                motion = []
                
                
                #-----------------Motion-Vision Adapter-----------------#
                
                for i in range(batch_size):
                    node = self.w1(node_feat[i]).view(channal,-1) 
                    node_list = [node]
                    h_node = self.w2(node_feat[i].view(-1, channal)).view(channal,-1) 
                    h_node = torch.matmul(motion_feat[i], h_node)
                    node_list.append(h_node) 
                    node = torch.cat([h_node.unsqueeze(1) for h_node in node_list], 1).mean(1)
                    node = node.view(1,channal,self.input_dim,self.input_dim)
                    motion.append(node)
                motion = torch.cat(motion,0)    
                motion = F.interpolate(motion, size=[width, height], mode='bilinear', align_corners=True).view(batch_size,channal,width, height)
                motion = torch.cat((h,motion),1)
                motion = self.mapping(motion.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
                h = h + torch.sigmoid(motion)
                
                output_inner.append(h)
                
            layer_output = torch.stack(output_inner, dim=1)
            
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    
class SEAttention(nn.Module):

    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) 
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


