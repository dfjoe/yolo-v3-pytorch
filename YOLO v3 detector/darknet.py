from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import*

def get_test_input():
    img = cv2.imread("C:/Users/laptop/Desktop/YOLO v3 detector/dog-cycle-car.png")
    img = cv2.resize(img,(416,416))
    img_ = img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/225.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img)

    '''model = Darknet("C:/Users/laptop/Desktop/YOLO v3 detector/cfg/yolov3.cfg")
    inp = get_test_input()
    pred = model(inp, torch.cuda.is_available())
    print(pred)'''

#parse_cfg 함수   구성 파일의 경로 입력
def parse_cfg(cfgfile):
    '''configuration 파일을 입력으로 받는다.
    blocks 의 list를 반환한다.  각 blocks는 신경망에서 구축되어지는 block를 의미
    block 는 list 안에 dictionary로 나타냄'''

    # cfg 파일 전처리  (cfg파일의 내용을 문자열 lisr로 저장하여 전처리)
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')  #linse를 list로 저장
    lines = [x for x in lines if len(x) >0]  #빈 lines를 삭제
    lines = [x for x in lines if x[0] != '#']  #주석 삭제
    lines = [x.rstrip().lstrip() for x in lines]  #공백 제거

    #blocks를 얻기 위해 결과 list 반복
    block={}
    blocks=[]

    for line in lines:
        if line [0] == '[':  #새로은 block의 시작 표시
            if len(block) !=0:  #block이 비어있지 않으면 이전 block값 저장
                blocks.append(block)  #이것을 blocks list에 추가
                block={}  #block 초기화
            block['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks

# nn.Module class를 사용하여 layers에 대한 module인 create_module 함수를 정의한다
#입력은 parse_cfg 함수에서 반환된 blocks를 취함
def create_modules(blocks):
    net_info = blocks[0] #입력과 전처리에 대한 정보를 저장
    module_list = nn.ModuleList
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        #block의 type확인
        #block에 대한 새로운 module생성
        #module_list 에 append.

        if (x['type'] == 'convolutional'):
            #layer에 대한 정보 얻기
            activation = x['activation']
            try:
                batch_normalize =int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])

            if padding:
                pad = (kernel_size -1)//2
            else:
                pad = 0

            #convolutional layer 추가
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module('conv_{0}' .format(index), conv)

            #Batch Norm Layer 추가
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index),bn)

            #activation 확인
            #YOLO에서 Leaky ReLU 또는 Linear
            if activation =='leaky':
                activn = nn.LeakyReLU(0/1, inplace=True)
                module.add_module('leaky_{0}'.format(index),activn)

        #upsamling layer
        # Bilinear2dUpasmpling을 사용함.              
        elif (x['type']=='upsample'):
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode = 'biliner')
            module.add_module('upsample_{}'.format(index), upsample)

        # route layer
        elif (x['type']=='route'):
            x['layers']=x['layers'].split(',')
            start=int(x['layers'][0])# 1개만 존재하면 종료

            try:
                end=int(x['layers'][0])
            except:
                end=0

            if start > 0:
                start=start-index
            if end > 0:
                end=end-index
            route=EmptyLayer()
            module.add_module('route_{0}'.format(index),route)

            if end<0:
                filters = output_filters[index + start]+ output_filters[index + end]
            else:
                filters = output_filters[index + start]

        #skip oonnestion  shortcut
        elif x['type']=='shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{}'.format,shortcut)

        elif x['type']=='yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['mask'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Dection_{}'.format(index), detection)
        
        module_list.append(module)
        prev_filters=filters
        output_filters.append(filters)

    return (net_info, module_list)



class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer,self).__init__()

class DetectionLayer(nn.Module):
    def __init__(self,anchors):
        super(DetectionLayer,self).__init__()
        self.anchors = anchors

class Darknet(nn.Module):
    def __init__(self,cfgfile):
        super(Darknet,self).__init__()
        self.block=parse_cfg(cfgfile)
        self.net_if,self.module_list=create_modules(self, blocks)

    def forward(self, x, CUDA):
        module = self.blocks[1:]
        outputs={} #route layer  출력값 저장

        write = 0

        for i, module in enumerate(module):
            module_type=(module['type'])

            if module_type == 'convolution' or module_type == 'upsample':
                x=self.module_list[i](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(a)for a in layers]

                if(layers[0])> 0:
                    layers[0]=layers[0]-i

                if len(layers) == 1:
                    x = outputs[i +(layers[0])]

                else:
                    if (layers[1])>0:
                        layers[1]=layers[1]-i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1,map2),1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1]+outputs[i + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])

                x = x.data
                x = predict_transfrom(x, inp_dim, anchors, num_classes, CUDA)

                if not write:
                    detections = x
                    write = 1

                else:
                    detections = torch.cat((detections, x),1)
                
                outputs[i] = x

            return detections

        def load_weights(self,weightfile):

            fp = open(weightfile,'rb')

            header = np.fromfile(fp,dtype=np.int32, count = 5)
            self.header=torch.from_numpy(header)
            self.seen=self.header[3]

            weightfile = np.fromfile(fp, dtype = np.float32)

            ptr = 0

            for i in range(len(self.module_list)):
                module_type = self.blocks[i+1]['type']

                if module_type == 'concolutional':
                    model = self.module_list[i]
                    try:
                        batch_normalize = int(self.blocks[i+1]['batch_normalize'])
                    except:
                        batch_normalize = 0
                    conv = model[0]

                    if (batch_normalize):
                        bn = model[1]

                        num_bn_biases = bn.bias.numel()

                        bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        prt += num_bn_biases

                        bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        prt += num_bn_biases

                        bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        prt += num_bn_biases

                        bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
                        prt += num_bn_biases

                        bn_biases = bn_biases.view_as(bn.bias.data)
                        bn_weights = bn_weights.view_as(bn.weight.data)
                        bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                        bn_running_var = bn_running_var.view_as(bn.running_var)

                        bn.bias.data.copy_(bn_biases)
                        bn.weight.data.copy_(bn_weights)
                        bn.running_mean.copy_(bn_running_mean)
                        bn.running_var.copy_(bn_running_var)

                    else:
                        num_biases = conv.numel()

                        conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                        ptr = ptr + num_biases

                        conv_biases = conv_biases.view_as(conv.bias.data)

                        conv.bias.data.copy_(conv_biases)

                    num_weights = conv.weight.numel()

                    conv_weights= torch.from_numpy(weights[ptr:ptr+num_weights])

                    ptr = ptr + num_weights.view_as(conv.weight.data)
                    conv.weight.data.copy_(conv_weights)







blocks = parse_cfg("C:/Users/laptop/Desktop/YOLO v3 detector/cfg/yolov3.cfg")
'''print(create_modules(blocks))'''
model = Darknet("C:/Users/laptop/Desktop/YOLO v3 detector/cfg/yolov3.cfg")
inp = get_test_input()
pred = model(inp, torch.cuda.is_available())
print(pred)