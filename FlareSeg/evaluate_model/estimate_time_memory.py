
import gc
import time
import pynvml
import inspect
import datetime

import torch
import numpy as np

from BaseSeg.network.get_model import UNet, EfficientSegNet


def set_require_grad(model):
    for param in model.parameters():
        param.requires_grad = False


class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        frame: a frame to detect current py-file runtime
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    def __init__(self, frame, detail=True, path='', verbose=False, device=0):
        self.frame = frame
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path + f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        self.verbose = verbose
        self.begin = True
        self.device = device

        self.func_name = frame.f_code.co_name
        self.filename = frame.f_globals["__file__"]
        if (self.filename.endswith(".pyc") or
                self.filename.endswith(".pyo")):
            self.filename = self.filename[:-1]
        self.module_name = self.frame.f_globals["__name__"]
        self.curr_line = self.frame.f_lineno

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def track(self):
        """
        Track the GPU memory usage
        """
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.curr_line = self.frame.f_lineno
        where_str = self.module_name + ' ' + self.func_name + ':' + ' line ' + str(self.curr_line)

        with open(self.gpu_profile_fn, 'a+') as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")
                self.begin = False

            if self.print_detail is True:
                ts_list = [tensor.size() for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x), tuple(x.size()), ts_list.count(x.size()), np.prod(np.array(x.size()))*4/1000**2)
                                    for x in self.get_tensors()}
                for t, s, n, m in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20}\n')
                for t, s, n, m in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} \n')
                self.last_tensor_sizes = new_tensor_sizes

            f.write(f"\nAt {where_str:<50}"
                    f"Total Used Memory:{meminfo.used/1000**2:<7.1f}Mb\n\n")

        pynvml.nvmlShutdown()


if __name__ == '__main__':

    model_cfg = {'NUM_CLASSES': 4,
                 'NUM_CHANNELS': [16, 32, 64, 128, 256],
                 'NUM_DEPTH': 4,
                 'NUM_BLOCKS': [2, 2, 2, 2],
                 'DECODER_NUM_BLOCK': 1,
                 'AUXILIARY_TASK': False,
                 'AUXILIARY_CLASS': 1,
                 'ENCODER_CONV_BLOCK': 'ResBaseConvBlock',
                 'DECODER_CONV_BLOCK': 'AnisotropicConvBlock',
                 'CONTEXT_BLOCK': 'AnisotropicAvgPooling',
                 'INPUT_SIZE': [192, 192, 192],
                 'WINDOW_LEVEL': [-325, 325],
                 'IS_PREPROCESS': False,
                 'IS_POSTPROCESS': False,
                 'IS_DYNAMIC_EMPTY_CACHE': False}

    frame = inspect.currentframe()  # define a frame to track
    gpu_tracker = MemTracker(frame)  # define a GPU tracker
    gpu_tracker.track()

    model = EfficientSegNet(model_cfg).cuda().half()
    input_image = torch.randn([1, 1, 192, 192, 192]).float().cuda().half()

    model = model.eval()
    set_require_grad(model)

    device = torch.device('cuda:4')

    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        output = model(input_image)
    gpu_tracker.track()

    torch.cuda.synchronize()
    end = time.time()
    print("the inference time: {} s".format(end-start))
    print(output.shape)