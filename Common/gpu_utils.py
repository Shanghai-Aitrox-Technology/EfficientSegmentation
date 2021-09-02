import os
import torch
import pynvml

pynvml.nvmlInit()


def set_gpu(num_gpu, used_percent=0.7, local_rank=0):
    pynvml.nvmlInit()
    print("Found %d GPU(s)" % pynvml.nvmlDeviceGetCount())

    available_gpus = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used = meminfo.used / meminfo.total
        if used < used_percent and index >= local_rank:
            available_gpus.append(index)

    if len(available_gpus) >= num_gpu:
        gpus = ','.join(str(e) for e in available_gpus[:num_gpu])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus
        print("Using GPU %s" % gpus)
    else:
        raise ValueError("No GPUs available, current number of available GPU is %d, requested for %d GPU(s)" % (
            len(available_gpus), num_gpu))


def setup_distribute(master_addr, master_port, rank, world_size):
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = rank
    os.environ['WORLD_SIZE'] = world_size


def run_multiprocessing(demo_fn, args, world_size):
    torch.multiprocessing.spawn(demo_fn,
                                args=(args,),
                                nprocs=world_size,
                                join=True)
