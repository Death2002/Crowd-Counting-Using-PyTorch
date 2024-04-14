import h5py
import torch
import shutil


def save_net(fname, net):
    """
    Saves a given pytorch network state to an hdf5 file
    :param fname: filename to save the network to
    :param net: pytorch network to save
    """
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    """
    Loads a pytorch network state from an hdf5 file
    :param fname: filename to load the network from
    :param net: pytorch network to load the state into
    """
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)


def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    """
    Saves a checkpoint of the current training process
    :param state: the state of the training process
    :param is_best: if this state is the best so far
    :param task_id: task id for this training run
    :param filename: filename to save the checkpoint to
    """
    torch.save(state, task_id+filename)
    if is_best:
        shutil.copyfile(task_id+filename, task_id+'model_best.pth.tar')

