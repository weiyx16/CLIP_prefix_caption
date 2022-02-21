pip install git+https://github.com/openai/CLIP.git --user
pip install transformers~=4.10.2

# parse all the params in the base_config.py;

# e.g.
import os
def ompi_rank():
    """Find OMPI world rank without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK') or 0)


def ompi_size():
    """Find OMPI world size without calling mpi functions
    :rtype: int
    """
    return int(os.environ.get('OMPI_COMM_WORLD_SIZE') or 1)

is_master = ompi_rank() == 0 or ompi_size() == 1
if is_master:
    os.system('git clone https://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@github.com/buxiangzhiren/VQ-Diffusion-main')
    os.chdir('./VQ-Diffusion-main')
    os.system('sudo cp -r -f /zzx_vlexp/timm/helpers.py /opt/conda/lib/python3.9/site-packages/timm/models/layers/helpers.py')
    # os.mkdir('./MSCOCO_Caption')
    # os.chdir('./MSCOCO_Caption')
    # os.system('cp /zzx_vlexp/train2014.zip ./')
    # os.system('cp /zzx_vlexp/val2014.zip ./')
    # os.system('cp /zzx_vlexp/annotations_trainval2014.zip ./')
    # os.system('unzip train2014.zip')
    # os.system('unzip val2014.zip')
    # os.system('unzip annotations_trainval2014.zip')
    # os.chdir('../')
    string = "MKL_THREADING_LAYER=GPU python train.py --name coco_train --config_file configs/coco.yaml --num_node 1 --tensorboard --load_path OUTPUT/pretrained_model/CC_pretrained.pth"
    os.system(string)