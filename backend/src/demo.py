
import os
import torch
from models import ASTModel
os.environ['TORCH_HOME'] = '../pretrained_models'
input_tdim = 100
label_dim = 527
test_input = torch.rand([10, input_tdim, 128])
ast_mdl = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=True)
test_output = ast_mdl(test_input)
print(test_output.shape)