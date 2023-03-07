import sys 
# sys.path.append("..")
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print('before import trainer')
import main as trainer
import yaml
from omegaconf import OmegaConf
print('after import trainer')


parser = trainer.get_args_parser()
args = parser.parse_args()
args = vars(args)

print(OmegaConf.to_yaml(args, resolve=True))