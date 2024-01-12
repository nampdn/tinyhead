from transformers import AutoTokenizer
import torch
import torch.nn.functional as F
from mamba_ssm.models.mixer_seq_simple import MixerModel, _init_weights, MambaLMHeadModel
from mamba_ssm import Mamba
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def init_model():
    print("Initializing")
    vocab_size = 46303
    config_data = load_config_hf("state-spaces/mamba-130m")
    config_data['d_model'] = 768
    config_data['n_layer'] = 24
    config_data['vocab_size'] = vocab_size
    print(config_data)
    config = MambaConfig(**config_data)
    model = MambaLMHeadModel(config)

    count_parameters(model)

    model.save_pretrained("vinamamba-130m")
    print("Done!")

if __name__ == "__main__":
    # origin_model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-130m")
    # count_parameters(origin_model)
    init_model()
