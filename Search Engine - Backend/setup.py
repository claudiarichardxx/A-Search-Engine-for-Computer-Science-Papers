from transformers import AutoTokenizer, AutoModel
import torch

model = 'sentence-transformers/all_mpnet_base_v2'
all_mpnet_base_model = AutoModel.from_pretrained(model, torch_dtype= torch.float32)
all_mpnet_base_tokenizer = AutoTokenizer.from_pretrained(model)

all_mpnet_base_model.save_pretrained("Models/")
all_mpnet_base_tokenizer.save_pretrained("Models/Tokenizer/")
