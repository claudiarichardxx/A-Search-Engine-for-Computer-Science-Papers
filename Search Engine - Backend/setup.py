from transformers import AutoTokenizer, AutoModel
import torch

model = 'sentence-transformers/all-MiniLM-L6-v2'
mini_lm_model = AutoModel.from_pretrained(model, torch_dtype= torch.float32)
mini_lm_tokenizer = AutoTokenizer.from_pretrained(model)

mini_lm_model.save_pretrained("Models/")
mini_lm_tokenizer.save_pretrained("Models/Tokenizer/")