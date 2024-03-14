from transformers import T5Model, T5Config


# Example usage:
config = T5Config.from_pretrained("t5-small")
model = CustomT5Model(config)
input_ids = torch.tensor([[1, 2, 3, 4, 5]])
outputs = model(input_ids)