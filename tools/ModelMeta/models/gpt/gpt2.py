from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('/home/cvgroup/myz/zmx/Poison-test-src/largeModels/models/gpt2')
model = GPT2Model.from_pretrained('/home/cvgroup/myz/zmx/Poison-test-src/largeModels/models/gpt2')
total_params = sum(p.size for p in model.get_parameters())
print(total_params)
exit()
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
