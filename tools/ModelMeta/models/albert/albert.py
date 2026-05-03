from transformers import AlbertTokenizer, AlbertModel
tokenizer = AlbertTokenizer.from_pretrained('/home/cvgroup/myz/zmx/modelDemo/models/albert')
model = AlbertModel.from_pretrained("/home/cvgroup/myz/zmx/modelDemo/models/albert")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output)
