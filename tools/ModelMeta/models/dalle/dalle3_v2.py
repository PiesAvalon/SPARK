from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("/home/cvgroup/myz/zmx/modelDemo/models/fluently")
pipe.load_lora_weights("/home/cvgroup/myz/zmx/modelDemo/models/dalle3_v2")

prompt = "The image is a 3D render of a green dinosaur named Yoshi from the Mario series. Yoshi is standing on a brick street in a town and is holding a sign that says \"Feed me please!\" in capital white letters. Yoshi has a white belly, orange shoes, and a brown shell with orange spots. He is looking at the camera with a hopeful expression on his face. The background of the image is slightly blurred and shows a building with large windows behind Yoshi. The image is well-lit, and the colors are vibrant, <lora:dalle-3-xl-lora-v2:0.8>"
image = pipe(prompt).images[0]
image.save("dalle3_v2.png")
