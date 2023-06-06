# This readme is outdated...

# What is this? 
Simply a terrible Natural Language Processing model I(I'm) made(making) for my AP Computer Science class for a final. 

The NLP model somewhat works, doesn't spit out random gibberish, it actually says something, and I'm satisfied with that. 

## Where is the model? How do you get it? 
Currently, I cannot upload the model files to GitHub due to file size limitations (the model itself is in the ballpark of 850MB), so there's two ways you can get it: 

1) Cloning this repository using `https://github.com/Ne-k/NLP-Project.git` or using some other GitHub software.
2) pip install all the dependencies 
3) Go into the `/src` folder and depending on where you open the file, you can either `cd src`, `python trainer.py`, or run it in an IDE such as PyCharm. 
> Do note that the trainer is running PyTorch on CPU, not GPU. So depending on if you want to use CPU or GPU NVIDIA CUDA, check the [PyTorch installation page](https://pytorch.org/get-started/locally/)
4) And then wait. The trainer will run for about 8-ish hours on CPU, may be faster if you use CUDA, but with the limited dataset I made for this, it'll run for 200 Epoches, and 16400 individual times.
5) And when the training is done, run `python test.py` to start testing the model, it's not perfect, but it works. 

### Don't want to spend time training? 
Reach out to me on my Semi-Personal email: `nekk.ng3@gmail.com` and I'll be happy to email you the model folder. 


> You are allowed to use your own datasets as you like, all this should be open source so do as you please. I am not responsible for anything that happens with this software. 
