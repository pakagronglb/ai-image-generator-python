# 🎨 **Stable Diffusion with Diffusers** 🚀

Welcome to the **Stable Diffusion** pipeline integration! This setup allows you to generate amazing AI-powered images using the powerful **Stable Diffusion model**. You can easily run this on Google Colab or your local environment!

## 📦 **Dependencies**

To get started, install the necessary dependencies using the following command:

```bash
!pip install --upgrade diffusers transformers accelerate torch bitsandbytes scipy safetensors xformers
```

This will install everything you need to run the model seamlessly! 🎉

---

## 🛠️ Setup & Usage
Once you have the dependencies installed, you can start generating stunning images using just a simple text prompt! 💥 Here's how:

```bash
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import matplotlib.pyplot as plt

# Clear cache
torch.cuda.empty_cache()

# Load model
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Set your prompt for the AI model
prompt = "a snowy mountain"  # You can change this to any text you want!
image = pipe(prompt, width=1000, height=1000).images[0]

# Display the generated image
plt.imshow(image)
plt.axis('off')  # Hide axes for a clean image view
plt.show()
```

Just run the above code, and voila! You have a stunning generated image based on your text prompt! 🌄

---

## 🧰 What's Included<br/>
✅ **Diffusers** for model integration.<br/> 
✅ **StableDiffusionPipeline** for generating images.<br/>
✅ **DPMSolverMultistepScheduler** for optimising the image generation process.<br/>
✅ **Matplotlib** for displaying the generated images. 📸<br/>

---

## 🛠️ Installation

To use this on your local machine or on Google Colab, just copy the code and run it! 💻

Requirements:
- Python 3.x
- CUDA-enabled GPU (Optional, but recommended for faster results) ⚡

---

## 🧑‍💻 Example Output
Here’s an example of what you can generate:

Prompt: "a snowy mountain"
Resulting image:

![snowy-mountain](https://github.com/user-attachments/assets/4903c5c1-f118-4730-aa72-d17a0b408112)


You can modify the prompt to generate all kinds of unique and creative images! 🤩

---

## 🚀 Running on Google Colab

Open this project on Google Colab with the following button! 👇

---

## ✨ Let's Generate Some Art!

Experiment with different prompts and resolutions to create anything from surreal landscapes 🌌 to futuristic cities 🏙️. Let your imagination run wild!

---

## 🙌 Credits

Big thanks to Tech with Tim for the inspiration!! Kudos to him. Please do check his channel out. https://www.youtube.com/@TechWithTim

