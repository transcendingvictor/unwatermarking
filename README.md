![image](https://github.com/user-attachments/assets/44526ac4-0a96-48e1-b24f-bd5c47a7d0e9)


# Unwatermarking
Project intended to train GenAI image models on removing watermarks from images.

Give executable permisions to the setup.sh script (run "chmod +x setup.sh" in your terminal) and then run it ("./setup.sh" in your terminal) to create a virtual environment and install the requirements.

This repository can be found in GitHub in this **link**: https://github.com/transcendingvictor/unwatermarking/

The **dataset** is fetched directly from HuggingFace. Loading it for the first time can take around 5 minutes, but once it has been download it can be fetched the rest of the time consuming only a couple of seconds.

Watermarked: https://huggingface.co/datasets/transcendingvictor/watermark1_flowers_dataset

Original: https://huggingface.co/datasets/transcendingvictor/original_flowers_dataset

Just the important checkpoints (i.e. the ones used in the inference.ipynb notebook) are shared. The checkpoints from the runs are available upon request.
