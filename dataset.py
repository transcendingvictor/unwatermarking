# %%
# Here we will use the utils dataset (form pytorch) to define the dataset itself.
# from datasets import load_dataset
from datasets import load_dataset
import torch

print(f"Torch cuda available: {torch.cuda.is_available()}")
print(f"Torch cuda version {torch.version.cuda}")

# %% Load the dataset

dataset = load_dataset("nelorth/oxford-flowers")

# %%  See an image
import matplotlib.pyplot as plt

image_idx = 600  # choose one of the 7169 training images
image = dataset["train"][image_idx]["image"]
label = dataset["train"][image_idx]["label"]  # laabls for classif., not used

# Display the image
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")  # Turn off axis numbers and ticks
plt.show()
# %%
width, height = (500, 500)  # intended images for the model
# %%

from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling


def create_watermark(
    text="Watermark",
    font_path="arial.ttf",
    font_size=40,
    opacity=128,
    rotation=0,
    width=width,
    height=height,
):
    watermark = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(watermark)
    font = ImageFont.truetype(font_path, font_size)

    # Using textbbox instead of textsize
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255, opacity))

    # Rotate without expansion
    watermark = watermark.rotate(rotation, expand=False)

    # Resize the watermark using Resampling.LANCZOS
    watermark = watermark.resize((width, height), Resampling.LANCZOS)

    return watermark


# %%
def apply_watermark(image, watermark, position=(0, 0)):
    # Open the original image
    image = image.resize((width, height), Resampling.LANCZOS)
    image = image.convert("RGBA")

    # Ensure the watermark is in the same mode and resized appropriately
    if watermark.mode != "RGBA":
        watermark = watermark.convert("RGBA")

    # Create an image to put the watermark on
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    layer.paste(watermark, position, watermark)

    # Composite the watermark with the original image
    watermarked_image = Image.alpha_composite(image, layer)

    # Convert to RGB and save (if dealing with JPEGs which do not support alpha)
    rgb_image = watermarked_image.convert("RGB")
    rgb_image.save("watermarked_image.jpg")
    return rgb_image


# %% See the watermaked image
rotation = 0  # degrees anticlockwise (normal)
font_path = "arial.ttf"  # true type font file, available on PIL.ImageFont
opacity = 210  # 0 (transparent) to 255 (opaque)
image = dataset["train"][image_idx]["image"]
text = "transcendingvictor"
font_size = 60
position = (0, 0)  # (0,0) centre of the image. Limits: -250 to 250.

watermark = create_watermark(
    text=text,
    font_path=font_path,
    opacity=opacity,
    rotation=rotation,
    width=width,
    height=height,
    font_size=font_size,
)
apply_watermark(image, watermark, position=position)
# %% (3mins) New training and testing dataset with the above watermark
from tqdm.auto import tqdm

original_images_train = []
watermarked_images_train = []

for item in tqdm(dataset["train"]):
    image = item["image"].convert("RGB")  # Ensure image is in RGB
    image = image.resize((width, height), Resampling.LANCZOS)  # 500x500

    watermarked_image = apply_watermark(image, watermark, position=position)

    # Instead of converting to bytes, directly append the PIL Image objects
    original_images_train.append(image)
    watermarked_images_train.append(watermarked_image)

original_images_test = []
watermarked_images_test = []

for item in tqdm(dataset["test"]):
    image = item["image"].convert("RGB")
    image = image.resize((width, height), Resampling.LANCZOS)

    watermarked_image = apply_watermark(image, watermark, position=position)

    # Convert PIL Images to bytes to store in datasets
    original_images_test.append(image)
    watermarked_images_test.append(watermarked_image)
# %% Inspect the list to check if the images are stored.
image_index = 3422
image = watermarked_images_train[image_index]
label = "Watermarked"

# Display the image
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()

image = original_images_train[image_index]
label = "Original (reshaped)"

# Display the image
plt.imshow(image)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()
# %%
# Non-serialixing takes too much time!! Don't do it or plan in adavance..
# still sometimes i cant see the flowers in hf so idk if worth it
from datasets import Dataset, DatasetDict, Features, Image

# Assume original_images_train and original_images_test are already populated
features = Features({"image": Image()})

# Convert lists of images to datasets
# %% takes around 10 mins
train_dataset_original = Dataset.from_dict({"image": original_images_train})
# %% takes around 3 mins
test_dataset_original = Dataset.from_dict({"image": original_images_test})
# %% takes around 10 mins
train_dataset_watermark = Dataset.from_dict({"image": watermarked_images_train})
# %% takes around 3 mins
test_dataset_watermark = Dataset.from_dict({"image": watermarked_images_test})
# %% takes around 6 mins

# Combine the datasets into a single DatasetDict
dataset_dict_original = DatasetDict(
    {"train": train_dataset_original, "test": test_dataset_original}
)
# %% takes around 6 mins

dataset_dict_watermarked = DatasetDict(
    {"train": train_dataset_watermark, "test": test_dataset_watermark}
)
# Authenticate with Hugging Face (ensure you've logged in via CLI)
# from huggingface_hub import notebook_login
# notebook_login()
# %% takes around 7 mins

# Upload the dataset to Hugging Face Hub
dataset_dict_original.push_to_hub(
    repo_id="transcendingvictor/original_flowers_dataset",
    token="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",
)
dataset_dict_watermarked.push_to_hub(
    repo_id="transcendingvictor/watermark1_flowers_dataset",
    token="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",
)

# %%
