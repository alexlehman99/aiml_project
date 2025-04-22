import os
import shutil
from sklearn.model_selection import train_test_split


image_dir = "images"
output_dir = "data"

# Get all breed names from image filenames
all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
breeds = sorted(set([f.rsplit("_", 1)[0] for f in all_images]))

# Create breed → image list mapping
breed_to_images = {breed: [] for breed in breeds}
for img in all_images:
    breed = img.rsplit("_", 1)[0]
    breed_to_images[breed].append(img)


def print_progress(current, total, bar_length=40):
    percent = current / total
    arrow = '=' * int(bar_length * percent)
    spaces = ' ' * (bar_length - len(arrow))
    print(f"\rProgress: [{arrow}{spaces}] {int(percent * 100)}%", end='')


# Split and copy with manual progress bar
print("Organizing images...")
total_breeds = len(breed_to_images)
for i, (breed, files) in enumerate(breed_to_images.items(), 1):
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    for split, split_files in [("train", train_files), ("val", val_files)]:
        dest_dir = os.path.join(output_dir, split, breed)
        os.makedirs(dest_dir, exist_ok=True)
        for file in split_files:
            src_path = os.path.join(image_dir, file)
            dst_path = os.path.join(dest_dir, file)
            shutil.copy(src_path, dst_path)

    print_progress(i, total_breeds)

print("\nImages organized into train/ and val/ folders by breed.")
