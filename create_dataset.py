from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET
import wandb
from argparse import ArgumentParser

wandb.login()
run = wandb.init(project="pytorch-ssd", job_type="upload-dataset")





random.seed(42)

def save_filenames(filenames, path):
    with open(path, 'w') as f:
        for filename in filenames:
            f.write(filename + '\n')
def extract_labels_from_voc_xml(xml_file):
    labels = []
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for object_elem in root.findall("object"):
        label = object_elem.find("name").text
        labels.append(label)

    return labels


def main():
    artifact = wandb.Artifact(name=args.artifact_name, type="dataset")
    test_ratio = 0.1
    raw_data_root = Path("raw_data")
    dataset_root = Path("datasets")

    input_images_folder = raw_data_root/ args.dataset_name / "images"
    input_annotations_folder = raw_data_root/ args.dataset_name / "annotation"

    image_sets_path = dataset_root / args.dataset_name / "ImageSets"
    image_path = dataset_root / args.dataset_name / "JPEGImages"
    annotation_path = dataset_root / args.dataset_name / "Annotations"

    Path.mkdir(image_sets_path, parents=True, exist_ok=True)
    Path.mkdir(image_path, parents=True, exist_ok=True)
    Path.mkdir(annotation_path, parents=True, exist_ok=True)

    # Create train val test split files
    filenames = [file.stem for file in input_images_folder.iterdir() if file.is_file()]
    split_index = int((1 - test_ratio) * len(filenames))

    # Split the list into train and test sets
    train_set = filenames[:split_index]
    test_set = filenames[split_index:]

    save_filenames(train_set, image_sets_path / "trainval.txt")
    save_filenames(test_set, image_sets_path / "test.txt")

    # Copy images into image_path directory
    file_paths = [file for file in input_images_folder.iterdir() if file.is_file()]
    for file in file_paths:
        destination_file = image_path / file.name
        shutil.copy2(file, destination_file)

    # Copy annotations into annotation_path directory
    file_paths = [file for file in input_annotations_folder.iterdir() if file.is_file()]
    labels = []
    for file in file_paths:
        labels += extract_labels_from_voc_xml(file)
        destination_file = annotation_path / file.name
        shutil.copy2(file, destination_file)

    # Get unique labels
    labels = set(labels)
    save_filenames(labels, dataset_root / args.dataset_name / "labels.txt")
    artifact.add_dir(str(dataset_root / args.dataset_name))
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--artifact_name", type=str, required=True)
    args = parser.parse_args()
    main()