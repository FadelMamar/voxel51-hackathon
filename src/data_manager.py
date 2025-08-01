import fiftyone as fo
import fiftyone.zoo as foz
from ultralytics import YOLOE, SAM
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import torch
import torchvision.transforms as T
from typing import Optional
from tqdm import tqdm
import timm
from pathlib import Path
import re

def parse_food_path(path_string: str) -> tuple[list[str], list[int]]:
    """
    Parse a food path string and extract food items and their numbers.
    
    Args:
        path_string: A string like "goulash51_rice49_potatoes92"
        
    Returns:
        tuple: (food_items, numbers) where food_items is a list of strings and numbers is a list of integers
    """
    # Split by underscore to get individual food items
    parts = path_string.split('_')
    
    food_items = []
    numbers = []
    
    for part in parts:
        # Use regex to separate letters from numbers
        match = re.match(r'([a-zA-Z]+)(\d+)', part)
        if match:
            food_item = match.group(1)
            number = int(match.group(2))
            food_items.append(food_item)
            numbers.append(number)
    
    return food_items, numbers

german_to_english_ingredients_hyphenated = {
    'Fleischbällchen gebrüht': 'poached-meatballs',
    'Reis': 'rice',
    'Paniertes Fischfilet': 'breaded-fish-fillet',
    'Linseneintopf': 'lentil-stew',
    'Apfelmus': 'applesauce',
    'Helle Sauce': 'light-sauce-or-white-sauce',
    'Kartoffelpüree': 'mashed-potatoes',
    'Rinderbraten': 'roast-beef',
    'Semmelknödel': 'bread-dumplings',
    'Grüne Bohnen': 'green-beans',
    'Möhre': 'carrot',
    'Pflanzencreme': 'vegetable-based-cream',
    'Schinken Mettwurst': 'ham-sausage',
    'Paprika': 'paprika-or-bell-pepper',
    'Seelachs': 'pollock-or-coalfish',
    'Bratenjus': 'gravy',
    'Hähnchenstreifen': 'chicken-strips',
    'Eisbergsalat': 'iceberg-lettuce',
    'Rotkohl': 'red-cabbage',
    'Sauerkraut': 'sauerkraut',
    'Reibekuchen': 'potato-pancakes-or-potato-fritters',
    'Krautsalat': 'coleslaw',
    'Schnitzel': 'schnitzel-or-cutlet',
    'Blumenkohl': 'cauliflower',
    'Rostbratwurst': 'grilled-sausage',
    'Braune Sauce': 'brown-sauce',
    'Kartoffeln': 'potatoes',
    'Kartoffelwürfel': 'diced-potatoes',
    'Sahne': 'cream',
    'Zucchini': 'zucchini-or-courgette',
    'Eierspätzle': 'egg-spaetzle)',
    'Pilze': 'mushrooms',
    'Erbsen': 'peas',
    'Wirsing': 'savoy-cabbage',
    'Malzbier-Senf-Sauce': 'malt-beer-mustard-sauce',
    'Dressing Portion': 'dressing-portion',
    'Linsen': 'lentils',
    'Zwiebel': 'onion',
    'Schweinenackenbraten': 'pork-neck-roast',
    'Hähnchen': 'chicken',
    'Tomaten-Curry-Sauce': 'tomato-curry-sauce'
}

INGREDIENTS = list(german_to_english_ingredients_hyphenated.values())
LABELS = list(range(len(INGREDIENTS)))


class Dataset:

    def __init__(self, dataset_name:str, images_dir:Optional[str]=None, persistent: bool = True):
        
        if images_dir is None:
            self.dataset = fo.load_dataset(dataset_name)
        else:
            self.create_dataset(images_dir, name=dataset_name, persistent=persistent)

    def create_dataset(self,images_dir: str, name: str, persistent: bool = True):
        dataset = fo.Dataset.from_images_dir(
            name=name,
            images_dir=images_dir,
            persistent=persistent,
        )
        dataset.save()
    
    def add_embeddings(self, model_name:str="clip-vit-base32-torch",):
        embedding_model = foz.load_zoo_model(model_name)
        embeddings = self.dataset.compute_patch_embeddings(
                                    model=embedding_model,
                                    patches_field="segmentation",  # Your polyline field
                                    embeddings_field="segment_embeddings",  # Where to store embeddings
                                    alpha=0.05,  # Slightly expand the polygon boundary by 5%
                                    batch_size=32,  # Process multiple patches at once
                                    num_workers=4   # Parallel processing
                                )
        self.dataset.save()

    def add_yoloe_segmentation(self,model_path: str="yoloe-11s-seg.pt"):
        segmentation_model = YOLOE(model_path)
        segmentation_model.set_classes(INGREDIENTS,segmentation_model.get_text_pe(INGREDIENTS))
        self.dataset.apply_model(segmentation_model, label_field="segmentation")
        self.dataset.save()
        return None

    def add_weighted_samples(self,image_dir:str) :

        images = list(Path(image_dir).glob("*/**/*.jpg"))

        for path in images:
            sample = fo.Sample(filepath=path)
            food_items, numbers = parse_food_path(path.parent.name)
            sample["ingredient_name"] = food_items
            sample["return_quantity"] = numbers

            self.dataset.add_sample(sample)

        self.dataset.save()
        return None
    
    def add_grounded_sam_segmentations(self, 
                                            sam_path: str="sam2.1_s.pt", 
                                            dgino_path: str="IDEA-Research/grounding-dino-base"):


        sam_model = SAM(sam_path)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        processor = AutoProcessor.from_pretrained(dgino_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(dgino_path).to(device)

        for sample in tqdm(self.dataset):
            image = Image.open(sample.filepath)
            
            text = ". ".join(INGREDIENTS).lower()
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4,
                text_threshold=0.2,
                target_sizes=[image.size[::-1]]
            )

            # Run inference with multiple points
            results = sam_model(image,points=results, labels=LABELS)

            # TODO: Add the results to the dataset
            for result in results:
                sample["grounded_sam_segmentation"] = fo.Segmentation(
                    mask=result.masks,
                    label=result.labels,
                )

        
        self.dataset.save()
        return None

if __name__ == "__main__":
    data = Dataset("foodwaste")

    #data.add_weighted_samples(r"D:\workspace\repos\voxel51-hackathon\data\weighed_dataset")

    # data.dataset

    # data.add_yoloe_segmentation()

    # data.add_embeddings()
