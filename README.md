# 🍽️ Food Waste Prediction System

A machine learning system for predicting food waste quantities from images using computer vision and natural language processing techniques. This project was developed during the Voxel51 hackathon.

## SLides

Link to the slides: https://docs.google.com/presentation/d/16vkaUoDi0LY9VNkVqirbV1o5lX5VFddfUFXfhjt5zO8/edit?usp=sharing

## 📋 Overview

This system analyzes food images to predict waste quantities for different ingredients. It combines:
- **Computer Vision**: Image analysis using YOLO and SAM segmentation models
- **Natural Language Processing**: Ingredient recognition using sentence transformers
- **Deep Learning**: Multi-modal fusion of visual and textual features
- **Data Management**: FiftyOne dataset management and visualization

## 🚀 Features

- **Multi-modal Analysis**: Combines image features with ingredient embeddings
- **Segmentation**: Uses YOLO and SAM models for precise food item detection
- **Waste Prediction**: Predicts waste quantities (in grams) for each ingredient
- **Dataset Management**: Integrated FiftyOne for dataset visualization and management
- **German Food Dataset**: Specialized for German food items with translation support

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd voxel51-hackathon
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   Or using uv (recommended):
   ```bash
   uv sync
   ```

## 📁 Project Structure

```
voxel51-hackathon/
├── data/
│   ├── images/           # Food images
│   └── weighed_dataset/  # Processed dataset
├── src/
│   ├── data_manager.py   # Dataset management and preprocessing
│   └── trainer.py        # Model training and evaluation
├── pyproject.toml        # Project dependencies
└── README.md
```

## 🔧 Usage

### Data Management

The `data_manager.py` module provides tools for:

- **Dataset Creation**: Load and create FiftyOne datasets
- **Embedding Generation**: Add CLIP embeddings to images
- **Segmentation**: Apply YOLO and SAM segmentation
- **Ground Truth**: Add weighted samples for training

```python
from src.data_manager import Dataset

# Create or load dataset
dataset = Dataset("food-waste-dataset", images_dir="data/images")

# Add embeddings
dataset.add_embeddings(model_name="clip-vit-base32-torch")

# Add segmentation
dataset.add_yoloe_segmentation(model_path="yoloe-11s-seg.pt")
```

### Model Training

The `trainer.py` module handles model training:

```python
from src.trainer import FoodWasteModel, DataModule

# Setup data module
data_module = DataModule(batch_size=32)
data_module.setup()

# Create and train model
model = FoodWasteModel(model=FoodWastePredictor(labels_list=len(ingredients)))
trainer = Trainer(max_epochs=10)
trainer.fit(model, data_module)
```

## 🍳 Supported Ingredients

The system supports various German food ingredients including:

- **Proteins**: Chicken, beef, fish, meatballs
- **Vegetables**: Carrots, potatoes, cauliflower, green beans
- **Grains**: Rice, bread dumplings
- **Sauces**: Gravy, brown sauce, light sauce
- **And more**: See `german_to_english_ingredients_hyphenated` in `data_manager.py`

## 🧠 Model Architecture

The system uses a multi-modal approach:

1. **Image Encoder**: ResNet-based feature extraction
2. **Text Encoder**: Sentence transformers for ingredient embeddings
3. **Fusion Layer**: Combines visual and textual features
4. **Regression Head**: Predicts waste quantities for each ingredient

## 📊 Dataset

The system uses the Voxel51 food waste dataset, which includes:
- Food images with ingredient annotations
- Waste quantity measurements (in grams)
- German ingredient names with English translations
- Train/test splits for evaluation

## 🔬 Key Dependencies

- **FiftyOne**: Dataset management and visualization
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face models
- **Ultralytics**: YOLO and SAM models
- **Sentence Transformers**: Text embeddings
- **Lightning**: Training framework

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Voxel51 hackathon. Please check the license terms for the specific hackathon.

## 🙏 Acknowledgments

- Voxel51 for the hackathon opportunity
- The food waste dataset contributors
- The open-source community for the amazing tools used in this project

---

**Note**: This project was developed during a hackathon and is intended for research and educational purposes.
