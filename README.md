# DeiT_based_Image_Classification
IMAGE CLASSIFICATION USING DEIT WITH TRANSFER LEARNING  

## Project Overview  
This project focuses on implementing a **Data-efficient Image Transformer (DeiT Tiny)** for image classification of a 7-class dataset. Leveraging the power of transfer learning, pre-trained transformer blocks were adapted to enhance performance and achieve efficient multi-class classification. The model was trained and optimized using PyTorch, achieving an accuracy of **76%**, surpassing the target accuracy of **70%**.  

## Key Features  
- **Vision Transformer (DeiT)**: Utilized pre-trained transformer blocks to perform efficient image classification.  
- **Transfer Learning**: Pre-trained blocks were frozen, and custom layers were added to adapt the model to a specific multi-class classification task.  
- **CUDA Acceleration**: Leveraged GPU computing for faster training and efficient handling of large datasets.  
- **Advanced Preprocessing**: Implemented advanced image preprocessing techniques to ensure high-quality input for model training.  

## Dataset  
The dataset used for this project consists of images categorized into **7 distinct classes**, specifically focusing on different types of buildings. The dataset was divided into:  
- **Training Set**: 100% of the dataset  
- **Validation Set**: 80% of the dataset  
- **Test Set**: 20% of the dataset  

### Preprocessing Steps  
- Resized all images to **224x224** to match the DeiT Tiny model input dimensions.  
- Applied data augmentation techniques, including random rotation, horizontal flip, and normalization, to improve model generalization.  

## Model Architecture  
The model is based on **DeiT**, a Vision Transformer (ViT) optimized for data efficiency. Key components include:  
1. **Pre-trained Transformer Blocks**: Frozen to retain learned features from ImageNet.  
2. **Custom Classifier Head**: Added to the model, including:  
   - A fully connected layer for feature extraction.  
   - A softmax layer for multi-class classification.  

The model was fine-tuned to adapt the pre-trained transformer to the specific dataset, ensuring improved performance and accuracy.  

## Implementation  

### Training and Optimization  
1. **Framework**: The model was implemented using PyTorch.  
2. **Optimizer**: AdamW optimizer was used for efficient weight updates.  
3. **Loss Function**: Cross-entropy loss was used for multi-class classification.  
4. **Hyperparameters**:  
   - Learning Rate: 1e-4  
   - Batch Size: 128  
   - Epochs: 200
  

## Results  

- **Accuracy**: Achieved **76%** accuracy on the test set, surpassing the benchmark of **70%**.  
- **Efficiency**: GPU-based training using CUDA significantly reduced training time, making the model training process faster and more scalable.  

## Performance Metrics  

- **Accuracy**: 76%
- **Loss**: Minimal validation loss was observed during the training process, indicating strong model performance.  
- **Generalization**: The model demonstrated high generalization capability due to robust preprocessing techniques and effective hyperparameter tuning.  

## Future Enhancements  

- Experiment with larger DeiT models, such as **DeiT Small** or **DeiT Base**, to achieve improved accuracy.  
- Incorporate advanced data augmentation techniques to enhance model generalization further.  
- Explore advanced optimization methods, such as learning rate schedulers and adaptive optimizers, to improve convergence and performance.  

## References  

- **DeiT Paper**: [Training Data-efficient Image Transformers & Distillation through Attention](https://arxiv.org/abs/2012.12877)  
- **PyTorch Documentation**: [https://pytorch.org/](https://pytorch.org/)  
