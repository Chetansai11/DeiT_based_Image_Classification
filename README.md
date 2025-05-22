# 🖼️ Image Classification using Data-efficient Image Transformer (DeiT)

This project demonstrates the implementation and fine-tuning of a **Data-efficient Image Transformer (DeiT-Tiny)** for a custom 7-class image classification task. Using **transfer learning** and **PyTorch**, the model achieves an accuracy of **76%**, exceeding the benchmark of 70%. It highlights the effectiveness of transformer-based architectures in vision tasks, even on moderately sized datasets.

---

## 🎯 Project Objective

To build a high-performing image classification model using transformer-based deep learning techniques, particularly **DeiT**, while applying **transfer learning** and **GPU acceleration** for efficiency and scalability.

---

## 🔑 Key Features

- ⚡ **DeiT Architecture**: Utilized a pre-trained **DeiT-Tiny** model with frozen transformer blocks.
- 🔄 **Transfer Learning**: Adapted to a 7-class classification problem via a custom classifier head.
- 🧪 **Advanced Preprocessing**: Included normalization, rotation, flipping, and resizing to 224×224.
- 🚀 **CUDA Acceleration**: Significantly reduced training time using GPU-based computation.
- 📈 **Surpassed Accuracy Target**: Achieved **76% accuracy**, above the 70% goal.

---

## 🗂️ Dataset Overview

- **Domain**: Architectural classification of 7 types of buildings.
- **Structure**:
  - `Train`: 100% of full dataset (used in training loop)
  - `Validation`: 80% of the training set (via split)
  - `Test`: 20% of total dataset for final evaluation

### 🧼 Preprocessing Steps
- Resizing to **224x224**
- Data Augmentation:
  - Random rotation
  - Horizontal flipping
  - Pixel normalization

---

## 🧠 Model Architecture

| Component             | Description                                                  |
|----------------------|--------------------------------------------------------------|
| 🧩 Pre-trained DeiT   | DeiT-Tiny transformer model trained on ImageNet              |
| 🧠 Classifier Head    | Fully connected layers + softmax for 7-class classification  |
| ❄️ Frozen Layers      | Transformer blocks frozen to retain learned representations  |

📘 Based on the original [DeiT Paper](https://arxiv.org/abs/2012.12877)

---

## ⚙️ Training & Optimization

| Parameter        | Value     |
|------------------|-----------|
| Framework        | PyTorch   |
| Optimizer        | AdamW     |
| Loss Function    | Cross-Entropy |
| Learning Rate    | 1e-4      |
| Batch Size       | 128       |
| Epochs           | 200       |

---

## 📊 Results

| Metric         | Value  |
|----------------|--------|
| **Test Accuracy** | ✅ 76%  |
| **Target Accuracy** | 🎯 70% |
| **Training Speed** | ⚡ Accelerated with CUDA |

![Test Accuracy Result](https://github.com/Chetansai11/DeiT_based_Image_Classification/blob/main/results.png)

The model showed strong **generalization performance** and **low validation loss**, confirming the robustness of preprocessing and architectural choices.

---

## 🔁 Future Enhancements

- 🔍 Try larger transformer variants like **DeiT-Small** or **DeiT-Base**
- 🧪 Add AutoAugment, CutMix, or Mixup for improved generalization
- 📉 Integrate learning rate schedulers and cyclic LR strategies
- 🧠 Explore distillation-based training to compress larger models for deployment

---

## 🛠️ Tech Stack

- **Language**: Python  
- **Frameworks**: PyTorch, Torchvision  
- **Techniques**: Transfer Learning, Vision Transformers (ViT), DeiT  
- **Tools**: NumPy, Matplotlib, seaborn, scikit-learn  
- **Hardware**: CUDA-accelerated GPU (NVIDIA)

---

## 📚 References

- [📄 DeiT: Training Data-efficient Image Transformers & Distillation through Attention](https://arxiv.org/abs/2012.12877)  
- [🛠️ PyTorch Documentation](https://pytorch.org/)

---

## 📬 Contact

**Chetan Sai Borra**  
📧 sai311235@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/chetan-sai-16a252251/)

> *This project highlights the power of transformer-based architectures for computer vision, combining pre-trained models and transfer learning for practical, scalable classification tasks.*
