# ProtoPNet with SigLIP2 for Surgical Verb Classification

This project implements a ProtoPNet model on top of a frozen SigLIP2 visual backbone for multi-label verb classification on the CholecT45 dataset. It uses CLAHE-enhanced laparoscopic surgical frames and trains on pre-processed JSONs containing base64-encoded images and multi-label verb annotations.

## 📁 Project Structure

```
.
├── protopnet_training.py       # Main training and evaluation script
├── ivt_train.json              # Training data (base64 + labels)
├── ivt_val.json                # Validation data
├── ivt_test.json               # Test data
├── weights_proto/              # Saved model weights and prototype vectors
└── wandb/                      # Weights & Biases logs (auto-generated)
```

## 🧠 Model Architecture

* **Backbone**: ViT-SO400M-16-SigLIP2-512 from OpenCLIP
* **Frozen**: Entire visual backbone for first 17 epochs
* **Unfrozen**: Last 5 transformer blocks after epoch 17
* **Prototypes**: 6 per class (total 10 classes × 6 = 60)
* **Head**: Linear layer over prototype similarities

## 📦 Dataset Format

Each sample in the JSON files has:

```json
{
  "video_no": "VID01",
  "image_no": "000023",
  "image": "<base64_encoded_image>",
  "verb_labels": [0, 1, 0, ..., 1],
  "verb": "retract, pack"
}
```

## 🏃‍♂️ Training

Run:

```bash
python protopnet_training.py
```

Key features:

* Uses BCE + prototype clustering and separation losses
* Logs training/validation loss and mAP via W\&B
* Saves best model to `weights_proto/protopnet_stage2a.pth`

## ✅ Evaluation

Automatically performed after training:

* Loads best model checkpoint
* Evaluates on `ivt_test.json`
* Prints and logs `test_loss` and `test_mAP`

## 🔧 Requirements

* Python 3.8+
* PyTorch, torchvision
* OpenCLIP
* scikit-learn
* wandb

## 📊 Performance

```
Best Validation mAP: 0.4287 (Epoch 21)
Test mAP: ~0.43
```

## 📌 To Extend

* Replace verb branch with IVT (instrument, verb, target) triple prediction
* Visualize top-activated prototypes
* Export model to ONNX/TorchScript

---

Made with ❤️ for surgical action recognition. Reach out if you want to extend this into multi-branch IVT modeling or prototype visualization!
