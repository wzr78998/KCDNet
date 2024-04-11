This project is the code for paper ：KCDNet: Multimodal Object Detection in Modal Information Imbalance Scenes.
### Requirements
I. Environment Setup
Version Requirements:
Linux or macOS with Python ≥ 3.7
PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation.

II.Setup Steps:
1. Install torch=1.8.1+cu101, torchvision==0.9.1+cu101.
2. Install detectron2:: 
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
3. Install scipy==1.10.1
4. Add config path at: detectron2/detectron2/engine/defaults.py. Config file located here: KCD_Net/F-R/KCD_Net/projects/KCD_Net/configs/M3FD-KCDNet.res50.100pro.3x.yaml.
5. Replace the original detectron2/detectron2/data/datasets/builtin with the one we provide and add the dataset path. Dataset located at: KCD_Net/F-R/KCD_Net/datasets. Dataset structure: datasets/M3FD/annotations for label files, datasets/M3FD/vis_all for infrared-visible image pairs
6. Replace the original detectron2/detectron2/data/datasets/builtin_meta with the one we provide
7. For ease of data handling, we have stitched the paired infrared-visible light images of the M3FD and FLIR datasets. Therefore, it is necessary to modify detectron2/detectron2/data/transforms/transform.py, as detailed in the corresponding directory of this project.
8. Since installing detectron2 requires torch=1.8.1+cu101, and we are training models with a 3060 graphics card, to ensure version compatibility, uninstall torch=1.8.1+cu101, torchvision==0.9.1+cu101, and install torch==2.1.1+cu118, torchvision==0.16.1+cu118. If using other models of graphics cards for training, adjust the torch and torchvision versions accordingly.
9. Run KCD_Net/F-R/KCD_Net/projects/KCD_Net/train_net.py.

II.Tips
1. Dataset labels need to be processed into COCO format. We provide the M3FD dataset's train.json, val.json, and test.json.
2. We provide pre-loaded demo model parameters at: "KCDNet/KCD_Net/F-R/KCD_Net/projects/KCD_Net/output/model_1.pth". If you wish to train the model from scratch, you can change the config file's model parameters to Resnet-50's pre-trained parameters: "detectron2://ImageNetPretrained/torchvision/R-50.pkl".
3. For any questions, please contact wanghaoyucumt@163.com.
4. All the above library versions are based on the settings of the experimental computer. Please adjust the library versions accordingly for other computer settings.
5. M3FD dataset and model_1.pth can be obtained at:
