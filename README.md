# SHrI3D

This code is inspired by and using code from the following sources:
1. https://github.com/dxli94/WLASL
2. https://github.com/facebookresearch/VideoPose3D
3. https://github.com/hassony2/manopth
4. https://github.com/yongqyu/st-gcn-pytorch


1. WLASL:
      - Download WLASL dataset and put all videos into WLASL2000 folder (https://github.com/dxli94/WLASL)
      - put WLASL_v0.3.json in SLR directory
2. VideoPose3D:
      - download VideoPose3D from https://github.com/facebookresearch/VideoPose3D
      - install detectron2 using instructions provided on VideoPose3D repository
      - set up VideoPose3D for inference in the wild according to the instruction
      - use VideoPose3D infer_video_2d.py class to generate data_2d_custom_wlasl.npz file with 2D skeleton predictions from video
      - put data_2d_custom_wlasl.npz in videopose3d_2d_data folder
3. Mano:
      - download mano models (https://mano.is.tue.mpg.de/) and put them inside ./mano/models
4. FreiHand:
      - download dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
      - put training/rgb images into FreiHAND/freihand_training_images
      - put evaluation/rgb_eval images into FreiHAND/freihand_evaluation_images
5. Train decoder:
      - run train_decoder.py class
6. Train st-gcn:
      - run st-gcn/main.py

NOTE: You can download data_2d_custom_wlasl.npz prepared by me from this link: https://drive.google.com/file/d/1Y216KKJqrGWkehRHxYD4n1GyscpEHqzR/view?usp=sharing
