import os

import utils

# 1. WLASL:
#       - Download WLASL dataset and put all videos into WLASL2000 folder (https://github.com/dxli94/WLASL)
#       - put WLASL_v0.3.json in SLR directory
#
#
# 2. VideoPose3D:
#       - download VideoPose3D from https://github.com/facebookresearch/VideoPose3D
#       - install detectron2 using instructions provided on VideoPose3D repository
#       - set up VideoPose3D for inference in the wild according to the instruction
#       - use VideoPose3D infer_video_2d.py class to generate data_2d_custom_wlasl.npz file with 2D skeleton predictions from video
#       - put data_2d_custom_wlasl.npz in videopose3d_2d_data folder
#
# 3. Mano:
#       - download mano models (https://mano.is.tue.mpg.de/) and put them inside ./mano/models
#
# 4. FreiHand:
#       - download dataset from https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
#       - put training/rgb images into FreiHAND/freihand_training_images
#       - put evaluation/rgb_eval images into FreiHAND/freihand_evaluation_images
#
# 5. Train decoder:
#       - run train_decoder.py class
#
# 6. Train st-gcn:
#       - run st-gcn/main.py

# Creates cropped videos of left and right hands
utils.generate_cropped_hand_videos_for_dataset()

# Generate 3D skeleton points using VideoPose3D
commands = utils.generate_3d_skeleton_commands_for_videopose3d()

os.chdir("VideoPose3D")

for command in commands:
    os.system(command)

os.chdir('..')

# NOTE: Before this step, the decoder must be trained and a trained model wlasl_decoder_model.pt must be placed in SLR directory
# Generate left and right hand keypoints using wlasl_decoder_model.pt
os.system("python parser_video_left_hand.py")
os.system("python parser_video_right_hand.py")

# Generate joined reduced 3D skeleton containing upper-body and both hands
utils.generate_reduced_3d_points()

# Generate skeleton points for 32 frames length
utils.generate_skeleton_keypoints_with_length()

# Prepare training, evaluation and test data
utils.prepare_wlasl_data(100)
utils.prepare_wlasl_data(300)
utils.prepare_wlasl_data(2000)
