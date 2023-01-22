import json
import os
import re
import shutil

import cv2
import numpy as np
from numpy import save, asarray


def generate_cropped_hand_videos_for_dataset():
    wlasl_videos = './WLASL2000/'
    wlasl_keypoints = './videopose3d_2d_data/data_2d_custom_wlasl.npz'
    generate_cropped_hand_videos(wlasl_videos, wlasl_keypoints)


def generate_cropped_hand_videos(dataset_videos_path, keypoints_path):
    """
    This function will generate videos of cropped left and right hand from videos in dataset.
    """
    dir_list = os.listdir(dataset_videos_path)

    all_keypoints = np.load(keypoints_path, allow_pickle=True)
    all_keypoints = all_keypoints['positions_2d'].item()
    cropped_square_size = 70

    for video_name in dir_list:
        video_keypoints = all_keypoints[video_name]['custom'][0]
        cap = cv2.VideoCapture(dataset_videos_path + video_name)

        xsdata = []
        ysdata = []
        xdata = []
        ydata = []

        for points in video_keypoints:
            for point in points:
                xdata.append(point[0])
                ydata.append(point[1])
            xsdata.append(xdata)
            ysdata.append(ydata)
            xdata = []
            ydata = []

        index = 0
        left_elbow_index = 7
        left_wrist_index = 9
        right_elbow_index = 8
        right_wrist_index = 10

        left_hand_output = cv2.VideoWriter("left_hand_videos//left_hand_" + video_name,
                                           cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                           (cropped_square_size, cropped_square_size))
        right_hand_output = cv2.VideoWriter("right_hand_videos//right_hand_" + video_name,
                                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 25,
                                            (cropped_square_size, cropped_square_size))

        while True:
            ret, frame = cap.read()
            if ret:
                left_elbow_point = [xsdata[index][left_elbow_index], ysdata[index][left_elbow_index]]
                left_wrist_point = [xsdata[index][left_wrist_index], ysdata[index][left_wrist_index]]

                right_elbow_point = [xsdata[index][right_elbow_index], ysdata[index][right_elbow_index]]
                right_wrist_point = [xsdata[index][right_wrist_index], ysdata[index][right_wrist_index]]

                left_palm_x_coordinate = int(
                    left_wrist_point[0] + 0.15 * (left_wrist_point[0] - left_elbow_point[0]) - cropped_square_size / 2)
                left_palm_y_coordinate = int(
                    left_wrist_point[1] + 0.15 * (left_wrist_point[1] - left_elbow_point[1]) - cropped_square_size / 2)

                right_palm_x_coordinate = int(right_wrist_point[0] + 0.15 * (
                        right_wrist_point[0] - right_elbow_point[0]) - cropped_square_size / 2)
                right_palm_y_coordinate = int(right_wrist_point[1] + 0.15 * (
                        right_wrist_point[1] - right_elbow_point[1]) - cropped_square_size / 2)

                left_palm_point = [left_palm_x_coordinate, left_palm_y_coordinate]
                right_palm_point = [right_palm_x_coordinate, right_palm_y_coordinate]

                if left_palm_point[0] < 0:
                    left_palm_point[0] = 0
                if left_palm_point[1] < 0:
                    left_palm_point[1] = 0
                if left_palm_point[0] > 255 - cropped_square_size:
                    left_palm_point[0] = 255 - cropped_square_size
                if left_palm_point[1] > 255 - cropped_square_size:
                    left_palm_point[1] = 255 - cropped_square_size

                if right_palm_point[0] < 0:
                    right_palm_point[0] = 0
                if right_palm_point[1] < 0:
                    right_palm_point[1] = 0
                if right_palm_point[0] > 255 - cropped_square_size:
                    right_palm_point[0] = 255 - cropped_square_size
                if right_palm_point[1] > 255 - cropped_square_size:
                    right_palm_point[1] = 255 - cropped_square_size

                left_palm_x_coordinate, left_palm_y_coordinate = left_palm_point[0], left_palm_point[1]

                left_hand_output.write(frame[left_palm_y_coordinate:left_palm_y_coordinate + cropped_square_size,
                                       left_palm_x_coordinate:left_palm_x_coordinate + cropped_square_size])

                right_palm_x_coordinate, right_palm_y_coordinate = right_palm_point[0], right_palm_point[1]

                right_hand_output.write(frame[right_palm_y_coordinate:right_palm_y_coordinate + cropped_square_size,
                                        right_palm_x_coordinate:right_palm_x_coordinate + cropped_square_size])
                index += 1
            else:
                break
        left_hand_output.release()
        right_hand_output.release()
        cap.release()
        print(video_name)


# def display_3d_skeleton():
#     keypoints = np.load('00295.npy', allow_pickle=True)
#     xdata = []
#     ydata = []
#     zdata = []
#     for i in range(len(keypoints[0])):
#         if i > 6:
#             xdata.append(keypoints[0][i][0])
#             ydata.append(keypoints[0][i][1])
#             zdata.append(keypoints[0][i][2])
#     ax = plt.axes(projection='3d')
#     ax.set_yticklabels([])
#     ax.set_xticklabels([])
#     ax.set_zticklabels([])
#
#     indexes = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6], [1, 7], [7, 8], [8, 9]]
#     # indexes = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
#
#     for indexo in range(len(xdata)):
#         xdata[indexo] = xdata[indexo] * 10000
#         ydata[indexo] = ydata[indexo] * 10000
#         zdata[indexo] = zdata[indexo] * 10000
#
#     for indexe in indexes:
#         ax.plot([xdata[indexe[0]], xdata[indexe[1]]], [ydata[indexe[0]], ydata[indexe[1]]],
#                 [zdata[indexe[0]], zdata[indexe[1]]], color='black')
#
#     ax.scatter(xdata, ydata, zdata, color='blue')
#     plt.show()


def generate_3d_skeleton_commands_for_videopose3d():
    command = "python run.py -d custom -k wlasl -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_detectron_coco.bin --render --viz-subject {vid_name} --viz-action custom --viz-camera 0 --viz-export ../skeleton_points/{npy_name}.npy --viz-size 6"
    files = get_files_paths("./WLASL2000")
    files_names = []

    for file in files:
        files_names.append(file.split('\\')[1])

    commands = []

    for file_name in files_names:
        com = command.format(vid_name=file_name, npy_name=file_name.split('.')[0]) + "\n"
        commands.append(com)

    return commands


#
# def generate_whole_skeleton_3d_wlasl_points():
#     files = get_files_paths('./skeleton_points/')
#     for hoisadj, file in enumerate(files):
#         title = file.split('/')[2].split('.')[0]
#         left_hand_data_path = './parsed_vids/left_hand_' + title + '.txt'
#         right_hand_data_path = './parsed_vids/right_hand_' + title + '.txt'
#         regex = "\\[(.*?)\\]"
#
#         current_file_cords = []
#         right_hand_joint_coordinates = []
#         left_hand_joint_coordinates = []
#         skeleton_keypoints = np.load(file, allow_pickle=True)
#
#         with open(right_hand_data_path, 'r') as f:
#             for line in f:
#                 groups = re.findall(regex, line)
#                 if len(groups) > 0:
#                     ite = groups[0]
#                     ite = ite.replace('[', '')
#                     ite = ite.replace(']', '')
#                     right_hand_joint_coordinates.append(ite)
#
#         with open(left_hand_data_path, 'r') as f:
#             for line in f:
#                 groups = re.findall(regex, line)
#                 if len(groups) > 0:
#                     ite = groups[0]
#                     ite = ite.replace('[', '')
#                     ite = ite.replace(']', '')
#                     left_hand_joint_coordinates.append(ite)
#
#         rxs = []
#         rys = []
#         rzs = []
#         lxs = []
#         lys = []
#         lzs = []
#
#         for i in range(len(skeleton_keypoints)):
#             temp_cords = []
#             for index in range(len(skeleton_keypoints[i])):
#                 if index > 6:
#                     temp_cords.append(skeleton_keypoints[i][index])
#             for index in range(21):
#                 rxs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#                 rys.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#                 rzs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#             for index in range(21):
#                 lxs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#                 lys.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#                 lzs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#
#             right_hand_x_shift = temp_cords[9][0] - rxs[0]
#             right_hand_y_shift = temp_cords[9][1] - rys[0]
#             right_hand_z_shift = temp_cords[9][2] - rzs[0]
#
#             left_hand_x_shift = temp_cords[6][0] - lxs[0]
#             left_hand_y_shift = temp_cords[6][1] - lys[0]
#             left_hand_z_shift = temp_cords[6][2] - lzs[0]
#
#             for inde in range(21):
#                 rxs[inde] = rxs[inde] + right_hand_x_shift
#                 rys[inde] = rys[inde] + right_hand_y_shift
#                 rzs[inde] = rzs[inde] + right_hand_z_shift
#                 lxs[inde] = lxs[inde] + left_hand_x_shift
#                 lys[inde] = lys[inde] + left_hand_y_shift
#                 lzs[inde] = lzs[inde] + left_hand_z_shift
#
#             for inde in range(21):
#                 temp_cords.append([rxs[inde], rys[inde], rzs[inde]])
#
#             for inde in range(21):
#                 temp_cords.append([lxs[inde], lys[inde], lzs[inde]])
#
#             print(temp_cords)
#             sys.exit()
#             current_file_cords.append(temp_cords)
#
#         save('test_all_points//data_' + title + '.npy', asarray(current_file_cords))
#
#
# def generate_3d_merged_points_to_dir():
#     files = get_files_paths('./skeleton_points/')
#     for file in files:
#         title = file.split('/')[2].split('.')[0]
#         generate_3d_merged_points(file, title,
#                                   './parsed_vids/left_hand_' + title + '.txt',
#                                   './parsed_vids/right_hand_' + title + '.txt')
#
#
# def generate_3d_merged_points(skeleton_data_path, title, left_hand_data_path, right_hand_data_path):
#     skeleton_connections = [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6], [6, 31], [1, 7], [7, 8], [8, 9], [9, 10],
#                             [10, 11], [11, 12], [12, 13], [13, 14], [10, 15], [15, 16], [16, 17], [17, 18], [10, 19],
#                             [19, 20], [20, 21], [21, 22], [10, 23], [23, 24], [24, 25], [25, 26], [10, 27], [27, 28],
#                             [28, 29], [29, 30], [31, 32], [32, 33], [33, 34], [34, 35], [31, 36], [36, 37], [37, 38],
#                             [38, 39], [31, 40], [40, 41], [41, 42], [42, 43], [31, 44], [44, 45], [45, 46], [46, 47],
#                             [31, 48], [48, 49], [49, 50], [50, 51]]
#     # fig = plt.figure()
#     # ax = fig.add_subplot(projection="3d")
#     regex = "\\[(.*?)\\]"
#     left_hand_joint_coordinates = []
#     right_hand_joint_coordinates = []
#     skeleton_keypoints = np.load(skeleton_data_path, allow_pickle=True)
#
#     with open(left_hand_data_path, 'r') as f:
#         for line in f:
#             groups = re.findall(regex, line)
#             if len(groups) > 0:
#                 ite = groups[0]
#                 ite = ite.replace('[', '')
#                 ite = ite.replace(']', '')
#                 left_hand_joint_coordinates.append(ite)
#
#     with open(right_hand_data_path, 'r') as f:
#         for line in f:
#             groups = re.findall(regex, line)
#             if len(groups) > 0:
#                 ite = groups[0]
#                 ite = ite.replace('[', '')
#                 ite = ite.replace(']', '')
#                 right_hand_joint_coordinates.append(ite)
#
#     xs = []
#     ys = []
#     zs = []
#
#     for i in range(len(skeleton_keypoints)):
#         xdata = []
#         ydata = []
#         zdata = []
#         for index in range(len(skeleton_keypoints[i])):
#             if index > 6:
#                 xdata.append(skeleton_keypoints[i][index][0] * 1000)
#                 ydata.append(skeleton_keypoints[i][index][1] * 1000)
#                 zdata.append(skeleton_keypoints[i][index][2] * 1000)
#         for index in range(21):
#             xdata.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#             ydata.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#             zdata.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#         for index in range(21):
#             xdata.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#             ydata.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#             zdata.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#
#         xs.append(xdata)
#         ys.append(ydata)
#         zs.append(zdata)
#
#     for i in range(len(xs)):
#         left_hand_x_shift = xs[i][6] - xs[i][31]
#         left_hand_y_shift = ys[i][6] - ys[i][31]
#         left_hand_z_shift = zs[i][6] - zs[i][31]
#
#         right_hand_x_shift = xs[i][9] - xs[i][10]
#         right_hand_y_shift = ys[i][9] - ys[i][10]
#         right_hand_z_shift = zs[i][9] - zs[i][10]
#
#         for inde in range(21):
#             xs[i][inde + 31] = xs[i][inde + 31] + left_hand_x_shift
#             ys[i][inde + 31] = ys[i][inde + 31] + left_hand_y_shift
#             zs[i][inde + 31] = zs[i][inde + 31] + left_hand_z_shift
#             xs[i][inde + 10] = xs[i][inde + 10] + right_hand_x_shift
#             ys[i][inde + 10] = ys[i][inde + 10] + right_hand_y_shift
#             zs[i][inde + 10] = zs[i][inde + 10] + right_hand_z_shift
#
#     skel_points = []
#
#     for i in range(len(xs)):
#         temp_cords = [xs[i], ys[i], zs[i]]
#
#     save('test_all_points//data_' + title + '.npy', asarray([xs, ys, zs]))
#     # for i in range(len(xs)):
#     #     ax.clear()
#     #     ax.scatter(xs[i], ys[i], zs[i], c='blue')
#     #     for connection in skeleton_connections:
#     #         ax.plot([xs[i][connection[0]], xs[i][connection[1]]], [ys[i][connection[0]], ys[i][connection[1]]],
#     #                 [zs[i][connection[0]], zs[i][connection[1]]], color='black')
#     #     ax.view_init(235, 270)
#     #     ax.set_yticklabels([])
#     #     ax.set_xticklabels([])
#     #     ax.set_zticklabels([])
#     #     plt.savefig('temp_screenshots/' + str(i) + '.png')
#     #     sys.exit()
#     #
#     # with imageio.get_writer('temp_gifs/' + skeleton_data_path.split('/')[2].split('.')[0] + '.gif', mode='I') as writer:
#     #     for indexo in range(len(xs)):
#     #         image = imageio.imread('temp_screenshots/' + str(indexo) + '.png')
#     #         writer.append_data(image)
#     #
#     # for filename in os.listdir('./temp_screenshots/'):
#     #     file_path = os.path.join('./temp_screenshots/', filename)
#     #     try:
#     #         if os.path.isfile(file_path) or os.path.islink(file_path):
#     #             os.unlink(file_path)
#     #         elif os.path.isdir(file_path):
#     #             shutil.rmtree(file_path)
#     #     except Exception as e:
#     #         print('Failed to delete %s. Reason: %s' % (file_path, e))
#
#
# def generate_3d_points():
#     files = get_files_paths('./skeleton_points/')
#
#     data = []
#
#     for file_index, file in enumerate(files):
#         title = file.split('/')[2].split('.')[0]
#         left_hand_data_path = './parsed_vids/left_hand_' + title + '.txt'
#         right_hand_data_path = './parsed_vids/right_hand_' + title + '.txt'
#         regex = "\\[(.*?)\\]"
#
#         current_file_cords = []
#         right_hand_joint_coordinates = []
#         left_hand_joint_coordinates = []
#         skeleton_keypoints = np.load(file, allow_pickle=True)
#
#         with open(right_hand_data_path, 'r') as f:
#             for line in f:
#                 groups = re.findall(regex, line)
#                 if len(groups) > 0:
#                     ite = groups[0]
#                     ite = ite.replace('[', '')
#                     ite = ite.replace(']', '')
#                     right_hand_joint_coordinates.append(ite)
#
#         with open(left_hand_data_path, 'r') as f:
#             for line in f:
#                 groups = re.findall(regex, line)
#                 if len(groups) > 0:
#                     ite = groups[0]
#                     ite = ite.replace('[', '')
#                     ite = ite.replace(']', '')
#                     left_hand_joint_coordinates.append(ite)
#
#         for i in range(len(skeleton_keypoints)):
#             rxs = []
#             rys = []
#             rzs = []
#             lxs = []
#             lys = []
#             lzs = []
#             temp_cords = []
#             for index in range(len(skeleton_keypoints[i])):
#                 # if index > 6:
#                 temp_cords.append(skeleton_keypoints[i][index] * 1000)
#             for index in range(21):
#                 rxs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#                 rys.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#                 rzs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#             for index in range(21):
#                 lxs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[0]))
#                 lys.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[1]))
#                 lzs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[2]))
#
#             # right_hand_x_shift = temp_cords[9][0] - rxs[0]
#             # right_hand_y_shift = temp_cords[9][1] - rys[0]
#             # right_hand_z_shift = temp_cords[9][2] - rzs[0]
#             #
#             # left_hand_x_shift = temp_cords[6][0] - lxs[0]
#             # left_hand_y_shift = temp_cords[6][1] - lys[0]
#             # left_hand_z_shift = temp_cords[6][2] - lzs[0]
#
#             right_hand_x_shift = temp_cords[16][0] - rxs[0]
#             right_hand_y_shift = temp_cords[16][1] - rys[0]
#             right_hand_z_shift = temp_cords[16][2] - rzs[0]
#
#             left_hand_x_shift = temp_cords[13][0] - lxs[0]
#             left_hand_y_shift = temp_cords[13][1] - lys[0]
#             left_hand_z_shift = temp_cords[13][2] - lzs[0]
#
#             for inde in range(len(rxs)):
#                 rxs[inde] = rxs[inde] + right_hand_x_shift
#                 rys[inde] = rys[inde] + right_hand_y_shift
#                 rzs[inde] = rzs[inde] + right_hand_z_shift
#                 lxs[inde] = lxs[inde] + left_hand_x_shift
#                 lys[inde] = lys[inde] + left_hand_y_shift
#                 lzs[inde] = lzs[inde] + left_hand_z_shift
#
#             for inde in range(len(rxs)):
#                 temp_cords.append([rxs[inde], rys[inde], rzs[inde]])
#
#             for inde in range(len(rxs)):
#                 temp_cords.append([lxs[inde], lys[inde], lzs[inde]])
#
#             current_file_cords.append(temp_cords)
#
#         data.append(current_file_cords)
#         print(file_index)
#         save('whole_body_graph_points_with_low//data_' + str(title) + '.npy', asarray(current_file_cords))


def generate_reduced_3d_points():
    hand_joint_indexes_to_stay = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20]
    files = get_files_paths('./skeleton_points')

    data = []

    for file_index, file in enumerate(files):
        title = file.split('\\')[1].split('.')[0]
        left_hand_data_path = './left_hand_keypoints/left_hand_' + title + '.txt'
        right_hand_data_path = './right_hand_keypoints/right_hand_' + title + '.txt'
        regex = "\\[(.*?)\\]"

        current_file_cords = []
        right_hand_joint_coordinates = []
        left_hand_joint_coordinates = []
        skeleton_keypoints = np.load(file, allow_pickle=True)

        with open(right_hand_data_path, 'r') as f:
            for line in f:
                groups = re.findall(regex, line)
                if len(groups) > 0:
                    ite = groups[0]
                    ite = ite.replace('[', '')
                    ite = ite.replace(']', '')
                    right_hand_joint_coordinates.append(ite)

        with open(left_hand_data_path, 'r') as f:
            for line in f:
                groups = re.findall(regex, line)
                if len(groups) > 0:
                    ite = groups[0]
                    ite = ite.replace('[', '')
                    ite = ite.replace(']', '')
                    left_hand_joint_coordinates.append(ite)

        for i in range(len(skeleton_keypoints)):
            rxs = []
            rys = []
            rzs = []
            lxs = []
            lys = []
            lzs = []
            temp_cords = []
            for index in range(len(skeleton_keypoints[i])):
                if index > 6:
                    temp_cords.append(skeleton_keypoints[i][index] * 1000)
            for index in hand_joint_indexes_to_stay:
                rxs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[0]))
                rys.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[1]))
                rzs.append(float(right_hand_joint_coordinates[i * 21 + index].split(",")[2]))
            for index in hand_joint_indexes_to_stay:
                lxs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[0]))
                lys.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[1]))
                lzs.append(float(left_hand_joint_coordinates[i * 21 + index].split(",")[2]))

            right_hand_x_shift = temp_cords[9][0] - rxs[0]
            right_hand_y_shift = temp_cords[9][1] - rys[0]
            right_hand_z_shift = temp_cords[9][2] - rzs[0]

            left_hand_x_shift = temp_cords[6][0] - lxs[0]
            left_hand_y_shift = temp_cords[6][1] - lys[0]
            left_hand_z_shift = temp_cords[6][2] - lzs[0]

            for index in range(len(rxs)):
                rxs[index] = rxs[index] + right_hand_x_shift
                rys[index] = rys[index] + right_hand_y_shift
                rzs[index] = rzs[index] + right_hand_z_shift
                lxs[index] = lxs[index] + left_hand_x_shift
                lys[index] = lys[index] + left_hand_y_shift
                lzs[index] = lzs[index] + left_hand_z_shift

            for index in range(len(rxs)):
                if index > 0:
                    temp_cords.append([rxs[index], rys[index], rzs[index]])

            for index in range(len(rxs)):
                if index > 0:
                    temp_cords.append([lxs[index], lys[index], lzs[index]])

            current_file_cords.append(temp_cords)

        data.append(current_file_cords)
        save('body_graph_points//data_' + str(title) + '.npy', asarray(current_file_cords))


def generate_skeleton_keypoints_with_length():
    videos_keypoints_paths = get_files_paths('./body_graph_points')
    index = 0
    for vid_keypoints_path in videos_keypoints_paths:
        vid_keypoints = np.load(vid_keypoints_path)
        vid_keypoints_prepared = []
        frames = vid_keypoints.shape[0]
        min_frames = 32
        if frames >= min_frames:
            starting_index = int(frames / 2) - int(min_frames / 2)
            for i in range(min_frames):
                vid_keypoints_prepared.append(vid_keypoints[starting_index + i])
        else:
            missing_frames = min_frames - vid_keypoints.shape[0]
            frames_to_add_before = int(missing_frames / 2) + missing_frames % 2
            frames_to_add_after = int(missing_frames / 2)

            for i in range(frames_to_add_before):
                vid_keypoints_prepared.append(vid_keypoints[0])
            for vid_keypoint in vid_keypoints:
                vid_keypoints_prepared.append(vid_keypoint)
            for i in range(frames_to_add_after):
                vid_keypoints_prepared.append(vid_keypoints[frames - 1])
        print(index)
        index += 1
        save('preprocessed_reduced_points//' + vid_keypoints_path.split('\\')[1], asarray(vid_keypoints_prepared))


def prepare_wlasl_data(top_glosses):
    data = json.load(open('WLASL_v0.3.json'))

    preprocessed_points_path = 'preprocessed_reduced_points'
    wlasl_target_dir = 'WLASL_reduced_skeleton'

    test_vid_ids = []
    train_vid_ids = []
    eval_vid_ids = []
    glosses = []

    for entry_index in range(top_glosses):
        glosses.append(data[entry_index]['gloss'])
        for instance_index in range(len(data[entry_index]['instances'])):
            data_file_name = 'data_' + data[entry_index]['instances'][instance_index]['video_id'] + '.npy'
            if data[entry_index]['instances'][instance_index]['split'] == 'train':
                shutil.copyfile('./' + preprocessed_points_path + '/' + data_file_name, './' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/train/' + data_file_name)
                train_vid_ids.append(data[entry_index]['instances'][instance_index]['video_id'])
            if data[entry_index]['instances'][instance_index]['split'] == 'test':
                shutil.copyfile('./' + preprocessed_points_path + '/' + data_file_name, './' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/test/' + data_file_name)
                test_vid_ids.append(data[entry_index]['instances'][instance_index]['video_id'])
            if data[entry_index]['instances'][instance_index]['split'] == 'val':
                shutil.copyfile('./' + preprocessed_points_path + '/' + data_file_name, './' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/val/' + data_file_name)
                eval_vid_ids.append(data[entry_index]['instances'][instance_index]['video_id'])

    test_vid_ids.sort()
    eval_vid_ids.sort()
    train_vid_ids.sort()

    for entry_index in range(len(data)):
        for instance in data[entry_index]['instances']:
            for i in range(len(test_vid_ids)):
                if test_vid_ids[i] == instance['video_id']:
                    test_vid_ids[i] = entry_index
            for i in range(len(train_vid_ids)):
                if train_vid_ids[i] == instance['video_id']:
                    train_vid_ids[i] = entry_index
            for i in range(len(eval_vid_ids)):
                if eval_vid_ids[i] == instance['video_id']:
                    eval_vid_ids[i] = entry_index

    save('./' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/test_labels_' + str(top_glosses), asarray(test_vid_ids))
    save('./' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/eval_labels_' + str(top_glosses), asarray(eval_vid_ids))
    save('./' + wlasl_target_dir + '/WLASL' + str(top_glosses) + '/train_labels_' + str(top_glosses), asarray(train_vid_ids))


def get_files_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths
