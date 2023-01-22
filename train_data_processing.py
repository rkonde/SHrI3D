import json
import shutil

from numpy import save, asarray

from utils import get_files_paths

wlasl_vids = get_files_paths('./WLASL2000')
data = json.load(open('WLASL_v0.3.json'))

preprocessed_points_path = 'preprocessed_reduced_points'
wlasl_target_dir = 'WLASL_reduced_skeleton'

vid_ids = []
test_vid_ids = []
train_vid_ids = []
eval_vid_ids = []
glosses = []

top_glosses = 100

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
