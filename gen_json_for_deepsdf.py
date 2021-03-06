import os
import json


source_top_dir = 'dataset_raw'
subdir_list = list(os.walk(source_top_dir))

# for json file output

json_filename = source_top_dir + '_bikes.json'
top_key = 'bike_dataset'
classes = ['bicycle', 'motorcycle', 'scooter']

dataset_structure_dict = {
    top_key: { class_key: [] for class_key in classes }
}

#print(dataset_structure_dict)

if len(subdir_list) == 0:
    print('[HERE: In process_all] WARNING source is empty')

for subdir in subdir_list:
    if len(subdir[0].split('/')) < 2:
        continue
    data_class = subdir[0].split('/')[-2]
    print(data_class)
    if data_class in classes:
        for in_filename in subdir[2]:
            print('[HERE: In process_all] Checking if %s is OBJ format' % in_filename)
            if '.obj' == in_filename[-4:]:
                print(f'Found obj {in_filename}')
                dataset_structure_dict[top_key][data_class].append(in_filename[:-4])

for data_class in classes:
    dataset_structure_dict['bike_dataset'][data_class] = sorted(dataset_structure_dict['bike_dataset'][data_class])


print(f"Writing json at {os.path.join('.', json_filename)}")
json_file_obj = open(os.path.join('.', json_filename), 'w')
json.dump(dataset_structure_dict, json_file_obj, indent=4)
json_file_obj.close()
