import os
import json

pjoin = os.path.join


def process_raw_dir(
        dataset_collection_folder,
        dataset_name,
        suffixes=['.obj', '.ply']):
    
    classes = []
    dataset_structure_dict = { dataset_name: {} }
    
    subdir_list = list(os.walk(pjoin(dataset_collection_folder, dataset_name)))
    if len(subdir_list) == 0:
        print('[HERE: In process_all] WARNING source is empty')
    
    
    for cur_path, _, files in subdir_list:
        # subdir[0]: current path
        #    if it is under 'datasets_raw', it should look like 'datasets_raw/dataset_name/class_name/instance_name/mesh_filename.obj'
        #    if it is under 'datasets_processed', it should look like 'datasets_processed/SdfSamples/dataset_name/class_name/instance_name.npz'
        # subdir[1]: folders under current path
        # subdir[2]: files under current path

        print(f'[In gen_json] Checking path {cur_path}')
    
        if len(cur_path.split('/')) != 4:
            continue

        class_name = cur_path.split('/')[2]
        instance_name = cur_path.split('/')[3]
        
        if class_name not in classes:
            classes.append(class_name)
            dataset_structure_dict[dataset_name][class_name] = []
        print(f'class_name = {class_name}, instance_name = {instance_name}')
    
        for in_filename in files:
            #print('[HERE: In process_all] Checking if %s is OBJ format' % in_filename)
            if in_filename[-4:] in suffixes:
                print(f'Found {in_filename}')
                dataset_structure_dict[dataset_name][class_name].append(instance_name)

    for class_name in classes:
        dataset_structure_dict[dataset_name][class_name] = sorted(dataset_structure_dict[dataset_name][class_name])


    print(f"Writing json at {pjoin(dataset_collection_folder, json_filename)}")
    json_file_obj = open(pjoin(dataset_collection_folder, json_filename), 'w')
    json.dump(dataset_structure_dict, json_file_obj, indent=4)
    json_file_obj.close()
    
def process_processed_dir(
        dataset_collection_folder,
        dataset_name,
        data_types):
    """
    data_types: The kinds of sampling you need. This should be a list of ['SdfSamples', 'NormalSamples'], etc.
        Only instances that exist in all of these sample types will be recorded in the json.
        The first in the list is used for extracting instance names. Others will be used for matching, which is why I suggest putting 'SdfSamples' first.
    """ 
    
    classes = []
    dataset_structure_dict = { dataset_name: {} }
    
    subdir_list = list(os.walk(pjoin(dataset_collection_folder)))
    if len(subdir_list) == 0:
        print('[HERE: In process_all] WARNING source is empty')
    
    
    for cur_path, _, files in subdir_list:
        # subdir[0]: current path
        #    if it is under 'datasets_raw', it should look like 'datasets_raw/dataset_name/class_name/instance_name/mesh_filename.obj'
        #    if it is under 'datasets_processed', it should look like 'datasets_processed/SdfSamples/dataset_name/class_name/instance_name.npz'
        # subdir[1]: folders under current path
        # subdir[2]: files under current path

        print(f'[In gen_json] Checking path {cur_path}')
    
        if len(cur_path.split('/')) != 4:
            continue

        # get correct names
        type_name = cur_path.split('/')[1]
        tmp_dataset_name = cur_path.split('/')[2]
        if type_name != data_types[0] or tmp_dataset_name != dataset_name:
            continue
        class_name = cur_path.split('/')[3]
        
        for instance_name in files:
            instance_name = instance_name[:-4]
        
            # check if all files exist
            for data_type in data_types:
                to_put_in_json = True
                if data_type == 'SdfSamples':
                    suffix = '.npz'
                if data_type == 'NormalSamples':
                    suffix = '.ply'
                if not pjoin(dataset_collection_folder, data_type, dataset_name, class_name, instance_name + suffix):
                    to_put_in_json = False
            if not to_put_in_json:
                continue
        
            if class_name not in classes:
                classes.append(class_name)
                dataset_structure_dict[dataset_name][class_name] = []
            print(f'class_name = {class_name}, instance_name = {instance_name}')
    
            dataset_structure_dict[dataset_name][class_name].append(instance_name)

    for class_name in classes:
        dataset_structure_dict[dataset_name][class_name] = sorted(dataset_structure_dict[dataset_name][class_name])

    print(f"Writing json at {pjoin(dataset_collection_folder, json_filename)}")
    json_file_obj = open(pjoin(dataset_collection_folder, json_filename), 'w')
    json.dump(dataset_structure_dict, json_file_obj, indent=4)
    json_file_obj.close()
    
    
if __name__ == '__main__':
    
    dataset_name = 'simple_shapes'
    json_filename = dataset_name + '.json'
    
    
    #dataset_collection_folder = 'datasets_processed'
    #types = ["SdfSamples", "NormalSamples"]
    #process_processed_dir(dataset_collection_folder, dataset_name, types)
    
    dataset_collection_folder = 'datasets_raw'
    process_raw_dir(dataset_collection_folder, dataset_name)


    
