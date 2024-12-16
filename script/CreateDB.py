import csv
import os
import shutil

# CSV文件路径
csv_file = 'class2.csv'
# glo_emd文件夹路径
source_directory = 'partial_emdb'
# 目标目录，用于存放根据Class分类的文件夹
target_directory = 'classified_partial_emdb'

# 确保目标目录存在
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

# 读取CSV文件，并按Class整理文件夹
with open(csv_file, mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)
    for row in reader:
        class_folder = os.path.join(target_directory, 'Class_' + row['Class'])
        emdb_id = row['EMDB ID']
        
        # 确保Class对应的文件夹存在
        if not os.path.exists(class_folder):
            os.makedirs(class_folder)

        # 移动相应的文件夹到目标Class文件夹中
        # 由于文件夹的命名格式是 “EMDB ID” 或者 “EMDB ID_后缀”, 我们将会匹配这两种格式
        moved = False
        for folder_name in os.listdir(source_directory):
            if folder_name.startswith(emdb_id):
                source_path = os.path.join(source_directory, folder_name)
                target_path = os.path.join(class_folder, folder_name)
                shutil.move(source_path, target_path)
                print(f"Moved {source_path} to {target_path}")
                moved = True
                break  # 假定每个EMDB ID只对应一个文件夹，找到后即停止搜索
        
        # 如果没有找到对应的文件夹，打印提示信息
        if not moved:
            print(f"Folder for EMDB ID {emdb_id} not found in {source_directory}.")
import os
import shutil

# 目标目录，其中包含了按类别整理的文件夹
target_directory = 'classified_partial_emdb'

# 遍历目标目录中的每个类别文件夹
for class_folder in os.listdir(target_directory):
    class_folder_path = os.path.join(target_directory, class_folder)
    
    # 确保这是一个文件夹
    if os.path.isdir(class_folder_path):
        
        # 遍历类别文件夹中的每个子文件夹
        for subfolder in os.listdir(class_folder_path):
            subfolder_path = os.path.join(class_folder_path, subfolder)
            
            # 确保这是一个文件夹
            if os.path.isdir(subfolder_path):
                
                # 遍历子文件夹中的每个文件
                for file_name in os.listdir(subfolder_path):
                    file_path = os.path.join(subfolder_path, file_name)
                    target_file_path = os.path.join(class_folder_path, file_name)
                    
                    # 检查这是否是一个文件
                    if os.path.isfile(file_path):
                        # 如果目标路径中已经存在文件，跳过或执行其他操作
                        if not os.path.exists(target_file_path):
                            # 移动文件到类别文件夹中
                            shutil.move(file_path, class_folder_path)
                            print(f"Moved {file_path} to {class_folder_path}")
                        else:
                            # 如果目标文件已存在，可以选择跳过或重命名
                            print(f"File {target_file_path} already exists. Skipping or taking other actions.")
import os
import shutil

# 目标目录，其中包含了按类别整理的文件夹
target_directory = 'classified_partial_emdb'

# 遍历目标目录中的每个类别文件夹
for class_folder in os.listdir(target_directory):
    class_folder_path = os.path.join(target_directory, class_folder)
    
    # 确保这是一个文件夹
    if os.path.isdir(class_folder_path):
        
        # 遍历类别文件夹中的每个子文件夹
        for subfolder in os.listdir(class_folder_path):
            subfolder_path = os.path.join(class_folder_path, subfolder)
            
            # 确保这是一个文件夹
            if os.path.isdir(subfolder_path):
                # 删除子文件夹
                shutil.rmtree(subfolder_path)
                print(f"Deleted subfolder: {subfolder_path}")

import numpy as np
import struct
import os
from numpy.core.fromnumeric import shape
import requests
def collect_files_in_directory(directory):
    
    # List all items in the directory
    items = os.listdir(directory)
    print(items)
    for item in items:
        classdir = directory+item
        for map_file in os.listdir(classdir):
            if map_file.endswith('.map'):
                print(map_file)
                base_url = "https://www.emdataresource.org/node/solr/emd/select?&q=id%3A"
                url = base_url +map_file.split('.')[0]
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    contour_level = data['response']['docs'][0].get('mapcontourlevel')
                old_emd_file_name=classdir +"/"+ map_file
                print(old_emd_file_name)
                new_emd_file= map_file.split('.')[0]
                new_emd_file_name=classdir +"/"+new_emd_file+"_"+str(contour_level)+".map"
                print(new_emd_file_name)
                os.rename(old_emd_file_name,new_emd_file_name)
        
            
# Example usage
directory_path = "./classified_partial_emdb/"
collect_files_in_directory(directory_path)

import os   
def Sample_Cluster(data_dir, map_name, threshold, VOXEL_SIZE):
    mrc_file = "%s/%s" % (data_dir, map_name)
    sub_map = map_name.split('_')[0]
    file_outputname = data_dir+"/"+"Points_"+sub_map+"_Key.xyz"
    sample_file = "%s/%s_%.2f.txt" % (data_dir, map_name[:-4], VOXEL_SIZE)
    print(file_outputname)
    if not os.path.exists(file_outputname):
        print(map_name)
        
        os.system("./Sample -a %s -t %.4f -s %.2f > %s" % (mrc_file, threshold, VOXEL_SIZE, sample_file))
        os.system("./CryoAlign_extract_keypoints %s %s %s %s" % (mrc_file,sample_file,threshold,file_outputname))
items = os.listdir("./classified_partial_emdb")
for item in items:
    item = "./classified_partial_emdb/"+item
    dirs=os.listdir(item)
    for filename in dirs:
        if filename.endswith('.map'):
            parts = filename.split('_')
            emd_id = parts[0]
            print(filename)
            contour_level = parts[1].split('.')[0] + '.' + parts[1].split('.')[1]
            contour_level = float(contour_level)
            Sample_Cluster(item, filename, contour_level, 5.0)