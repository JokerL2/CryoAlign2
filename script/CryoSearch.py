import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# 根目录
root_dir = './classified_partial_xyz'

# 获取所有 Class 目录
class_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

# 用于统计对比次数的变量
comparison_count = 0

# 定义文件对比函数
def compare_files(txt_file_path, xyz_file_path, other_txt_file_path, other_xyz_file_path, current_class_path):
    print(f'Comparing {txt_file_path} and {xyz_file_path} with {other_txt_file_path} and {other_xyz_file_path}')
    
    # 假设你使用系统命令执行对比
    os.system("./CryoAlign_alignment %s %s %s %s %s %s" % (root_dir, current_class_path, other_xyz_file_path, xyz_file_path, other_txt_file_path, txt_file_path))
    
    return 1  # 返回对比计数

# 设置并行线程的数量
max_threads = 4  # 你可以根据需要调整此值

# 并行执行
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = []

    # 遍历每一个 Class 目录
    for class_dir in class_dirs:
        current_class_path = os.path.join(root_dir, class_dir)
        
        # 获取当前目录中的所有 .txt 文件
        txt_files = [f for f in os.listdir(current_class_path) if f.endswith('.txt')]
        
        # **对比当前类目录中的文件**
        for i in range(len(txt_files)):
            txt_file = txt_files[i]
            
            # 找到对应的 .xyz 文件
            base_filename = txt_file[:-4]  # 去掉 .txt 得到基础文件名
            xyz_file = "Points_" + base_filename.split('_')[0] + "_Key" + '.xyz'
            
            # 检查 .xyz 文件是否存在
            if not os.path.exists(os.path.join(current_class_path, xyz_file)):
                continue
            
            txt_file_path = os.path.join(current_class_path, txt_file)
            xyz_file_path = os.path.join(current_class_path, xyz_file)
            
            # **当前类内的文件对比**
            for j in range(i + 1, len(txt_files)):  # 确保不会重复对比自己
                other_txt_file = txt_files[j]
                
                other_base_filename = other_txt_file[:-4]
                other_xyz_file = "Points_" + other_base_filename.split('_')[0] + "_Key" + '.xyz'
                
                # 检查当前类中 .xyz 文件是否存在
                if not os.path.exists(os.path.join(current_class_path, other_xyz_file)):
                    continue
                
                other_txt_file_path = os.path.join(current_class_path, other_txt_file)
                other_xyz_file_path = os.path.join(current_class_path, other_xyz_file)
                
                # 提交到线程池进行异步处理
                futures.append(executor.submit(compare_files, txt_file_path, xyz_file_path, other_txt_file_path, other_xyz_file_path, current_class_path))
            
            # **对比其他类目录中的文件**
            for other_class_dir in class_dirs:
                if other_class_dir == class_dir:
                    continue  # 跳过同一个类目录，防止重复
                
                other_class_path = os.path.join(root_dir, other_class_dir)
                
                # 获取其他 Class 目录中的所有 .txt 文件
                other_txt_files = [f for f in os.listdir(other_class_path) if f.endswith('.txt')]
                
                for other_txt_file in other_txt_files:
                    other_base_filename = other_txt_file[:-4]
                    other_xyz_file = "Points_" + other_base_filename.split('_')[0] + "_Key" + '.xyz'
                    
                    # 检查其他类目录中的 .xyz 文件是否存在
                    if not os.path.exists(os.path.join(other_class_path, other_xyz_file)):
                        continue
                    
                    other_txt_file_path = os.path.join(other_class_path, other_txt_file)
                    other_xyz_file_path = os.path.join(other_class_path, other_xyz_file)
                    
                    # 提交到线程池进行异步处理
                    #futures.append(executor.submit(compare_files, txt_file_path, xyz_file_path, other_txt_file_path, other_xyz_file_path, current_class_path))

    # 收集任务结果并统计对比次数
    for future in as_completed(futures):
        comparison_count += future.result()

# 输出总对比次数
print(f'Total comparisons made: {comparison_count}')
