import numpy as np
import mrcfile
import argparse
import os
from EMAN2 import *
import glob

def transform_map(infile, T_icp, scale, outfile):

	a = EMData(infile, 0)
	T_icp = np.dot(np.dot(np.array([[1/scale, 0, 0, 0], [0, 1/scale, 0, 0], [0, 0, 1/scale, 0], [0, 0, 0, 1]]), T_icp),
               np.array([[scale, 0, 0, 0], [0, scale, 0, 0], [0, 0, scale, 0], [0, 0, 0, 1]]))

	T_eman = T_icp.copy()
	x_cor = a.get_xsize()
	y_cor = a.get_ysize()
	z_cor = a.get_zsize()
	coord_T = np.array([[1, 0, 0, x_cor / 2],
			[0, 1, 0, y_cor / 2],
			[0, 0, 1, z_cor / 2],
			[0, 0, 0, 1.0]])
	coord_T_inv = np.array([[1, 0, 0, -x_cor / 2],
			[0, 1, 0, -y_cor / 2],
			[0, 0, 1, -z_cor / 2],
			[0, 0, 0, 1.0]])
	T_eman = np.dot(coord_T_inv, np.dot(T_icp, coord_T))
	#print("T_eman: ", T_eman)
	T_eman = T_eman.reshape(-1).tolist()
	t = Transform(T_eman[:12])
	#print(t.get_matrix())
	a.transform(t)
	#print(a.get_attr_dict())
	a.write_image(outfile, 0, EMUtil.get_image_ext_type("mrc"), False, None, file_mode_map["float"])
	return

def main(map_dir,RT_dir,action):
    # 初始化一个空列表来存储矩阵
    
    with mrcfile.open(map_dir) as mrc:
        pixel_size = mrc.voxel_size.x
        print(pixel_size)
    
    data_dir = os.path.dirname(map_dir)
    # 打开文件并读取每一行
    if action == 'mask':
        matrices = []
        with open(data_dir+"/"+RT_dir, 'r') as file:
            next(file)  # 跳过标题行
            for line in file:
                parts = line.split('\t')  # 分割每一行
                if len(parts) > 1:
                    matrix_str = parts[1]  # 第二列的值
                    # 分割字符串并转换为浮点数
                    matrix_values = []
                    # 先按分号分割，得到四个矩阵行
                    rows = matrix_str.split(';')
                    for row in rows:
                        # 再按逗号分割，得到矩阵元素
                        elements = row.split(',')
                        # 将每个元素转换为浮点数并添加到列表中
                        matrix_values.extend([float(element) for element in elements])
                    # 将值重新整形为4x4矩阵
                    matrix = np.array(matrix_values).reshape(4, 4)
                    matrices.append(matrix)
        file_name = os.path.basename(map_dir)
        # 去除文件扩展名
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 提取 EMD-3695
        em_prefix = file_name_without_extension.split('.')[0]
        print(em_prefix)
        for i, T in enumerate(matrices):
            output_filename = f"{em_prefix}_trans{i+1}.map"  # 生成输出文件名
            output_dir = os.path.join(os.path.dirname(map_dir), output_filename)
            print(f"Transforming map {i+1} with matrix {T}")
            transform_map(map_dir, T, pixel_size, output_dir)
    elif action == 'glo':
        file_name = os.path.basename(map_dir)
        # 去除文件扩展名
        file_name_without_extension = os.path.splitext(file_name)[0]
        # 提取 EMD-3695
        em_prefix = file_name_without_extension.split('.')[0]
        print(em_prefix)
        T_glo = np.load(data_dir+"/"+RT_dir)
        print(T_glo)
        output_filename = f"{em_prefix}_trans.map"
        output_dir = os.path.join(os.path.dirname(map_dir), output_filename)
        transform_map(map_dir, T_glo, pixel_size, output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform maps using given transformation matrices.")
    parser.add_argument("map_dir", type=str, help="Directory of the input map file.")
    parser.add_argument("RT_dir", type=str, help="Directory of the RT file.")
    parser.add_argument("--flag", type=str, choices=['glo', 'mask'], help="Flag to determine the type of operation.")
    args = parser.parse_args()
    main(args.map_dir,args.RT_dir, args.flag if args.flag else 'glo')
    
