import os
import numpy as np
import tifffile as tiff
import open3d as o3d
from pathlib import Path

import tqdm
from PIL import Image
import math
import preprocess.mvtec3d_utils as mvt_util
import argparse


def get_edges_of_pc(organized_pc):
    unorganized_edges_pc = organized_pc[0:10, :, :].reshape(organized_pc[0:10, :, :].shape[0]*organized_pc[0:10, :, :].shape[1],organized_pc[0:10, :, :].shape[2])
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc,organized_pc[-10:, :, :].reshape(organized_pc[-10:, :, :].shape[0] * organized_pc[-10:, :, :].shape[1],organized_pc[-10:, :, :].shape[2])],axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, 0:10, :].reshape(organized_pc[:, 0:10, :].shape[0] * organized_pc[:, 0:10, :].shape[1],organized_pc[:, 0:10, :].shape[2])], axis=0)
    unorganized_edges_pc = np.concatenate([unorganized_edges_pc, organized_pc[:, -10:, :].reshape(organized_pc[:, -10:, :].shape[0] * organized_pc[:, -10:, :].shape[1],organized_pc[:, -10:, :].shape[2])], axis=0)
    unorganized_edges_pc = unorganized_edges_pc[np.nonzero(np.all(unorganized_edges_pc != 0, axis=1))[0],:]
    return unorganized_edges_pc

def get_plane_eq(unorganized_pc,ransac_n_pts=50):
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc))
    plane_model, inliers = o3d_pc.segment_plane(distance_threshold=0.004, ransac_n=ransac_n_pts, num_iterations=1000)
    return plane_model

def remove_plane(organized_pc_clean, organized_rgb ,distance_threshold=0.005):
    # PREP PC
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc_clean)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)
    clean_planeless_unorganized_pc = unorganized_pc.copy()
    planeless_unorganized_rgb = unorganized_rgb.copy()

    # REMOVE PLANE
    plane_model = get_plane_eq(get_edges_of_pc(organized_pc_clean))
    distances = np.abs(np.dot(np.array(plane_model), np.hstack((clean_planeless_unorganized_pc, np.ones((clean_planeless_unorganized_pc.shape[0], 1)))).T))
    plane_indices = np.argwhere(distances < distance_threshold)

    planeless_unorganized_rgb[plane_indices] = 0
    clean_planeless_unorganized_pc[plane_indices] = 0
    clean_planeless_organized_pc = clean_planeless_unorganized_pc.reshape(organized_pc_clean.shape[0],
                                                                          organized_pc_clean.shape[1],
                                                                          organized_pc_clean.shape[2])
    planeless_organized_rgb = planeless_unorganized_rgb.reshape(organized_rgb.shape[0],
                                                                          organized_rgb.shape[1],
                                                                          organized_rgb.shape[2])
    return clean_planeless_organized_pc, planeless_organized_rgb



def connected_components_cleaning(organized_pc, organized_rgb, image_path):
    unorganized_pc = mvt_util.organized_pc_to_unorganized_pc(organized_pc)
    unorganized_rgb = mvt_util.organized_pc_to_unorganized_pc(organized_rgb)

    nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
    o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
    labels = np.array(o3d_pc.cluster_dbscan(eps=0.006, min_points=30, print_progress=False))


    unique_cluster_ids, cluster_size = np.unique(labels,return_counts=True)
    max_label = labels.max()
    if max_label>0:
        print("##########################################################################")
        print(f"Point cloud file {image_path} has {max_label + 1} clusters")
        print(f"Cluster ids: {unique_cluster_ids}. Cluster size {cluster_size}")
        print("##########################################################################\n\n")

    largest_cluster_id = unique_cluster_ids[np.argmax(cluster_size)]
    outlier_indices_nonzero_array = np.argwhere(labels != largest_cluster_id)
    outlier_indices_original_pc_array = nonzero_indices[outlier_indices_nonzero_array]
    unorganized_pc[outlier_indices_original_pc_array] = 0
    unorganized_rgb[outlier_indices_original_pc_array] = 0
    organized_clustered_pc = unorganized_pc.reshape(organized_pc.shape[0],
                                                                          organized_pc.shape[1],
                                                                          organized_pc.shape[2])
    organized_clustered_rgb = unorganized_rgb.reshape(organized_rgb.shape[0],
                                                    organized_rgb.shape[1],
                                                    organized_rgb.shape[2])
    return organized_clustered_pc, organized_clustered_rgb

def roundup_next_100(x):
    return int(math.ceil(x / 100.0)) * 100

def pad_cropped_pc(cropped_pc, single_channel=False):
    orig_h, orig_w = cropped_pc.shape[0], cropped_pc.shape[1]
    round_orig_h = roundup_next_100(orig_h)
    round_orig_w = roundup_next_100(orig_w)
    large_side = max(round_orig_h, round_orig_w)

    a = (large_side - orig_h) // 2
    aa = large_side - a - orig_h

    b = (large_side - orig_w) // 2
    bb = large_side - b - orig_w
    if single_channel:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb)), mode='constant')
    else:
        return np.pad(cropped_pc, pad_width=((a, aa), (b, bb), (0, 0)), mode='constant')

def generate_surface_normal(organized_pcd_image: np.ndarray, k_estimation: int = 30,
                            k_orientation: int = 50) -> np.ndarray:
    """
    从有组织的点云图像生成表面法线图。

    该函数严格遵循PIRN论文中描述的方法论：
    1. 从(H, W, 3)的图像中提取有效点。
    2. 使用Open3D估计每个点的法线 (k=30)。
    3. 使用一致性切平面法统一法线方向 (k=50)。
    4. 将法线向量映射回原始图像网格，并编码为的RGB图像。

    Args:
        organized_pcd_image (np.ndarray): 一个形状为 (H, W, 3) 的NumPy数组，
                                        代表有组织的点云，其中背景点为。
        k_estimation (int): 用于法线估计的最近邻数量。
        k_orientation (int): 用于法线方向一致性传播的邻域大小。

    Returns:
        np.ndarray: 一个形状为 (H, W, 3) 的uint8类型的NumPy数组，
                    代表编码为RGB颜色的表面法线图。
    """
    # --- 步骤 1: 输入验证和数据准备 ---
    if not isinstance(organized_pcd_image,
                      np.ndarray) or organized_pcd_image.ndim != 3 or len(organized_pcd_image.shape) != 3:
        raise ValueError("输入必须是一个形状为 (H, W, 3) 的NumPy数组。")

    h, w, _ = organized_pcd_image.shape
    points = organized_pcd_image.reshape(-1, 3)

    # --- 步骤 2: 过滤无效点 ---
    valid_points_mask = np.any(points != 0, axis=1)
    valid_points = points[valid_points_mask]

    if len(valid_points) < k_estimation:
        # 如果有效点太少，无法计算法线，返回黑色图像
        return np.zeros((h, w, 3), dtype=np.uint8)

    original_indices = np.where(valid_points_mask)

    # --- 步骤 3: Open3D点云对象创建 ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    # --- 步骤 4: 法线估计 ---
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=k_estimation)
    )

    # --- 步骤 5: 法线方向一致化 ---
    pcd.orient_normals_consistent_tangent_plane(k=k_orientation)

    # --- 步骤 6: 投影与颜色编码 ---
    normal_map = np.zeros((h * w, 3), dtype=np.uint8)
    computed_normals = np.asarray(pcd.normals)

    # 将法线向量从 [-1, 1] 范围线性映射到  的颜色范围
    colored_normals = ((computed_normals + 1) / 2 * 255).astype(np.uint8)

    # 将颜色编码的法线填充回它们在图像中的正确位置
    normal_map[original_indices] = colored_normals
    normal_map = normal_map.reshape(h, w, 3)

    # 可选：确保法线主要指向相机（Z轴正方向），这有助于可视化和模型稳定性
    # 映射后，Z=1 -> B=255。我们希望蓝色通道的平均值偏高。
    # 我们只在有足够前景像素的情况下进行此检查，以避免被噪声误导
    if np.mean(normal_map[normal_map > 0]) < 128:
        normal_map = 255 - normal_map
        # 将背景像素重置为0
        normal_map[np.all(organized_pcd_image == 0, axis=2)] = 0

    return normal_map
import matplotlib.pyplot as plt
def preprocess_pc(tiff_path,save_path):
    # READ FILES
    organized_pc = mvt_util.read_tiff_organized_pc(tiff_path)
    rgb_path = str(tiff_path).replace("xyz", "rgb").replace("tiff", "png")
    gt_path = str(tiff_path).replace("xyz", "gt").replace("tiff", "png")
    organized_rgb = np.array(Image.open(rgb_path))

    organized_gt = None
    gt_exists = os.path.isfile(gt_path)
    if gt_exists:
        organized_gt = np.array(Image.open(gt_path))

    # REMOVE PLANE
    planeless_organized_pc, planeless_organized_rgb = remove_plane(organized_pc, organized_rgb)


    # PAD WITH ZEROS TO LARGEST SIDE (SO THAT THE FINAL IMAGE IS SQUARE)
    padded_planeless_organized_pc = pad_cropped_pc(planeless_organized_pc, single_channel=False)
    padded_planeless_organized_rgb = pad_cropped_pc(planeless_organized_rgb, single_channel=False)
    #if gt_exists:
    #    padded_organized_gt = pad_cropped_pc(organized_gt, single_channel=True)

    organized_clustered_pc, organized_clustered_rgb = connected_components_cleaning(padded_planeless_organized_pc, padded_planeless_organized_rgb, tiff_path)
    # SAVE PREPROCESSED FILES
    # tiff.imsave(tiff_path, organized_clustered_pc)

    # 构建法线图的保存路径
    normal_map_path =str(tiff_path).replace("xyz", "normal").replace("tiff", "png")
    normal_map_path = normal_map_path.replace("ori", save_path)
    rgb_path = rgb_path.replace("ori", save_path)
    gt_path = gt_path.replace("ori", save_path)

    # normal_map_path.parent.mkdir(parents=True, exist_ok=True)
    normal_map = generate_surface_normal(organized_clustered_pc)

    plt.imshow(normal_map)
    plt.show()
    plt.imshow(organized_clustered_pc[:,:,0])
    plt.show()
    print()
    # Image.fromarray(normal_map).save(normal_map_path)
    # Image.fromarray(organized_clustered_rgb).save(rgb_path)
    # if gt_exists:
    #    padded_organized_gt = pad_cropped_pc(organized_gt, single_channel=True)
    #    Image.fromarray(padded_organized_gt).save(gt_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess MVTec 3D-AD')
    parser.add_argument('dataset_path', type=str, help='The root path of the MVTec 3D-AD. The preprocessing is done inplace (i.e. the preprocessed dataset overrides the existing one)')

    parser.add_argument('--index', type=int,)
    args = parser.parse_args()


    # root_path = args.dataset_path
    # paths = Path(root_path).rglob('*.tiff')
    # print(f"Found {len(list(paths))} tiff files in {root_path}")
    # processed_files = 0
    # for path in tqdm.tqdm(list(Path(root_path).rglob('*.tiff'))):
    #     preprocess_pc(path)
    #     processed_files += 1
    #     if processed_files % 50 == 0:
    #         print(f"Processed {processed_files} tiff files...")

    class_names = [['bagel', 'bagel', ], ['dowel', 'cable_gland', ], ['carrot', 'tire', ], ['cookie', 'potato', ],
                   ['rope', 'foam', ], ['peach', 'peach', ]]
    class_names = class_names[args.index]

    save_path = "mvtec3DRGBSN"
    root_path = args.dataset_path
    print(os.listdir(root_path))
    paths = Path(root_path).rglob('*.tiff')

    processed_files = 0

    datas = Path(root_path).rglob('*.tiff')
    datas = filter(lambda x: class_names[0] in str(x) or class_names[1] in str(x), datas)
    # print(f"Found {len(list(datas))} tiff files in {class_names}")
    for path in tqdm.tqdm(list(datas)):
        preprocess_pc(path, save_path)
        processed_files += 1
        # if processed_files % 50 == 0:
        #     print(f"Processed {processed_files} tiff files...")








