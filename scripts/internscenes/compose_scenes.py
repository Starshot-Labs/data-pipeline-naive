

import os
import pdb
import json
import trimesh
import numpy as np
from tqdm import tqdm
from trimesh.transformations import rotation_matrix

import threading
from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
BASE_DIR = Path(os.getcwd()).parent

ASSET_LIBRARY_FOLDER = os.path.join(BASE_DIR, "data/asset_library")
SCENE_SAVE_DIR = os.path.join(BASE_DIR, "tutorial/examples/composed_scenes")
SCENE_INFO_DIR = os.path.join(BASE_DIR, "data/Layout_info")

class AssetMeshLoader():
    def __init__(self):

        # other assets:
        self.asset_dir = ASSET_LIBRARY_FOLDER
        self.obja_uid_2_rotation = json.load(open(os.path.join(ASSET_LIBRARY_FOLDER, "uid_2_angle.json")))
        self.pm_uid_2_origin_cate = json.load(open(os.path.join(ASSET_LIBRARY_FOLDER, "uid_2_origin_cate.json")))

    def get_mesh_path(self, uid):
        mesh_path = None

        if uid.startswith("objaverse/"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("objaverse_old/"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("partnet_mobility"):
            mesh_path = os.path.join(self.asset_dir, uid,  "whole.glb") 
        elif uid.startswith("3D-FUTURE-model"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("hssd-models"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("gen_assets"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        elif uid.startswith("gr100"):
            mesh_path = os.path.join(self.asset_dir, uid + ".glb")
        else:
            raise ValueError(f"Invalid uid: {uid}")

        return mesh_path

    def load_init_mesh(self, uid, use_texture=False):
        mesh_path = self.get_mesh_path(uid)

        if use_texture:
            mesh = trimesh.load(mesh_path)
        else:
            mesh = trimesh.load(mesh_path, force="mesh")

        return mesh
    
    def load_init_rotation(self, uid):
        transform = None

        if uid.startswith("objaverse/"):
            rot_radius = self.obja_uid_2_rotation[uid.split("objaverse/")[-1]] / 180.0 * np.pi
            transform = rotation_matrix(rot_radius, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("objaverse_old/"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("partnet_mobility"):
            
            transform =  rotation_matrix(np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

            # if partnet_mobility is ["pen", "remote", "phone"] category, need to rotate 180 degree around Z axis, 
            # then rotate 90 degree around Y axis
            pm_cate = self.pm_uid_2_origin_cate[uid]
            if pm_cate in ["Pen", "Remote", "Phone"]:
                rotation_matrix_1 = rotation_matrix(np.pi, [0, 0, 1])
                rotation_matrix_2 = rotation_matrix(np.pi / 2, [0, 1, 0])
                transform = rotation_matrix_2 @ rotation_matrix_1 @ transform
            
        elif uid.startswith("3D-FUTURE-model"):
            transform =  rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("hssd-models"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("gen_assets"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        elif uid.startswith("gr100"):
            transform = rotation_matrix(0.5 * np.pi, [0, 0, 1]) @ rotation_matrix(0.5 * np.pi, [1, 0, 0])

        else:
            raise ValueError(f"Invalid uid: {uid}")

        return transform

    def load_canonical_mesh(self, uid, use_texture=False):
        '''
        canonical : means the object is facing the X-axis direction in trimesh, and the Z-axis is up
        '''
        mesh = self.load_init_mesh(uid, use_texture)
        transform = self.load_init_rotation(uid)

        mesh_origen_centroid =  mesh.bounding_box.centroid
        mesh.apply_translation(- mesh_origen_centroid)
        mesh.apply_transform(transform)
        
        return mesh


class SceneComposer():
    def __init__(self, asset_mesh_loader = None):
        self.asset_mesh_loader = asset_mesh_loader if asset_mesh_loader is not None else AssetMeshLoader()

        # file paths
        self.scene_files_dir = SCENE_SAVE_DIR
        self.scene_info_dir = SCENE_INFO_DIR

    def get_scale_transform_from_rules(self, mesh_size, instance_info, bbox_data_key = "bbox"):
        '''
        introduce some special rules
        calculate scale based on mesh_size & instance_info's bbox, return a 4x4 transform matrix
        '''
        if instance_info["category"] not in ["carpet", "clothes"]:
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale = target_size / mesh_size
            scale_matrix = np.diag([scale[0], scale[1], scale[2], 1])
            return scale_matrix

        elif instance_info["category"] == "carpet":
            # add the judgment of carpet wrong orientation causing excessive stretching
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale_factors = target_size / mesh_size

            if target_size[2]/target_size[0] > 150 or target_size[2]/target_size[1] > 150:
                # 地毯错误朝向导致过度拉伸
                if target_size[2]/target_size[0] > target_size[2]/target_size[1]:
                    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [0, 1, 0]) # 绕 Y 轴转90度
                    target_size = np.array([target_size[2], target_size[0], target_size[1]])
                    scale_factors = target_size / mesh_size
                    scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])
                else:
                    rotation_matrix = trimesh.transformations.rotation_matrix(0.5 * np.pi, [1, 0, 0]) # 绕 X 轴转90度
                    target_size = np.array([target_size[0], target_size[2], target_size[1]])
                    scale_factors = target_size / mesh_size
                    scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])

                return scale_matrix @ rotation_matrix

            else:
                # carpet normal orientation
                scale_matrix = np.diag([scale_factors[0], scale_factors[1], scale_factors[2]/100.0, 1])

                return scale_matrix

        elif instance_info["category"] == "clothes":
            target_size = np.array(instance_info[bbox_data_key][3:6])
            scale = target_size / mesh_size
            min_scale = min(scale)
            scale_matrix = np.diag([min_scale, min_scale, min_scale, 1])

        return scale_matrix
        

    def compose_scene_from_instance_infos(self, instance_infos, output_glb_path, use_texture, bbox_data_key = "bbox"):
        # init scene
        scene = trimesh.scene.Scene()
        lock = threading.Lock()

        def process_single_instance(instance):
            """
            function to process single instance, for multi-threading
            """
            if instance["model_uid"] == '':
                print(f"model_uid is empty. No retrieval reslut. (cate: {instance['category']})")
                return None

            uid = instance["model_uid"]
            mesh = self.asset_mesh_loader.load_canonical_mesh(uid, use_texture = use_texture)

            # get geometry name
            geometry_name = instance["category"] + "@" + instance["model_uid"]

            # transform
            transform_final = np.eye(4)
            box_data = instance[bbox_data_key]

            # scale
            mesh_size = mesh.bounding_box.extents
            scale_transform_matrix = self.get_scale_transform_from_rules(mesh_size, instance,   bbox_data_key = bbox_data_key)
            transform_final = scale_transform_matrix @ transform_final

            # rotation
            euler_angles = np.array(box_data[6:9])
            rotation_matrix = trimesh.transformations.euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
            transform_final = rotation_matrix @ transform_final

            # translation
            center = np.array(box_data[0:3])
            transform_final[:3, 3] = center

            # rotate -90 degree around X axis, to make the scene Y-up, so that the scene has the correct upward orientation
            rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
            transform_final = rotation_matrix @ transform_final

            total_transform = transform_final
            
            return mesh, geometry_name, total_transform

        def process_instance(index):
            instance = instance_infos[index]
            result = process_single_instance(instance)

            if result is not None:
                mesh_or_scene, parent_node_name, transform = result
                parent_node_name = str(index) + '_' + parent_node_name
                with lock:
                    scene.graph.update(frame_to=parent_node_name, matrix=transform)

                    if isinstance(mesh_or_scene, trimesh.Scene):
                        for geom_name, mesh_part in mesh_or_scene.geometry.items():
                            nodes_for_this_geometry = mesh_or_scene.graph.geometry_nodes.get(geom_name, [])

                            for i, node_name_in_subscene in enumerate(nodes_for_this_geometry):
                                internal_transform, _ = mesh_or_scene.graph.get(node_name_in_subscene)
                                scene.add_geometry(
                                    mesh_part,
                                    geom_name=f"{parent_node_name}_{geom_name}_{i}",
                                    transform=internal_transform,
                                    parent_node_name=parent_node_name
                                )
                    else: # 
                        scene.add_geometry(
                            mesh_or_scene,
                            geom_name=parent_node_name + "_geom",
                            parent_node_name=parent_node_name
                        )
                print(f"process_instance {index} done")

        # use thread pool to process all instances
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for index in range(len(instance_infos)):
                futures.append(executor.submit(process_instance, index))
            
            # wait for all tasks to complete and handle exceptions
            for future in futures: #tqdm(futures, desc="exporting scene..."):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing instance: {str(e)}")

        # remove texture from scene glb
        if not use_texture:
            from trimesh.visual.texture import TextureVisuals
            empty_visual = TextureVisuals()
            for geometry in scene.geometry.values():
                geometry.visual = empty_visual
            
        if output_glb_path != None:
            trimesh.exchange.export.export_mesh(scene, output_glb_path)
        return scene


    def compose_one_scene(self, scene_name, use_texture = True, add_floor = True, add_wall = True, add_ceiling = True):

        input_instance_infos_path = os.path.join(self.scene_info_dir, scene_name, "layout.json")
        output_glb_path = os.path.join(self.scene_files_dir, scene_name, "glb_scene.glb")
        instance_infos = json.load(open(input_instance_infos_path))
        trimesh_scene = self.compose_scene_from_instance_infos(instance_infos, None, use_texture, bbox_data_key = "bbox")

        if add_floor:
            try:
                floor = trimesh.load(os.path.join(self.scene_files_dir, scene_name, "StructureMesh", "floor.glb"))
                trimesh_scene.add_geometry(floor, geom_name=f"floor")
            except Exception as e:
                print(f"Error adding floor: {e}")

        if add_wall:
            try:
                wall = trimesh.load(os.path.join(self.scene_files_dir, scene_name, "StructureMesh", "wall.glb"))
                trimesh_scene.add_geometry(wall, geom_name=f"wall")
            except Exception as e:
                print(f"Error adding wall: {e}")

        if add_ceiling:
            try:
                ceiling = trimesh.load(os.path.join(self.scene_files_dir, scene_name, "StructureMesh", "ceiling.glb"))
                trimesh_scene.add_geometry(ceiling, geom_name=f"ceiling")
            except Exception as e:
                print(f"Error adding ceiling: {e}")

        # remove texture from scene glb
        if not use_texture:
            from trimesh.visual.texture import TextureVisuals
            empty_visual = TextureVisuals()
            for geometry in trimesh_scene.geometry.values():
                geometry.visual = empty_visual
            
        if output_glb_path != None:
            trimesh.exchange.export.export_mesh(trimesh_scene, output_glb_path)
            print(f"Composed glb scene has been saved to {output_glb_path}")



if __name__ == "__main__":
    scene_name = "scannet/scene0000_00"
    scene_composer = SceneComposer()
    scene_composer.compose_one_scene(scene_name, use_texture = True, add_floor = True, add_wall = True, add_ceiling = True)



