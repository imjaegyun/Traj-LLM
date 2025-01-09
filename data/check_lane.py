import os
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

def check_lanes_in_all_scenes(nuscenes_path):
    """
    NuScenes 데이터를 순회하면서 lane (혹은 lane_polygon) 정보를 찾아보고,
    어느 scene에서 lanes를 찾는지 콘솔에 출력하는 예시 함수.
    """
    # 1. NuScenes 객체 로드 (trainval or test 버전에 맞춰 조정)
    nusc = NuScenes(version="v1.0-trainval", dataroot=nuscenes_path, verbose=True)

    # 2. 사용 가능한 map_name들을 모두 넣어보고, 어느 map에서 lane이 나오는지 확인
    #    (필요한 맵만 넣어도 됩니다)
    map_names = [
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage",
        "singapore-woodlands"
    ]
    
    # 3. 전체 scene 순회
    for scene_idx, scene in enumerate(nusc.scene):
        scene_token = scene["token"]
        scene_name = scene["name"] if "name" in scene else f"Scene_{scene_idx}"
        first_sample_token = scene["first_sample_token"]

        # 해당 scene의 첫 샘플을 로드해서 ego pose를 얻는다.
        first_sample = nusc.get("sample", first_sample_token)
        sample_data_token = first_sample["data"]["LIDAR_TOP"]  # LIDAR_TOP 기준
        sample_data = nusc.get("sample_data", sample_data_token)
        ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])

        x, y = ego_pose["translation"][0], ego_pose["translation"][1]

        # 4. map_names를 전부 시도해보면서 lane(또는 lane_polygon) layer를 조회
        #    만약 lane을 찾으면 출력
        found_any_lane = False
        for map_name in map_names:
            try:
                nusc_map = NuScenesMap(dataroot=nuscenes_path, map_name=map_name)
                
                # 원하는 레이어를 설정 (아래 두 가지 중 하나 시도)
                layer_name = "lane" 
                #layer_name = "lane_polygon"

                # 특정 반경(r=50m) 내에 lane(혹은 lane_polygon)을 찾는다
                records_in_radius = nusc_map.get_records_in_radius(x, y, 50, [layer_name])

                # records_in_radius는 dict 형태일 수도 있고, 레이어명이 key로 들어있을 수도 있음
                # dict_keys(['lane_polygon']) 형태인지 확인
                if not records_in_radius or layer_name not in records_in_radius:
                    continue  # 이 map_name에는 lane이 없는 걸로 넘어간다.

                lane_ids = records_in_radius[layer_name]  # 이건 list 형태 (lane id 목록)
                if len(lane_ids) > 0:
                    found_any_lane = True
                    print(f"[Scene {scene_idx} - {scene_name}] map='{map_name}', layer='{layer_name}' -> lane count = {len(lane_ids)}")
                    
            except Exception as ex:
                print(f"[Scene {scene_idx}] map='{map_name}' 에서 에러 발생: {ex}")
        
        if not found_any_lane:
            print(f"[Scene {scene_idx} - {scene_name}] 어떤 map에서도 lane을 찾지 못했습니다.")

if __name__ == "__main__":
    nuscenes_path = "/home/user/data/Nuscenes"  # 실제 NuScenes 데이터가 있는 경로
    check_lanes_in_all_scenes(nuscenes_path)
