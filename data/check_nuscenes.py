#!/usr/bin/env python3

"""
check_nuscenes.py

NuScenes 데이터 구조를 간단히 확인하는 코드 예시.
"""

from nuscenes.nuscenes import NuScenes

def main():
    # NuScenes 데이터 경로
    nuscenes_path = "/home/user/data/Nuscenes"
    nusc = NuScenes(version='v1.0-trainval', dataroot=nuscenes_path, verbose=True)

    # Scene 정보 확인
    print("Total scenes:", len(nusc.scene))
    for i, scene in enumerate(nusc.scene[:5]):  # 첫 5개 Scene 출력
        print(f"Scene {i}:")
        print("  Token:", scene['token'])
        print("  Description:", scene['description'])
        print("  First sample token:", scene['first_sample_token'])
        print("  Last sample token:", scene['last_sample_token'])
        print("  Log:", scene['log_token'])

        # 로그를 통해 location 확인
        log_record = nusc.get("log", scene["log_token"])
        print("  Location:", log_record["location"])  # 예: singapore-onenorth 등

    # Sample 정보 확인
    first_sample_token = nusc.scene[0]['first_sample_token']
    sample = nusc.get('sample', first_sample_token)
    print("\nSample Information:")
    print("  Token:", sample['token'])
    print("  Timestamp:", sample['timestamp'])
    print("  Sensor Data Keys:", sample['data'].keys())
    print("  Annotation Keys:", sample['anns'])

    # Sample Data (센서 데이터) 확인
    lidar_top_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_top_token)
    print("\nLIDAR_TOP Information:")
    print("  Filename:", lidar_data['filename'])
    print("  Ego Pose Token:", lidar_data['ego_pose_token'])
    print("  Calibrated Sensor Token:", lidar_data['calibrated_sensor_token'])

    # Ego Pose 정보 확인
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    print("\nEgo Pose Information:")
    print("  Translation (x, y, z):", ego_pose['translation'])
    print("  Rotation (qx, qy, qz, qw):", ego_pose['rotation'])
    print("  Timestamp:", ego_pose['timestamp'])

    # Annotation 정보 확인
    first_annotation_token = sample['anns'][0]
    annotation = nusc.get('sample_annotation', first_annotation_token)
    print("\nAnnotation Information:")
    print("  Instance Token:", annotation['instance_token'])
    print("  Category Name:", annotation['category_name'])
    print("  Attributes:", annotation['attribute_tokens'])
    print("  Translation (x, y, z):", annotation['translation'])
    print("  Size (width, length, height):", annotation['size'])

    # annotation에 달려 있는 attribute 자세히 확인
    if annotation['attribute_tokens']:
        for attr_token in annotation['attribute_tokens']:
            attr_record = nusc.get('attribute', attr_token)
            print("  -> Attribute Name:", attr_record['name'])
    else:
        print("  -> No attributes found.")

    # Instance 정보 확인
    instance = nusc.get('instance', annotation['instance_token'])
    print("\nInstance Information:")
    print("  Instance token:", instance["token"])

    # KeyError 수정: instance에는 "anns"라는 필드가 없습니다.
    # 대신 "nbr_annotations" 사용
    print("  # of annotations in this instance (nbr_annotations):", instance["nbr_annotations"])
    print("  First annotation token:", instance["first_annotation_token"])
    print("  Last annotation token:", instance["last_annotation_token"])

    # 만약 해당 인스턴스에 연결된 모든 annotation token을 확인하고 싶다면:
    ann_list = []
    ann_token = instance["first_annotation_token"]
    while ann_token != "":
        ann_record = nusc.get("sample_annotation", ann_token)
        ann_list.append(ann_record["token"])
        ann_token = ann_record["next"]

    print("  All annotation tokens in this instance:", ann_list)
    print("  Total annotation count via linked list:", len(ann_list))

    # Sample의 다른 센서(예: CAM_FRONT) 정보도 확인
    if 'CAM_FRONT' in sample['data']:
        front_cam_token = sample['data']['CAM_FRONT']
        front_cam_data = nusc.get('sample_data', front_cam_token)
        print("\nFront Camera (CAM_FRONT) Information:")
        print("  Filename:", front_cam_data['filename'])
        print("  Is keyframe:", front_cam_data['is_key_frame'])
        print("  Width x Height:", front_cam_data['width'], "x", front_cam_data['height'])

        # 카메라의 ego pose, calibration등을 추가로 보고 싶다면:
        front_cam_ego_pose = nusc.get('ego_pose', front_cam_data['ego_pose_token'])
        print("  CAM_FRONT Ego pose translation:", front_cam_ego_pose['translation'])

if __name__ == "__main__":
    main()
