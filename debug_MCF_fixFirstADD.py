from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("C:/Users/dujin/Documents/Code/YOLOv11-RGBT/runs/vedai/VEDAI-yolo11n-obb-midfusion-MCF-fixFirstADD/weights/best.pt")
    print(model.state_dict())