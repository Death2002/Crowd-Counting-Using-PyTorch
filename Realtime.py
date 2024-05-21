import cv2
import numpy as np
import torch
from model import CDENet
import PIL.Image as Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = CDENet()
model = model.cuda()
checkpoint = torch.load('0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

def process_frame(frame):
    img = transform(frame).unsqueeze(0).cuda()
    output = model(img)
    predicted_count = int(output.detach().cpu().sum().numpy())
    # print("Predicted Count : ", predicted_count)
    density_map = output.detach().cpu().reshape(output.detach().cpu().shape[2], output.detach().cpu().shape[3])
    return predicted_count, density_map

def main():
    # cap = cv2.VideoCapture('1.mp4')
    cap = cv2.VideoCapture(2  )


    while True:
        ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            pil_frame = Image.fromarray(frame_rgb)  # Convert to PIL Image
            predicted_count, density_map = process_frame(pil_frame)

            # Convert density map to a NumPy array
            density_map_np = np.array(density_map)

            # Normalize the density map
            density_map_normalized = cv2.normalize(density_map_np, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Resize the density map to a bigger size
            density_map_resized = cv2.resize(density_map_normalized, (density_map_normalized.shape[1] * 5, density_map_normalized.shape[0] * 5))
            density_map_colored = cv2.applyColorMap(density_map_resized, cv2.COLORMAP_JET)
            # Display the original frame and the density map
            cv2.putText(frame, f'Predicted Count: {predicted_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Density Map', density_map_colored)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
