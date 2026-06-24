import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO

# 1. Define the Perspective Translator
def get_car_perspective_label(raw_camera_label):
    """
    Translates camera-based labels to car-based labels.
    """
    perspective_map = {
        'front-right': 'front-left-side',
        'front-left':  'front-right-side',
        'back-right':  'back-left-side',
        'back-left':   'back-right-side',
        'side-right':  'left-side',
        'side-left':   'right-side',
        'front':       'front',
        'back':        'back'
    }
    return perspective_map.get(raw_camera_label, raw_camera_label)

# 2. Load your trained model
# Make sure the path points to your best model file
model = YOLO('models/best_car_angle.pt')

# 3. Run prediction
results = model.predict(
    source='testvideo.mp4', 
    save=True,      
    show=False,     
    conf=0.5,
    stream=True # stream=True is better for video memory
)

# 4. Preview and Save the results
for i, result in enumerate(results):
    # Extract the original camera-perspective label
    top1_id = result.probs.top1
    raw_label = result.names[top1_id]
    
    # Get your custom car-perspective label
    car_side = get_car_perspective_label(raw_label)
    
    # Update the result label for visualization
    result.names[top1_id] = car_side
    
    # Plot the result with the updated label
    im_array = result.plot() 
    im = Image.fromarray(im_array[..., ::-1])
    
    # Optional: To prevent notebook crash on long videos, 
    # you can show only every 30th frame (e.g., if i % 30 == 0:)
    if i % 30 == 0: 
        plt.figure(figsize=(6, 6))
        plt.imshow(im)
        plt.axis('off')
        plt.title(f"Car Perspective: {car_side}")
        plt.show()
    
    # print every 30 frames to keep the terminal clean
    if i % 30 == 0:
        print(f"Frame {i} - Detected: {raw_label} -> Perspective: {car_side}")

print(f"Processing complete. Results saved to: {results[0].save_dir}")