#!/usr/bin/env python3  
# cd dreamzero
# python -m eval_utils.model_inference_from_image_and_prompt
import numpy as np  
import uuid  
from eval_utils.policy_client import WebsocketClientPolicy  
  
def load_and_preprocess_image(image_path):  
    """Load your image and convert to required format"""  
    # Use your preferred image loading method (OpenCV, PIL, etc.)  
    # Ensure output is (H, W, 3) uint8 numpy array  
    img = cv2.imread(image_path)  # Example with OpenCV  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img  
  
def run_inference_with_custom_images(api_host, api_port, right_img_path, left_img_path, wrist_img_path, prompt):  
    # Connect to server  
    client = WebsocketClientPolicy(host=api_host, port=api_port)  
      
    # Load your images  
    right_image = load_and_preprocess_image(right_img_path)  
    left_image = load_and_preprocess_image(left_img_path)    
    wrist_image = load_and_preprocess_image(wrist_img_path)  
      
    # Create observation  
    obs = {  
        "observation/exterior_image_0_left": right_image,  
        "observation/exterior_image_1_left": left_image,  
        "observation/wrist_image_left": wrist_image,  
        "observation/joint_position": np.zeros(7, dtype=np.float64),  
        "observation/cartesian_position": np.zeros(6, dtype=np.float64),   
        "observation/gripper_position": np.zeros(1, dtype=np.float64),  
        "prompt": prompt,  
        "session_id": str(uuid.uuid4())  
    }  
      
    # Send inference request  
    actions = client.infer(obs)  
    print(f"Received actions with shape: {actions.shape}")  
      
    # Server automatically generates and saves videos  
    # Videos are saved to the server's output directory  
      
    return actions  
  
# Usage  
actions = run_inference_with_custom_images(  
    api_host="your-api-host.com",  
    api_port=8000,  
    right_img_path="path/to/right_camera.jpg",  
    left_img_path="path/to/left_camera.jpg",   
    wrist_img_path="path/to/wrist_camera.jpg",  
    prompt="pick up the red block and place it in the bowl"  
)