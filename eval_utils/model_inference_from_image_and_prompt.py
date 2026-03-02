"""
# Install 
  cd dreamzero
  # torch cpu
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r eval_utils\requirements.model_inference.txt
# Inference:
  cd dreamzero
  python -m eval_utils.model_inference_from_image_and_prompt \
        --policy-server-host "localhost" \
        --policy-server-port 8000 \
        --right-img-path "path/to/right_camera.jpg" \
        --left-img-path "path/to/left_camera.jpg" \
        --wrist-img-path "path/to/wrist_camera.jpg" \
        --prompt "pick up the red block and place it in the bowl"
    
"""
import argparse
import cv2
import numpy as np  
import uuid  
import torch
from PIL import Image
from abc import ABC, abstractmethod

from eval_utils.policy_client import WebsocketClientPolicy  
from openpi_client import image_tools


class InferenceClient(ABC):
    @abstractmethod
    def __init__(self, args) -> None:
        """
        Initializes the client.
        """
        pass

    @abstractmethod
    def infer(self, obs, instruction) -> dict:
        """
        Does inference on observation and returns the final processed
        dictionary used to do inference.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the client to start a new episode.
        """
        pass


class DreamZeroJointPosClient(InferenceClient):
    def __init__(self, 
                remote_host:str = "localhost", 
                remote_port:int = 6000,
                open_loop_horizon:int = 8,
    ) -> None:
        self.client = WebsocketClientPolicy(host=remote_host, port=remote_port)
        self.open_loop_horizon = open_loop_horizon
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        right_img = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        wrist_img = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        left_img = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        combined = np.concatenate([right_img, wrist_img, left_img], axis=1)
        return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.session_id = str(uuid.uuid4())

    def infer(self, obs: dict, instruction: str) -> dict:
        """
        Infer the next action from the policy in a server-client setup
        """
        curr_obs = self._extract_observation(obs)
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            self.actions_from_chunk_completed = 0
            request_data = {
                "observation/exterior_image_0_left": image_tools.resize_with_pad(curr_obs["right_image"], 180, 320),
                "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs["left_image"], 180, 320),
                "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 180, 320),
                "observation/joint_position": curr_obs["joint_position"].astype(np.float64),
                "observation/cartesian_position": np.zeros((6,), dtype=np.float64),  # dummy cartesian position
                "observation/gripper_position": curr_obs["gripper_position"].astype(np.float64),
                "prompt": instruction,
                "session_id": self.session_id,
            }
            for k, v in request_data.items():
                print(f"{k}: {v.shape if not isinstance(v, str) else v}")
            
            actions = self.client.infer(request_data)
            assert len(actions.shape) == 2, f"Expected 2D array, got shape {actions.shape}"
            assert actions.shape[-1] == 8, f"Expected 8 action dimensions (7 joints + 1 gripper), got {actions.shape[-1]}"
            self.pred_action_chunk = actions


        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        img1 = image_tools.resize_with_pad(curr_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        img3 = image_tools.resize_with_pad(curr_obs["left_image"], 224, 224)
        both = np.concatenate([img1, img2, img3], axis=1)

        return {"action": action, "viz": both}

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        # Assign images
        right_image = obs_dict["policy"]["external_cam"][0].clone().detach().cpu().numpy()
        left_image = obs_dict["policy"]["external_cam_2"][0].clone().detach().cpu().numpy()
        wrist_image = obs_dict["policy"]["wrist_cam"][0].clone().detach().cpu().numpy()

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "left_image": left_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }

def load_and_preprocess_image(image_path):  
    """Load your image and convert to required format"""  
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    return img  
  
def run_inference_with_custom_images(policy_server_host, policy_server_port, right_img_path, left_img_path, wrist_img_path, prompt):  
    # Connect to server using the new client
    client = DreamZeroJointPosClient(remote_host=policy_server_host, remote_port=policy_server_port)  
      
    # Load your images  
    right_image = load_and_preprocess_image(right_img_path)  
    left_image = load_and_preprocess_image(left_img_path)    
    wrist_image = load_and_preprocess_image(wrist_img_path)  
      
    # Create observation structure containing PyTorch tensors
    # formatted specifically for _extract_observation to unpack
    obs = {  
        "policy": {
            "external_cam": [torch.from_numpy(right_image)],
            "external_cam_2":[torch.from_numpy(left_image)],
            "wrist_cam":[torch.from_numpy(wrist_image)],
            "arm_joint_pos": torch.zeros(7, dtype=torch.float64),
            "gripper_pos": torch.zeros(1, dtype=torch.float64)
        }
    }  
      
    # Send inference request (handles chunked reasoning & resizing inherently)
    result = client.infer(obs, instruction=prompt)  
    
    action = result["action"]
    print(f"Received action with shape: {action.shape}")  
    
    # Optional: you can save or show `result["viz"]` here if desired.
      
    return action  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model inference with custom images and a prompt.")
    
    parser.add_argument("--policy-server-host", type=str, default="your-api-host.com", 
                        help="Host address for the Websocket Policy Server (default: your-api-host.com)")
    parser.add_argument("--policy-server-port", type=int, default=8000, 
                        help="Port for the Websocket Policy Server (default: 8000)")
    parser.add_argument("--right-img-path", type=str, required=True, 
                        help="File path to the right camera image")
    parser.add_argument("--left-img-path", type=str, required=True, 
                        help="File path to the left camera image")
    parser.add_argument("--wrist-img-path", type=str, required=True, 
                        help="File path to the wrist camera image")
    parser.add_argument("--prompt", type=str, required=True, 
                        help="Instruction prompt (e.g., 'pick up the red block and place it in the bowl')")

    args = parser.parse_args()

    # Usage  
    actions = run_inference_with_custom_images(  
        policy_server_host=args.policy_server_host,  
        policy_server_port=args.policy_server_port,  
        right_img_path=args.right_img_path,  
        left_img_path=args.left_img_path,   
        wrist_img_path=args.wrist_img_path,  
        prompt=args.prompt  
    )