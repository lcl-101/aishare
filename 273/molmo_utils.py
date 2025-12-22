"""
Utility functions for Molmo2 processing.
"""

from typing import Optional
from PIL import Image
import requests
from io import BytesIO


def process_vision_info(messages: list[dict]) -> tuple[list, list, dict]:
    """
    Process vision information from messages.
    
    This function extracts images and videos from the message format used in chat templates.
    
    Args:
        messages: List of message dictionaries containing content with images/videos.
        
    Returns:
        Tuple of (images, videos, video_kwargs)
        - images: List of (PIL.Image, metadata) tuples
        - videos: List of (video_path/url, metadata) tuples
        - video_kwargs: Additional keyword arguments for video processing
    """
    images = []
    videos = []
    video_kwargs = {}
    
    for message in messages:
        content = message.get("content", [])
        if isinstance(content, str):
            continue
            
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                
                if item_type == "image":
                    image = item.get("image")
                    if isinstance(image, str):
                        # URL or file path
                        if image.startswith("http://") or image.startswith("https://"):
                            response = requests.get(image, stream=True)
                            image = Image.open(BytesIO(response.content))
                        else:
                            image = Image.open(image)
                    
                    metadata = {
                        "width": image.width,
                        "height": image.height,
                    }
                    images.append((image, metadata))
                    
                elif item_type == "video":
                    video = item.get("video")
                    # Video can be URL, file path, or numpy array
                    metadata = {
                        "width": item.get("width"),
                        "height": item.get("height"),
                    }
                    videos.append((video, metadata))
                    
                    # Extract any video-specific kwargs
                    if "fps" in item:
                        video_kwargs["fps"] = item["fps"]
                    if "num_frames" in item:
                        video_kwargs["num_frames"] = item["num_frames"]
    
    return images, videos, video_kwargs
