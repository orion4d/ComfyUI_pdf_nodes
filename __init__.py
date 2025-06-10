# This file makes the nodes available to ComfyUI

# Import all node classes and mappings from your comfyui_pdf_nodes.py file
from .comfyui_pdf_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, WEB_DIRECTORY

# Export the mappings for ComfyUI to discover
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']