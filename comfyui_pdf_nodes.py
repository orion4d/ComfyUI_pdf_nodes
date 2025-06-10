import os
import logging
import io # Pour les flux en mémoire
import re # Pour l'analyse de la sélection de pages
import numpy as np
import torch # ComfyUI utilise PyTorch pour les tenseurs d'image

from pypdf import PdfReader, PdfWriter, PageObject # MODIFIÉ ICI
from pypdf.errors import FileNotDecryptedError
# from pypdf.page import PageObject # ANCIENNE LIGNE À SUPPRIMER SI PRÉSENTE

# Tenter d'importer PyMuPDF (fitz) et Pillow
try:
    import fitz  # PyMuPDF
    from PIL import Image
    PYMUPDF_INSTALLED = True
except ImportError:
    PYMUPDF_INSTALLED = False
    print("--------------------------------------------------------------------")
    print("WARNING: PyMuPDF (fitz) or Pillow not installed.")
    print("PDFNodes requiring image rendering (Preview, PagesToImages) will be disabled.")
    print("Please install them: pip install PyMuPDF Pillow")
    print("--------------------------------------------------------------------")

# Setup basic logging
logger = logging.getLogger(__name__)

# Custom type for clarity, essentially List[PageObject] from pypdf
PDF_PAGES = "PDF_PAGES"
MAX_PAGES_TO_RENDER_LIMIT = 100 # Hard limit

# --- Helper Function for Page Selection Parsing ---
def parse_page_selection(selection_str: str, total_pages: int, max_limit: int = MAX_PAGES_TO_RENDER_LIMIT) -> list[int]:
    """
    Parses a page selection string (1-indexed) into a list of 0-indexed page numbers.
    Examples: "1,3-5,8", "all", "1-10"
    Clips selection to total_pages and enforces max_limit.
    """
    if total_pages == 0:
        return []

    selected_indices = set()
    # Normalize "all" or empty string to select all pages
    if not selection_str or selection_str.lower().strip() == "all":
        selection_str = f"1-{total_pages}"

    parts = selection_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-', 1))
                # Convert to 0-indexed and ensure start <= end
                start_idx = max(0, start - 1)
                end_idx = min(total_pages - 1, end - 1)
                for i in range(start_idx, end_idx + 1):
                    selected_indices.add(i)
            else:
                page_num = int(part)
                page_idx = page_num - 1 # Convert to 0-indexed
                if 0 <= page_idx < total_pages:
                    selected_indices.add(page_idx)
        except ValueError:
            print(f"PDFNodes Helper: Invalid page selection part: '{part}'")
            continue # Skip invalid parts

    # Sort and limit
    sorted_indices = sorted(list(selected_indices))
    if len(sorted_indices) > max_limit:
        print(f"PDFNodes Helper: Page selection exceeds maximum of {max_limit}. Truncating.")
        return sorted_indices[:max_limit]
    return sorted_indices


class PDFLoadNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pdf_path": ("STRING", {"default": "input/example.pdf", "multiline": False})},
            "optional": {"password": ("STRING", {"default": "", "multiline": False})}
        }
    RETURN_TYPES = (PDF_PAGES, "INT") # Return pages and page count
    RETURN_NAMES = ("pdf_pages", "page_count")
    FUNCTION = "load"
    CATEGORY = "PDF"

    def load(self, pdf_path, password=None):
        if not os.path.exists(pdf_path):
            error_msg = f"PDFLoadNode: File not found at {pdf_path}"
            logger.error(error_msg); print(f"ERROR: {error_msg}")
            return ([], 0)
        try:
            effective_password = password if password and password.strip() else None
            reader = PdfReader(pdf_path, password=effective_password)
            pages = list(reader.pages)
            page_count = len(pages)
            print(f"PDFLoadNode: Loaded {page_count} pages from {pdf_path}")
            return (pages, page_count)
        except FileNotDecryptedError:
            error_msg = f"PDFLoadNode: Could not decrypt PDF at {pdf_path}. Is the password correct?"
            logger.error(error_msg); print(f"ERROR: {error_msg}")
            return ([], 0)
        except Exception as e:
            error_msg = f"PDFLoadNode: Error loading PDF {pdf_path}: {e}"
            logger.error(error_msg); print(f"ERROR: {error_msg}")
            return ([], 0)

class PDFMergeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"pdf_pages_a": (PDF_PAGES,), "pdf_pages_b": (PDF_PAGES,)}
        }
    RETURN_TYPES = (PDF_PAGES,)
    FUNCTION = "merge"
    CATEGORY = "PDF"

    def merge(self, pdf_pages_a, pdf_pages_b):
        writer = PdfWriter()
        pages_a = pdf_pages_a if pdf_pages_a is not None else []
        pages_b = pdf_pages_b if pdf_pages_b is not None else []
        for page in pages_a: writer.add_page(page)
        for page in pages_b: writer.add_page(page)
        merged_pages = list(writer.pages)
        print(f"PDFMergeNode: Merged {len(pages_a)} + {len(pages_b)} pages into {len(merged_pages)} pages.")
        return (merged_pages,)

class PDFSelectPageAndExtractTextNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "page_index": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}),
            }
        }
    RETURN_TYPES = (PDF_PAGES, "STRING")
    RETURN_NAMES = ("selected_page_as_pdf_pages", "extracted_text")
    FUNCTION = "select_page"
    CATEGORY = "PDF"

    def select_page(self, pdf_pages: list[PageObject], page_index: int):
        if not pdf_pages:
            print("PDFSelectPageAndExtractTextNode: Input PDF pages list is empty.")
            return ([], "")
        num_pages = len(pdf_pages)
        if not (0 <= page_index < num_pages):
            error_msg = f"PDFSelectPageAndExtractTextNode: Page index {page_index} is out of bounds (0-{num_pages-1}). Returning empty."
            logger.warning(error_msg); print(f"WARNING: {error_msg}")
            return ([], "")
        selected_page_obj = pdf_pages[page_index]
        text = ""
        try:
            text = selected_page_obj.extract_text() if hasattr(selected_page_obj, 'extract_text') else ''
            if not text: text = ""
        except Exception as e:
            print(f"PDFSelectPageAndExtractTextNode: Error extracting text from page {page_index}: {e}")
            text = f"[Error extracting text: {e}]"
        print(f"PDFSelectPageAndExtractTextNode: Selected page {page_index}. Extracted {len(text)} chars.")
        return ([selected_page_obj], text,)

class PDFExtractTextFromPagesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "page_selection": ("STRING", {"default": "all", "multiline": False, "placeholder": "e.g., 1,3-5,all (1-indexed)"}),
            },
            "optional": { "separator": ("STRING", {"default": "\n\n--- Page Break ---\n\n"}) }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "extract_text_from_selected_pages"
    CATEGORY = "PDF"

    def extract_text_from_selected_pages(self, pdf_pages: list[PageObject], page_selection: str, separator="\n\n--- Page Break ---\n\n"):
        if not pdf_pages:
            print("PDFExtractTextFromPagesNode: Input PDF pages list is empty.")
            return ("",)
        
        total_doc_pages = len(pdf_pages)
        selected_indices = parse_page_selection(page_selection, total_doc_pages, max_limit=total_doc_pages) # No render limit for text

        if not selected_indices:
            print("PDFExtractTextFromPagesNode: No pages selected or invalid selection.")
            return ("",)

        extracted_texts = []
        for i in selected_indices:
            page = pdf_pages[i]
            try:
                text = page.extract_text() if hasattr(page, 'extract_text') else ''
                if text is None: text = ""
                extracted_texts.append(text)
            except Exception as e:
                print(f"PDFExtractTextFromPagesNode: Error extracting text from page {i}: {e}")
                extracted_texts.append(f"[Error extracting text from page {i}: {e}]")
        
        full_text = separator.join(extracted_texts)
        print(f"PDFExtractTextFromPagesNode: Extracted text from {len(selected_indices)} selected pages. Total {len(full_text)} chars.")
        return (full_text,)

class PDFSaveNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "output_directory": ("STRING", {"default": "output/"}),
                "filename_prefix": ("STRING", {"default": "comfy_pdf"}),
                "overwrite_existing": (["enable", "disable"], {"default": "disable"}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_pdf_path",)
    FUNCTION = "save"
    CATEGORY = "PDF"
    OUTPUT_NODE = True

    def save(self, pdf_pages: list[PageObject], output_directory: str, filename_prefix: str, overwrite_existing: str):
        if not pdf_pages:
            print("PDFSaveNode: No PDF pages provided to save.")
            return ("",)
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory, exist_ok=True)
                print(f"PDFSaveNode: Created output directory: {output_directory}")
            except Exception as e:
                print(f"ERROR: PDFSaveNode: Could not create output directory {output_directory}: {e}")
                return ("",)
        filename = f"{filename_prefix}.pdf"
        output_path = os.path.join(output_directory, filename)
        if overwrite_existing == "disable" and os.path.exists(output_path):
            counter = 1
            base_filename, _ = os.path.splitext(filename_prefix)
            while os.path.exists(output_path):
                filename = f"{base_filename}_{counter}.pdf"
                output_path = os.path.join(output_directory, filename)
                counter += 1
            print(f"PDFSaveNode: File existed, saving as {output_path}")
        writer = PdfWriter()
        for page in pdf_pages: writer.add_page(page)
        try:
            with open(output_path, "wb") as f: writer.write(f)
            print(f"PDFSaveNode: Successfully saved {len(pdf_pages)} pages to {output_path}")
            return (output_path,)
        except Exception as e:
            error_msg = f"PDFSaveNode: Error saving PDF to {output_path}: {e}"
            logger.error(error_msg); print(f"ERROR: {error_msg}")
            return ("",)

class PDFGetPageCountNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pdf_pages": (PDF_PAGES,)}}
    RETURN_TYPES = ("INT",)
    FUNCTION = "get_page_count"
    CATEGORY = "PDF"

    def get_page_count(self, pdf_pages: list[PageObject]):
        if pdf_pages is None: return (0,)
        count = len(pdf_pages)
        print(f"PDFGetPageCountNode: PDF has {count} pages.")
        return (count,)

class PDFPreviewPageNode: # For single page quick preview
    @classmethod
    def INPUT_TYPES(cls):
        if not PYMUPDF_INSTALLED:
            return {"required": {"warning": ("STRING", {"default": "PyMuPDF or Pillow not installed. Node disabled.", "multiline": True})}}
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "page_index": ("INT", {"default": 0, "min": 0, "step": 1, "display": "number"}), # 0-indexed
                "dpi": ("INT", {"default": 150, "min": 72, "max": 600, "step": 1, "display": "number"}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preview_single_page"
    CATEGORY = "PDF"
    OUTPUT_NODE = True

    def _render_page_obj_to_tensor(self, page_obj: PageObject, dpi: int):
        # Helper for rendering a single pypdf PageObject to a tensor
        temp_pdf_writer = PdfWriter()
        temp_pdf_writer.add_page(page_obj)
        img_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32) # Default blank
        with io.BytesIO() as temp_pdf_stream:
            temp_pdf_writer.write(temp_pdf_stream)
            temp_pdf_stream.seek(0)
            try:
                doc = fitz.open(stream=temp_pdf_stream.read(), filetype="pdf")
                if doc.page_count > 0:
                    page_fitz = doc.load_page(0)
                    pix = page_fitz.get_pixmap(dpi=dpi)
                    img_pil = Image.frombytes("RGBA" if pix.alpha else "RGB", [pix.width, pix.height], pix.samples)
                    img_np = np.array(img_pil).astype(np.float32) / 255.0
                    img_tensor = torch.from_numpy(img_np).unsqueeze(0)
                doc.close()
            except Exception as e:
                print(f"ERROR (PDFPreviewPageNode): Error rendering PDF page with PyMuPDF: {e}")
        return img_tensor

    def preview_single_page(self, pdf_pages: list[PageObject], page_index: int, dpi: int, **kwargs):
        if not PYMUPDF_INSTALLED:
            print("ERROR: PDFPreviewPageNode requires PyMuPDF (fitz) and Pillow to be installed.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        if not pdf_pages:
            print("PDFPreviewPageNode: Input PDF pages list is empty.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        num_pages_total = len(pdf_pages)
        if not (0 <= page_index < num_pages_total):
            print(f"PDFPreviewPageNode: Page index {page_index} is out of bounds (0-{num_pages_total-1}). Previewing page 0 if available.")
            page_index = 0
            if not (0 <= page_index < num_pages_total):
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        page_to_render = pdf_pages[page_index]
        img_tensor = self._render_page_obj_to_tensor(page_to_render, dpi)
        print(f"PDFPreviewPageNode: Rendered page index {page_index} at {dpi} DPI. Shape: {img_tensor.shape}")
        return (img_tensor,)

class PDFPagesToImagesNode: # New node for multi-page rendering
    @classmethod
    def INPUT_TYPES(cls):
        if not PYMUPDF_INSTALLED:
            return {"required": {"warning": ("STRING", {"default": "PyMuPDF or Pillow not installed. Node disabled.", "multiline": True})}}
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "page_selection": ("STRING", {"default": "all", "multiline": False, "placeholder": "e.g., 1,3-5,all (1-indexed)"}),
                "dpi": ("INT", {"default": 150, "min": 72, "max": 600, "step": 1}),
                "max_pages_to_render": ("INT", {"default": 10, "min": 1, "max": MAX_PAGES_TO_RENDER_LIMIT, "step": 1}),
            }
        }
    RETURN_TYPES = ("IMAGE",) # Returns a batch of images
    FUNCTION = "render_selected_pages_to_images"
    CATEGORY = "PDF"
    OUTPUT_NODE = True

    def render_selected_pages_to_images(self, pdf_pages: list[PageObject], page_selection: str, dpi: int, max_pages_to_render: int, **kwargs):
        if not PYMUPDF_INSTALLED:
            print("ERROR: PDFPagesToImagesNode requires PyMuPDF (fitz) and Pillow to be installed.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),) # Return a single blank image
        if not pdf_pages:
            print("PDFPagesToImagesNode: Input PDF pages list is empty.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        total_doc_pages = len(pdf_pages)
        # Use max_pages_to_render from input as the limit for parse_page_selection for this node
        selected_indices = parse_page_selection(page_selection, total_doc_pages, max_limit=min(max_pages_to_render, MAX_PAGES_TO_RENDER_LIMIT))

        if not selected_indices:
            print("PDFPagesToImagesNode: No pages selected or invalid selection for rendering.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        image_tensors_list = []
        preview_node_renderer = PDFPreviewPageNode() # Instantiate to use its _render_page_obj_to_tensor

        for i, page_idx in enumerate(selected_indices):
            if i >= max_pages_to_render : # Double check limit from widget
                print(f"PDFPagesToImagesNode: Reached max_pages_to_render limit ({max_pages_to_render}). Stopping.")
                break
            page_obj = pdf_pages[page_idx]
            img_tensor = preview_node_renderer._render_page_obj_to_tensor(page_obj, dpi) # B, H, W, C
            image_tensors_list.append(img_tensor)
        
        if not image_tensors_list:
            print("PDFPagesToImagesNode: No images were rendered.")
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)

        batched_images = torch.cat(image_tensors_list, dim=0)
        print(f"PDFPagesToImagesNode: Rendered {batched_images.shape[0]} pages into a batch. Shape: {batched_images.shape}")
        return (batched_images,)

class PDFRotatePagesNode: # New node for rotation
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pdf_pages": (PDF_PAGES,),
                "page_selection": ("STRING", {"default": "all", "multiline": False, "placeholder": "e.g., 1,3-5,all (1-indexed)"}),
                "rotation_angle": ([0, 90, 180, 270], {"default": 0}), # pypdf .rotate() adds
            }
        }
    RETURN_TYPES = (PDF_PAGES,)
    FUNCTION = "rotate_selected_pages"
    CATEGORY = "PDF"

    def rotate_selected_pages(self, pdf_pages: list[PageObject], page_selection: str, rotation_angle: int):
        if not pdf_pages:
            print("PDFRotatePagesNode: Input PDF pages list is empty.")
            return ([],)
        if rotation_angle == 0: # No rotation needed
            print("PDFRotatePagesNode: Rotation angle is 0, no changes made.")
            return (pdf_pages,)

        total_doc_pages = len(pdf_pages)
        # For rotation, we can apply to all selected pages without a render limit
        selected_indices = parse_page_selection(page_selection, total_doc_pages, max_limit=total_doc_pages)

        if not selected_indices:
            print("PDFRotatePagesNode: No pages selected or invalid selection for rotation.")
            return (pdf_pages,) # Return original if no valid selection

        # Create a new list of pages to return.
        # pypdf's PageObject.rotate() modifies the page in-place.
        # If we iterate and modify, the original list's objects are changed.
        # This is often fine, but returning a new list containing these (now modified)
        # objects is a common pattern.
        
        # To ensure we don't modify the input if it's used elsewhere *before* this node
        # in a complex workflow, we should ideally work on copies if pypdf didn't modify in place.
        # Since it *does* modify in place, the caller needs to be aware.
        # For simplicity here, we'll modify in place and return the same list reference
        # or a new list of the same (modified) objects.
        # Let's return a *new* list containing the potentially modified page objects.
        
        output_pages = []
        modified_count = 0
        for i, page_obj in enumerate(pdf_pages):
            if i in selected_indices:
                # Important: PageObject.rotate() adds to the current rotation.
                # For simplicity, we just apply the selected angle.
                # If you need absolute rotation, it's more complex.
                page_obj.rotate(rotation_angle) # Modifies page_obj in-place
                modified_count +=1
            output_pages.append(page_obj) # Add original or modified page

        print(f"PDFRotatePagesNode: Rotated {modified_count} selected pages by {rotation_angle} degrees.")
        return (output_pages,)


# NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
NODE_CLASS_MAPPINGS = {
    "PDFLoad": PDFLoadNode,
    "PDFMerge": PDFMergeNode,
    "PDFSelectPageAndExtractText": PDFSelectPageAndExtractTextNode,
    "PDFExtractTextFromPages": PDFExtractTextFromPagesNode,
    "PDFSave": PDFSaveNode,
    "PDFGetPageCount": PDFGetPageCountNode,
    "PDFRotatePages": PDFRotatePagesNode, # New
}
if PYMUPDF_INSTALLED:
    NODE_CLASS_MAPPINGS["PDFPreviewPage"] = PDFPreviewPageNode # Single page preview
    NODE_CLASS_MAPPINGS["PDFPagesToImages"] = PDFPagesToImagesNode # Multi-page to image batch

NODE_DISPLAY_NAME_MAPPINGS = {
    "PDFLoad": "Load PDF 📄",
    "PDFMerge": "Merge PDFs ➕📄",
    "PDFSelectPageAndExtractText": "Select Page & Extract Text 🎯📄",
    "PDFExtractTextFromPages": "Extract Text (Select Pages) 🔍📄", # Updated name
    "PDFSave": "Save PDF 💾📄",
    "PDFGetPageCount": "Get PDF Page Count #️⃣📄",
    "PDFRotatePages": "Rotate PDF Pages 🔄📄", # New
}
if PYMUPDF_INSTALLED:
    NODE_DISPLAY_NAME_MAPPINGS["PDFPreviewPage"] = "Preview PDF Page (Single) 🖼️📄" # Clarified
    NODE_DISPLAY_NAME_MAPPINGS["PDFPagesToImages"] = "PDF Pages to Images (Batch) 🖼️📄🎞️" # New

WEB_DIRECTORY = None
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']

print("------------------------------------")
print("ComfyUI PDF Nodes (v3 - Multi-Page Images & Rotation)")
print("Nodes loaded:")
for MAPPING_NAME, _ in NODE_CLASS_MAPPINGS.items():
    display_name = NODE_DISPLAY_NAME_MAPPINGS.get(MAPPING_NAME, MAPPING_NAME)
    status = ""
    if MAPPING_NAME in ["PDFPreviewPage", "PDFPagesToImages"] and not PYMUPDF_INSTALLED:
        status = " (Disabled - PyMuPDF/Pillow missing)"
    print(f"  - {display_name} ({MAPPING_NAME}){status}")
if not PYMUPDF_INSTALLED:
    print("WARNING: For PDF Preview/Image Rendering, please install PyMuPDF and Pillow:")
    print("         pip install PyMuPDF Pillow")
print("------------------------------------")