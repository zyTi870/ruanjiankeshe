import os
import shutil
from docx import Document
from docx.shared import Pt

# Configuration
docx_path = '课设报告-AI器官芯片毒性显微检测平台.docx'
media_dir = 'media'
output_html = 'index.html'
css_file = 'styles.css'

# Ensure media directory exists
if os.path.exists(media_dir):
    shutil.rmtree(media_dir)
os.makedirs(media_dir)

# Helper to save image from relationship ID
def save_image_from_rid(doc, run, media_dir):
    # Find all drawing elements in the run
    drawings = run.element.xpath('.//w:drawing')
    saved_images = []
    
    for drawing in drawings:
        # Find blip (image reference)
        blips = drawing.xpath('.//a:blip')
        for blip in blips:
            rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            if rId:
                try:
                    part = doc.part.related_parts[rId]
                    # Check if it's an image
                    if 'image' in part.content_type:
                        # Generate a filename
                        image_filename = f"image_{rId}.{part.content_type.split('/')[-1]}"
                        # Fix some extensions
                        if image_filename.endswith('.jpeg'): image_filename = image_filename.replace('.jpeg', '.jpg')
                        if image_filename.endswith('.x-wmf'): image_filename = image_filename.replace('.x-wmf', '.wmf')
                        
                        image_path = os.path.join(media_dir, image_filename)
                        
                        # Save content
                        with open(image_path, 'wb') as f:
                            f.write(part.blob)
                        
                        saved_images.append(image_filename)
                except KeyError:
                    print(f"Warning: Could not find part for rId {rId}")
                    pass
    
    # Also check for 'pict' (older format images)
    picts = run.element.xpath('.//w:pict')
    for pict in picts:
        imagedatas = pict.xpath('.//v:imagedata')
        for imagedata in imagedatas:
            rId = imagedata.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
            if rId:
                 try:
                    part = doc.part.related_parts[rId]
                    if 'image' in part.content_type:
                        image_filename = f"image_{rId}.{part.content_type.split('/')[-1]}"
                        image_path = os.path.join(media_dir, image_filename)
                        with open(image_path, 'wb') as f:
                            f.write(part.blob)
                        saved_images.append(image_filename)
                 except KeyError:
                    pass

    return saved_images

# Step 1: Read Text and Generate HTML
print("Reading document and generating HTML...")
doc = Document(docx_path)

html_content = []
html_content.append('<!DOCTYPE html>')
html_content.append('<html lang="zh-CN">')
html_content.append('<head>')
html_content.append('<meta charset="UTF-8">')
html_content.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
html_content.append('<title>AI器官芯片毒性显微检测平台 - 实验报告</title>')
html_content.append('<link rel="stylesheet" href="styles.css">')
html_content.append('</head>')
html_content.append('<body>')
html_content.append('<div class="container">')

# Filter regex for sensitive info
import re
sensitive_patterns = [
    r'姓名[:：\s]*[\u4e00-\u9fa5]{2,4}',
    r'学号[:：\s]*\d+',
    r'成\s*员[:：].*',
    r'指导教师[:：].*',
    r'任课教师[:：].*'
]

def is_sensitive(text):
    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            return True
    return False

# Iterate paragraphs
for para in doc.paragraphs:
    # Process text
    text = para.text.strip()
    
    # Skip sensitive info
    if is_sensitive(text):
        continue

    # Determine tag based on style
    style_name = para.style.name
    tag = 'p'
    if 'Heading 1' in style_name:
        tag = 'h1'
    elif 'Heading 2' in style_name:
        tag = 'h2'
    elif 'Heading 3' in style_name:
        tag = 'h3'
    elif 'Heading 4' in style_name:
        tag = 'h4'
    elif 'Title' in style_name:
        tag = 'h1' # Treat Title as H1
    
    # Check for images in runs
    para_images = []
    for run in para.runs:
        images = save_image_from_rid(doc, run, media_dir)
        para_images.extend(images)
    
    # If paragraph has text, write it
    if text:
        html_content.append(f'<{tag}>{text}</{tag}>')
    
    # Insert images immediately after the paragraph (or inside if we wanted, but after is safer for layout)
    # If the paragraph was empty but had images, they will just appear.
    for img in para_images:
        html_content.append(f'<div class="image-container"><img src="{media_dir}/{img}" alt="实验图片" loading="lazy"></div>')

html_content.append('</div>') # container
html_content.append('</body>')
html_content.append('</html>')

with open(output_html, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_content))

print(f"Generated {output_html}")
