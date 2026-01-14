import os
import shutil
import re
from docx import Document
from PIL import Image

# Configuration
docx_path = '课设报告-AI器官芯片毒性显微检测平台.docx'
media_dir = 'media'
output_html = 'index.html'
css_file = 'styles.css'
code_dir_link = 'https://github.com/zyTi870/ruanjiankeshe/tree/main/code' # GitHub link

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
                        
                        # SKIP image_rId7.jpg
                        if 'image_rId7.jpg' in image_filename:
                            continue

                        image_path = os.path.join(media_dir, image_filename)
                        
                        # Save content
                        with open(image_path, 'wb') as f:
                            f.write(part.blob)
                        
                        # Process image_rId8.png
                        if 'image_rId8.png' in image_filename:
                            try:
                                with Image.open(image_path) as img:
                                    # Rotate 90 degrees (make horizontal if vertical)
                                    # "横过来" -> Usually rotate -90 (270) or 90
                                    # Let's rotate -90 (counter-clockwise) which is common for landscape
                                    img = img.rotate(-90, expand=True)
                                    
                                    # Resize "改小一点" - 50%
                                    width, height = img.size
                                    new_size = (int(width * 0.5), int(height * 0.5))
                                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                                    
                                    img.save(image_path)
                            except Exception as e:
                                print(f"Error processing {image_filename}: {e}")

                        saved_images.append(image_filename)
                except KeyError:
                    print(f"Warning: Could not find part for rId {rId}")
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

# Header Information Pattern
# We'll treat the first few paragraphs (before "课设背景") as header info
header_mode = True
header_content = []

# Filter regex for sensitive info
sensitive_patterns = [
    r'姓名[:：\s]*[\u4e00-\u9fa5]{2,4}',
    r'学号[:：\s]*\d+',
    r'成\s*员[:：].*',
    r'指导教师[:：].*',
    r'任课教师[:：].*'
]
# Specific text to remove
remove_text = "请你在最终提交时将路径与脚本名替换为你项目中的真实名称"

def is_sensitive(text):
    for pattern in sensitive_patterns:
        if re.search(pattern, text):
            return True
    return False

# Iterate paragraphs
for para in doc.paragraphs:
    # Process text
    text = para.text.strip()
    
    # Filter specific text
    if remove_text in text:
        text = text.replace(remove_text, "")
        if not text.strip(): continue # Skip if empty after removal

    # Skip sensitive info
    if is_sensitive(text):
        continue

    # Determine tag based on style
    style_name = para.style.name
    tag = 'p'
    if 'Heading 1' in style_name:
        tag = 'h1'
        header_mode = False # Stop header mode on first H1
    elif 'Heading 2' in style_name:
        tag = 'h2'
        header_mode = False
    elif 'Heading 3' in style_name:
        tag = 'h3'
    elif 'Heading 4' in style_name:
        tag = 'h4'
    elif 'Title' in style_name:
        tag = 'h1' 
    
    # Check for images in runs
    para_images = []
    for run in para.runs:
        images = save_image_from_rid(doc, run, media_dir)
        para_images.extend(images)
    
    # Logic for Header Mode
    # If text matches typical header info, keep in header mode
    if header_mode and text:
        if "课设背景" in text or "1.1" in text:
            header_mode = False
        else:
            # Accumulate header paragraphs
            # Special styling for specific lines
            if "AI器官芯片" in text:
                header_content.append(f'<h1 class="report-title">{text}</h1>')
            elif "中国农业大学" in text or "课程设计" in text:
                 header_content.append(f'<div class="header-info-large">{text}</div>')
            else:
                 header_content.append(f'<div class="header-info">{text}</div>')
            continue # Skip adding to main body yet

    # If we just exited header mode, flush header content
    if not header_mode and header_content:
        html_content.append('<header class="report-header">')
        html_content.append('\n'.join(header_content))
        # Add Code Link
        html_content.append(f'<div class="code-link"><a href="{code_dir_link}" target="_blank">📂 查看项目完整代码 (Project Code)</a></div>')
        html_content.append('</header>')
        header_content = [] # Clear

    # If paragraph has text, write it
    if text:
        html_content.append(f'<{tag}>{text}</{tag}>')
    
    # Insert images
    for img in para_images:
        html_content.append(f'<div class="image-container"><img src="{media_dir}/{img}" alt="实验图片" loading="lazy"></div>')

html_content.append('</div>') # container
html_content.append('</body>')
html_content.append('</html>')

with open(output_html, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_content))

print(f"Generated {output_html}")
