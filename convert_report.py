import os
import zipfile
import re
import shutil
import xml.etree.ElementTree as ET

# Configuration
docx_path = '课设报告-AI器官芯片毒性显微检测平台.docx'
media_dir = 'media'
output_html = 'index.html'
css_file = 'styles.css'

# Ensure media directory exists
if os.path.exists(media_dir):
    shutil.rmtree(media_dir)
os.makedirs(media_dir)

# Namespaces for parsing docx xml
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

# Step 1: Extract Media and Read XML
print("Extracting media and content...")
media_files = []
document_xml = None

with zipfile.ZipFile(docx_path, 'r') as z:
    # Extract media
    for file_info in z.infolist():
        if file_info.filename.startswith('word/media/') and not file_info.filename.endswith('/'):
            source = z.open(file_info)
            target_filename = os.path.basename(file_info.filename)
            if not target_filename:
                continue
            target_path = os.path.join(media_dir, target_filename)
            with open(target_path, 'wb') as f:
                f.write(source.read())
            media_files.append(target_filename)
            print(f"Extracted: {target_filename}")
            
    # Read document.xml
    with z.open('word/document.xml') as f:
        document_xml = f.read()

# Identify audio/video
audio_files = [f for f in media_files if f.lower().endswith(('.mp3', '.wav', '.m4a'))]
video_files = [f for f in media_files if f.lower().endswith(('.mp4', '.mov', '.avi'))]
image_files = [f for f in media_files if f not in audio_files and f not in video_files]

# Step 2: Parse XML and Generate HTML
print("Parsing XML...")
root = ET.fromstring(document_xml)

html_content = []
html_content.append('<!DOCTYPE html>')
html_content.append('<html lang="zh-CN">')
html_content.append('<head>')
html_content.append('<meta charset="UTF-8">')
html_content.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
html_content.append('<title>实验报告 - AI器官芯片毒性显微检测平台</title>')
html_content.append('<link rel="stylesheet" href="styles.css">')
html_content.append('</head>')
html_content.append('<body>')
html_content.append('<div class="container">')

# Add Audio Player
if audio_files:
    html_content.append('<div class="audio-section">')
    html_content.append('<h2>音频展示</h2>')
    for audio in audio_files:
        html_content.append(f'<audio controls><source src="{media_dir}/{audio}"></audio>')
    html_content.append('</div>')
else:
    html_content.append('<div class="audio-section">')
    html_content.append('<!-- Audio player placeholder -->')
    html_content.append('<p class="note">（此处可插入音频文件）</p>')
    html_content.append('</div>')

# Filter regex for sensitive info
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

# Iterate paragraphs in XML
body = root.find('w:body', ns)
for p in body.findall('w:p', ns):
    # Get paragraph style
    pPr = p.find('w:pPr', ns)
    style = "Normal"
    if pPr is not None:
        pStyle = pPr.find('w:pStyle', ns)
        if pStyle is not None:
            style = pStyle.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val')
    
    # Get text
    texts = [node.text for node in p.findall('.//w:t', ns) if node.text]
    text = ''.join(texts).strip()
    
    if not text:
        continue
        
    if is_sensitive(text):
        continue

    # Map style to HTML
    # Note: Styles in XML might be "1", "2", "3" or "Heading1", "Heading2" etc.
    # We check for "Heading" or single digits which often map to headings in localizations
    
    if style in ['Heading1', '1', 'Heading 1']:
        html_content.append(f'<h1>{text}</h1>')
    elif style in ['Heading2', '2', 'Heading 2']:
        html_content.append(f'<h2>{text}</h2>')
    elif style in ['Heading3', '3', 'Heading 3']:
        html_content.append(f'<h3>{text}</h3>')
    else:
        html_content.append(f'<p>{text}</p>')

# Append Media Gallery
if image_files or video_files:
    html_content.append('<div class="media-gallery">')
    html_content.append('<h2>实验图片与视频</h2>')
    
    for vid in video_files:
        html_content.append(f'<div class="media-item"><video controls width="100%"><source src="{media_dir}/{vid}"></video><p>{vid}</p></div>')
        
    for img in image_files:
        html_content.append(f'<div class="media-item"><img src="{media_dir}/{img}" alt="{img}" loading="lazy"></div>')
    
    html_content.append('</div>')

html_content.append('</div>') # container
html_content.append('</body>')
html_content.append('</html>')

with open(output_html, 'w', encoding='utf-8') as f:
    f.write('\n'.join(html_content))

print(f"Generated {output_html}")
