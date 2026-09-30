# -*- coding: utf-8 -*-
"""提取正式题面 docx 文本（仅用于核对附件 C 的角色规定）"""
import zipfile
import re

P = r'd:\F题\F题\算力约束下提升大语言模型能力的资源配置建模.docx'
with zipfile.ZipFile(P) as z:
    xml = z.read('word/document.xml').decode('utf-8', errors='ignore')

# 按段落切分
paras = re.findall(r'<w:p[ >].*?</w:p>', xml, flags=re.S)
out = []
for p in paras:
    texts = re.findall(r'<w:t[^>]*>(.*?)</w:t>', p, flags=re.S)
    s = ''.join(texts)
    s = (s.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
         .replace('&quot;', '"').replace('&#39;', "'"))
    if s.strip():
        out.append(s.strip())

print(f'段落数 = {len(out)}')
for i, s in enumerate(out):
    print(f'[{i:>3}] {s}')
