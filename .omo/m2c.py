def process_file(path, header_path, mode):
    with open(header_path, 'r', encoding='utf-8') as f:
        header = f.read()
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if mode == 'simple':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_leading_blank_lines(lines))
    elif mode == 'strip_main':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_main_comment(lines))
    elif mode == 'strip_embed':
        lines = content.splitlines(keepends=True)
        body = ''.join(strip_embed_comment(lines))
    elif mode == 'strip_docstring':
        body = strip_chinese_class_docstring(content)
    elif mode == 'r_shebang':
        lines = content.splitlines(keepends=True)
        if lines and lines[0].startswith('#!'):
            keep = [lines[0]]
            rest = lines[1:]
            i = 0
            while i < len(rest) and (rest[i].strip() == '' or rest[i].lstrip().startswith('#')):
                i += 1
            body = ''.join(keep + rest[i:])
        else:
            body = content
    elif mode == 'r_simple':
        lines = content.splitlines(keepends=True)
        i = 0
        while i < len(lines) and (lines[i].strip() == '' or lines[i].lstrip().startswith('#')):
            i += 1
        body = ''.join(lines[i:])
    else:
        body = content
    if not body.startswith(chr(10)):
        sep = chr(10)
    else:
        sep = ''
    new_content = header + sep + body
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return len(new_content)
