def strip_chinese_class_docstring(content):
    tq = chr(34)*3
    needle = "PyTorch Dataset"
    lines = content.splitlines(keepends=True)
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == tq and i+1 < len(lines) and needle in (lines[i+1] if i+1 < len(lines) else tq):
            new_lines.append(line)
            i += 1
            while i < len(lines):
                if lines[i].strip() == tq:
                    new_lines.append(lines[i])
                    i += 1
                    break
                i += 1
            continue
        new_lines.append(line)
        i += 1
    return chr(34).join(new_lines)
