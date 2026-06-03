def strip_leading_blank_lines(lines):
    i = 0
    while i < len(lines) and lines[i].strip() == '':
        i += 1
    return lines[i:]


def strip_main_comment(lines):
    if lines and lines[0].strip() == '# main.py':
        return strip_leading_blank_lines(lines[1:])
    return lines


def strip_embed_comment(lines):
    out = list(lines)
    for _ in range(2):
        if out and (out[0].lstrip().startswith('#') or out[0].strip() == ''):
            out = out[1:]
        else:
            break
    return strip_leading_blank_lines(out)
