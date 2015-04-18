def strip_newsgroup_header(self, text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """

    _before, _blankline, after = text.partition('\n\n')
    return after


def strip_newsgroup_quoting(self, text):
    """
    Given text in "news" format, strip lines beginning with the quote
    characters > or |, plus lines that often introduce a quoted section
    (for example, because they contain the string 'writes:'.)
    """
    _QUOTE_RE = re.compile(r'(writes in|writes:|wrote:|says:|said:'
                           r'|^In article|^Quoted from|^\||^>)')
    good_lines = [line for line in text.split('\n')
                  if not _QUOTE_RE.search(line)]
    return '\n'.join(good_lines)


def strip_newsgroup_footer(self, text):
    """
    Given text in "news" format, attempt to remove a signature block.
    As a rough heuristic, we assume that signatures are set apart by either
    a blank line or a line made of hyphens, and that it is the last such line
    in the file (disregarding blank lines at the end).
    """
    lines = text.strip().split('\n')
    for line_num in range(len(lines) - 1, -1, -1):
        line = lines[line_num]
        if line.strip().strip('-') == '':
            break

    if line_num > 0:
        return '\n'.join(lines[:line_num])
    else:
        return text


def run():
    s = '../../../data/newsgroups/20_newsgroups'
    files = os.listdir(s)
    files.sort()
    fi = files.pop(0)
    s_c = s + fi + '/'
    paths = os.listdir(s_c)
    dt = text_label()
    for path in paths:

        fin = open(s_c + path, 'r')
        data = fin.read()  # need to change the where the is stored
        data = (data.decode('latin1'))

        data = dt.strip_newsgroup_header(data)
        data = dt.strip_newsgroup_footer(data)
        data = dt.strip_newsgroup_quoting(data)
        dt.add(data, 0)  # data and target
        target = 0
    #num = 0
    for f in files:
        s_c = s + f + '/'
        paths = os.listdir(s_c)
        for path in paths:
            fin = open(s_c + path, 'r')
            data = fin.read()  # need to c
            data = (data.decode('latin1'))
            data = dt.strip_newsgroup_header(data)
            data = dt.strip_newsgroup_footer(data)
            data = dt.strip_newsgroup_quoting(data)
            dt.add(data, 1)
            target = 1
        #num += 1
    return dt
