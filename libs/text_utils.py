import re
import pymorphy2 as pm
from natasha import Combinator, DEFAULT_GRAMMARS


def merge_directory_into_raw_text(dir_name, fnout):
    import os

    dir_name = dir_name.rstrip("/") + "/"
    corpora1 = ""

    for fname in os.listdir(dir_name):
        with open(dir_name + fname, 'r', encoding='utf-8') as fin:
            print(fname)
            text = fin.read()
            corpora1 += text

    if isinstance(fnout, str):
        with open(fnout, "w", encoding="utf-8") as f:
            f.write(corpora1)
    else:
        fnout.write(corpora1)


# да простят мне боги все это месиво из регулярок - старался не разделять прямую речь на отдельные предложения, а также
# учитывать любовь Федора Михайловича ставить терминальные знаки препинания посреди предложений
# вроде работает, но сам уже тут не разберусь, пихал сюда регулярки последовательно избавляясь от проблем и создавая новые
def split_raw_into_sentences(text):
    symbs = re.compile(r"[^А-Яа-я:!\?,\.\"— - \n]")
    text = re.sub(symbs, "", text)

    text = re.sub(r'\*|\x01|\xa0|--|\t|"|(\[.*\])', "", text)
    text = re.sub(r'[A-Za-z]+[!.,;:\-?]*', "", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r'(?<=[А-Яа-я,;:\—" ])\n(?=[а-яPS])', r" ", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"(?<=[А-Яа-я])[ ]*\n", r".\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\n(?! \— [а-я])", "", text)
    text = re.sub(r"([!?.,:;]+)(?=[А-Яа-я])", r"\1 ", text)
    text = re.sub(
        r"(?<!\W[А-Я]\.)(?<!\WP\.)(?<!\WP\. S\.)(?<!\W(т|Т)\.)(?<!\W(т|Т)\. (к|К|д|Д)\.)(?<=\.|\?|\!)(?! [а-я])(?! \— [а-я])\s",
        "\n", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r";", ",", text)
    text = re.sub(r"\—(?=.)", "— ", text)
    text = re.sub(r"([!?.,:;]+)(?=[А-Яа-я])", r"\1 ", text)
    text = re.sub(r"(?<=[!?.,:;])\—", " —", text)
    text = re.sub(r"\.{4,}", "", text)
    text = re.sub(r"\s\.\s", "\n", text)
    text = re.sub(r"(?<=[!.,?:;])P.", "\nP.", text)
    text = re.sub(r"(?<=[!.,?:;])\— (?![а-я])", "\n—", text)
    text = re.sub(
        r"(?<!\W[А-Я]\.)(?<!\WP\.)(?<!\WP\. S\.)(?<!\W(т|Т)\.)(?<!\W(т|Т)\. (к|К|д|Д)\.)(?<=\.|\?|\!)(?! [а-я])(?! \— [а-я])\s",
        "\n", text)
    text = re.sub(r"(?<=[!?.,:;])\n\— (?=[а-я])", " — ", text)
    text = re.sub(r"\([\d]*\)", "", text)
    text = re.sub(r"(\.\n){2,}", "\n", text)
    text = re.sub(r"[IXV]+", r"", text)
    text = re.sub(r"\s\.\s", "\n", text)
    text = re.sub(r"\s([!?.,:;])", "\1", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"\n[ ]+", "\n", text)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"(?<=[!?.,:;])\n\— (?=[а-я])", " — ", text)
    return text


morph = pm.MorphAnalyzer()
combinator = Combinator(DEFAULT_GRAMMARS)
symbs = re.compile(r"[^А-Яа-я:!\?,\.\"— -]")
clear = re.compile(r"[ _]{2,}")
punct = re.compile(r"(\.\.\.|!\.\.|\?\.\.|[:!\?,\.\"—])")


def _morph_line(line, normalize=True, tag=True, ner=True):
    if ner:
        for grammar, tokens in combinator.resolve_matches(combinator.extract(line)):
            for token in tokens:
                start, end = token.position
                line = line[:start] + "@" * (end - start) + line[end:]

    line = re.sub(punct, r" <\1>", line)
    words = line.split(" ")
    if normalize or tag:
        parsed = [morph.parse(w)[0] for w in words]
        if normalize:
            words = [wparsed.normal_form for wparsed in parsed]
        if tag:
            tags = [wparsed.tag.cyr_repr for wparsed in parsed]
            line = '_'.join(words) + ';' + '_'.join(tags)
        else:
            line = '_'.join(words)
    else:
        line = '_'.join(words)
    return line


def _prepare_line(line):
    line = re.sub(symbs, "", line)
    line = re.sub(r"[ ]+", " ", line)
    return line


def _clear_line(line):
    line = re.sub("@+", "<@>", line)
    line = re.sub(clear, " ", line)
    line = re.sub("><", ">_<", line)
    line = line.strip("_")
    line = line.strip()
    return line


def _proc_line(line, normalize=True, tag=True, ner=True):
    line = _prepare_line(line)
    line = _morph_line(line, normalize, tag, ner)
    line = _clear_line(line)
    return line


def _worker(line_t, normalize, tag, ner):
    return _proc_line(line_t, normalize, tag, ner)


def preprocess_sentences(fnin, fnout, normalize=False, tag=False, ner=False, workers=8):
    """Умеет нормализовать, сохранять тэги и находить именованные сущности"""

    from multiprocessing import Pool
    p = Pool(workers)

    with open(fnin, encoding="utf-8") as fin:
        lines = fin.readlines()
    from functools import partial
    worker = partial(_worker, normalize=normalize, tag=tag, ner=ner)

    with open(fnout, "w", encoding="utf-8") as fout:
        l = len(lines)
        step = l // 1000
        for i in range(0, l + step, step):
            plines = p.map(worker, lines[i:i + step])
            print(i, "/", l, end="\r")
            for pline in plines:
                print(pline, file=fout, sep='')

    print("Done")
