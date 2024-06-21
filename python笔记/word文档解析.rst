word文档解析
========================

https://python-docx.readthedocs.io/en/latest/

.. code-block:: python

    import docx
    from docx import Document
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    from xml.etree import ElementTree as ET
    from lxml import etree
    import io
    import zipfile

    doc = docx.Document(io.BytesIO(file_bytes))
    # 读取不同的段落信息
    TextList = []
    for paragraph in doc.paragraphs:
        pd = {}
        pd["text"] = paragraph.text
        # 检查段落是否居中, 1 代表居中对齐
        pd["alignment"] = paragraph.alignment
        # 遍历段落中的每个运行（Run）, 检查字体是否加粗
        pd["bolds"] = []
        for run in paragraph.runs:
            if run.font.bold:
                pd["bolds"].append(run.text.strip())
        TextList.append(pd)

    with zipfile.ZipFile(io.BytesIO(file_bytes), 'r') as docx_zip:

.. code-block:: python 

    import docx
    from docx import Document
    from docx.oxml.text.paragraph import CT_P
    from docx.oxml.table import CT_Tbl
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    from xml.etree import ElementTree as ET
    from lxml import etree
    import io
    import zipfile

    doc = docx.Document(io.BytesIO(file_bytes))
    # 读取不同的段落信息
    TextList = []
    elements = []
    for element in doc.element.body.iter():
        # print(element)
        elements.append(element)
        # 处理段落
        if isinstance(element,CT_P):
            paragraph = Paragraph(element,element.getparent())
            pd = {}
            pd["text"] = paragraph.text
            # 检查段落是否居中, 1 代表居中对齐
            pd["alignment"] = paragraph.alignment
            # 遍历段落中的每个运行（Run）, 检查字体是否加粗
            pd["bolds"] = []
            for run in paragraph.runs:
                if run.font.bold:
                    pd["bolds"].append(run.text.strip())
            TextList.append(pd)

        #  处理表格
        if isinstance(element, CT_Tbl):
            table = Table(element, doc)
            for row in table.rows:
                pd = {}
                row_cells = [cell.text for cell in row.cells]
                # print(row_cells)
                pd["text"] = (" ".join([cell.text.replace('\n', ' ') for cell in row.cells]))
                TextList.append(pd)