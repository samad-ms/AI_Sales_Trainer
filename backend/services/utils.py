def load_reference_text(doc_file):
    reference_text = ""
    if doc_file:
        if doc_file.type == "application/pdf":
            import PyPDF2
            reader = PyPDF2.PdfReader(doc_file)
            reference_text = "\n".join([page.extract_text() for page in reader.pages])
        elif doc_file.type == "text/plain":
            reference_text = doc_file.read().decode()
        elif doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            import docx
            doc = docx.Document(doc_file)
            reference_text = "\n".join([para.text for para in doc.paragraphs])
    return reference_text
