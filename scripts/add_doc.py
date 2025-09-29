import sys
import os

# Ensure project root is in sys.path for 'src' imports when running from scripts/
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.rag_pipeline import rag_pipeline

def main():
    if len(sys.argv) > 1:
        doc_path = sys.argv[1]
    else:
        doc_path = os.path.join('./data', 'Merchant Quick loan TRS.docx')

    exists = os.path.exists(doc_path)
    print('[DOC_PATH]', doc_path, exists)
    if not exists:
        print('[ERROR] Document not found at path above.')
        return

    try:
        res = rag_pipeline.add_single_document(doc_path)
        print('[ADD_RESULT]', res)
    except Exception as e:
        print('[ERROR] Failed to add document:', str(e))
        return

    try:
        vc = rag_pipeline.get_vector_count()
        print('[VECTOR_COUNT]', vc)
    except Exception as e:
        print('[ERROR] Failed to get vector count:', str(e))

if __name__ == '__main__':
    main()