import sys

from knowledge_extraction.bilstm_crf.app import medical_ner
nlu_path='D:/LLM+KG/KBQA-for-Diagnosis-main/KBQA-for-Diagnosis-main/knowledge_extraction/bilstm_crf'
sys.path.append(nlu_path)
result = medical_ner(["淋球菌性尿道炎怎么办呢"])
print(result)

