from typing import TypedDict, List
from langchain_core.documents import Document
# States class comprises of 3 attributes :
# 1. question (Input)
# 2. context (Input)
# 3. answer (Output)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str