from rag import *

query_engine  = index.as_query_engine()
response = query_engine.query("Hello, what is your name and what is your purpose?")
print(response)