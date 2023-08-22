import argparse
import functools
import os
from typing import Literal, Optional, Union

from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

import weaviate
from langchain import prompts
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.smith import RunEvalConfig
from langchain.vectorstores import Weaviate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langsmith import Client
from langsmith import RunEvaluator
from langchain import load as langchain_load
from operator import itemgetter
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.storage import RedisStore, EncoderBackedStore
from langchain.utilities.redis import get_client
from langchain.schema.document import Document

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-2",
}

def create_chain(
    retriever: BaseRetriever,
    model_provider: Union[Literal["openai"], Literal["anthropic"]],
    chat_history: Optional[list] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Runnable:
    model_name = model or _MODEL_MAP[model_provider]
    model = _PROVIDER_MAP[model_provider](model=model_name, temperature=temperature)
    
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""
    
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    _template = """
    You are an expert programmer and problem-solver, tasked to answer any question about Langchain. Using the provided context, answer the user's question to the best of your ability using the resources provided.
    If you really don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
    <context>
        {context} 
    <context/>"""

    if chat_history:
        _inputs = RunnableMap(
                {
                    "standalone_question": {
                        "question": lambda x: x["question"],
                        "chat_history": lambda x: x["chat_history"],
                    } | CONDENSE_QUESTION_PROMPT | model | StrOutputParser(),
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: x["chat_history"],
                }
            )
        _context = {
            "context": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["question"], 
            "chat_history": lambda x: x["chat_history"],
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", _template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])
    else:
        _inputs = RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: [],
            }
        )
        _context = {
            "context": itemgetter("question") | retriever,
            "question": lambda x: x["question"], 
            "chat_history": lambda x: [],
        }
        prompt = ChatPromptTemplate.from_messages([
            ("system", _template),
            ("human", "{question}"),
        ])
    
    chain = (
        _inputs
        | _context
        | prompt 
        | ChatOpenAI(model="gpt-4", temperature=temperature)
        | StrOutputParser()
    )
    
    return chain


def _get_retriever():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

    embeddings = OpenAIEmbeddings(chunk_size=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    print("Index has this many vectors", client.query.aggregate("LangChain_parents_idx").with_meta_count().do())
    weaviate_client = Weaviate(
        client=client,
        index_name="LangChain_parents_idx",
        text_key="text",
        embedding=embeddings,
        by_text=False,
        attributes=["source", "doc_id"],
    )

    def key_encoder(key: int) -> str:
        return json.dumps(key)

    def value_serializer(value: float) -> str:
        if isinstance(value, Document):
            value = {
                'page_content': value.page_content,
                'metadata': value.metadata,
            }
        return json.dumps(value)

    def value_deserializer(serialized_value: str) -> Document:
        value = json.loads(serialized_value)
        if 'page_content' in value and 'metadata' in value:
            return Document(page_content=value['page_content'], metadata=value['metadata'])
        else:
            return value

    client = get_client('redis://default:c16f99d1cc694b7fb9380db03abbe341@fly-chat-langchain.upstash.io')
    abstract_store = RedisStore(client=client)
    store = EncoderBackedStore(
        store=abstract_store,
        key_encoder=key_encoder,
        value_serializer=value_serializer,
        value_deserializer=value_deserializer
    )
    
    retriever = ParentDocumentRetriever(
        vectorstore=weaviate_client, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={'k': 10}
    )
    
    return retriever

class CustomHallucinationEvaluator(RunEvaluator):

    @staticmethod
    def _get_llm_runs(run: Run) -> Run:
        runs = []
        for child in (run.child_runs or []):
            if run.run_type == "llm":
                runs.append(child)
            else:
                runs.extend(CustomHallucinationEvaluator._get_llm_runs(child))


    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult:
        llm_runs = self._get_llm_runs(run)
        if not llm_runs:
            return EvaluationResult(key="hallucination", comment="No LLM runs found")
        if len(llm_runs) > 0:
            return EvaluationResult(key="hallucination", comment="Too many LLM runs found")
        llm_run = llm_runs[0]
        messages = llm_run.inputs["messages"]
        langchain_load(json.dumps(messages))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Questions")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--prompt-type", default="chat")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    retriever = _get_retriever()
    constructor = functools.partial(
        create_chain,
        retriever=retriever,
        model_provider=args.model_provider,
    )
    chain = constructor()
    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=constructor,
        evaluation=eval_config,
        verbose=True,
    )
    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
