import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Literal,
    Union,
)
from langchain_core.callbacks import CallbackManagerForLLMRun
from fastapi import Request as fastapi_request
from langchain.llms.base import LLM
import json
import os
import backoff
from fastapi import HTTPException
import httpx
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel, Field
import requests
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.utils import get_from_dict_or_env
import uuid
from app.utils.custom_loguru import logger
from app.utils.custom_httpx import CustomHttpX
from app.utils.config_cache import cache_get, cache_set
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.documents.base import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import StuffDocumentsChain, LLMChain
from opensearchpy.exceptions import NotFoundError
from langchain.chains.conversational_retrieval.base import (
    BaseConversationalRetrievalChain,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
    Callbacks,
)
from typing import List, Optional, Mapping, Any
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from vsl_utils.v2 import request_response_metering
from vsl_utils.v2.constants import uom, vendors
import openai
from langchain_core.language_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from app.utils.misc import is_output_token_limited_model, calculate_max_tokens

httpx_client = CustomHttpX(raise_for_status=False)

CLIENT_ID = os.environ.get("CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", "")
PINGFEDERATE_URL = os.environ.get("PINGFEDERATE_URL", "")
IAS_OPENAI_CHAT_URL = os.environ.get("IAS_OPENAI_CHAT_URL", "")
IAS_OPENAI_URL = os.environ.get("IAS_OPENAI_URL", "")
IAS_EMBEDDINGS_URL = os.environ.get("IAS_EMBEDDINGS_URL", "")
IAS_BEDROCK_URL = os.getenv("IAS_BEDROCK_URL")
IAS_BEDROCK_EMBEDDING_URL = os.getenv("IAS_BEDROCK_EMBEDDING_URL")


# To Raise Generic OPENAI EMBEDDING ERROR
class GenericException(Exception):
    def __init__(self, message, status_code: int = None):
        super().__init__(message)
        self.status_code = (
            status_code
            if status_code
            else message.status_code if isinstance(message, GenericException) else ""
        )


def is_http_4xx_error(exception):
    return isinstance(exception, GenericException) and exception.status_code in [
        400,
        401,
        403,
        404,
        405,
        422,
    ]


async def afederate_auth() -> str:
    """Obtains auth access token for accessing GPT endpoints"""
    try:
        # Check in cache.
        token = await cache_get(f"{CLIENT_ID}-token")
        if token is None:
            payload = f"client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}"
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
            }
            response = await httpx_client.request(
                "post", PINGFEDERATE_URL, data=payload, headers=headers
            )
            if response.status_code != 200:
                logger.error(
                    f"Error calling OpenAI access token API: {response.status_code}, {response.json()}"
                )
                raise Exception(
                    f"Error calling OpenAI access token API: {response.status_code}, {response.json()}"
                )

            token = response.json()["access_token"]

            # Cache the token.
            await cache_set(f"{CLIENT_ID}-token", token)
        return token
    except httpx.TimeoutException as e:
        logger.error(f"Federate Auth Timeout {e}")
        raise HTTPException(status_code=408, detail=f"Error: Federate Auth Timeout")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise e


# This is a temporary function to support synchronous.
# TODO: The proper solution is to use asynchronous
# asynchronous in the OpenSearch vector store implementations which is not available currently.
def federate_auth() -> str:
    """Obtains auth access token for accessing GPT endpoints"""
    try:
        payload = f"client_id={CLIENT_ID}&client_secret={CLIENT_SECRET}"
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = requests.post(PINGFEDERATE_URL, headers=headers, data=payload)
        if response.status_code != 200:
            logger.error(
                f"Error calling OpenAI access token API: {response.status_code}, {response.json()}"
            )
            raise Exception(
                f"Error calling OpenAI access token API: {response.status_code}, {response.json()}"
            )

        token = response.json()["access_token"]
        return token
    except httpx.TimeoutException as e:
        logger.error(f"Federate Auth Timeout {e}")
        raise HTTPException(status_code=408, detail=f"Error: Federate Auth Timeout")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise e


async def aget_auth_token(bearer_auth):
    # If the x-vsl-client_id is not provided in headers, then get the token from federate_auth
    if bearer_auth is None:
        token = await afederate_auth()
    else:
        token = (
            bearer_auth.split(" ")[1]
            if len(bearer_auth.split(" ")) > 1 and len(bearer_auth.split(" ")[1]) > 1
            else await afederate_auth()
        )
    return token


# This is a temporary function to support synchronous.
# TODO: The proper solution is to use asynchronous
# asynchronous in the OpenSearch vector store implementations which is not available currently.
def get_auth_token(bearer_auth):
    # If the x-vsl-client_id is not provided in headers, then get the token from federate_auth
    if bearer_auth is None:
        token = federate_auth()
    else:
        token = (
            bearer_auth.split(" ")[1]
            if len(bearer_auth.split(" ")) > 1 and len(bearer_auth.split(" ")[1]) > 1
            else federate_auth()
        )
    return token


@backoff.on_exception(
    backoff.expo, GenericException, max_tries=20, max_time=60, giveup=is_http_4xx_error
)
def ias_openai_chat_completion(
    user_message: str,
    engine: str,
    temperature: float,
    max_tokens: int,
    system_message: str = None,
    client_id: str = None,
    x_vsl_client_id: str = None,
    r_parent_wf_req_id: str = None,
    bearer_token: str = None,
) -> str:
    """
    Generates a chat completion response for OpenAI model
    :param token: auth token
    :param user_message: user's prompt
    :param engine: model capable for chat completion i.e. gpt*
    :param temperature: value 0-1 that tells model to be more precise or generative
    :param max_tokens: max tokens the prompt & response should be. It depends on the model's capacity
    :return: response from OpenAI model
    """
    try:
        payload = {
            "engine": engine,
            "messages": [
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_message:
            payload["messages"].insert(0, {"role": "system", "content": system_message})

        token = get_auth_token(bearer_token)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if x_vsl_client_id is not None:
            headers["x-vsl-client_id"] = x_vsl_client_id
        elif client_id is not None:
            headers["x-vsl-client_id"] = client_id
        
        if r_parent_wf_req_id is not None:
            headers['x-vsl-parent-req-id'] = r_parent_wf_req_id

        logger.info("Calling chat completion endpoint")
        response = requests.post(IAS_OPENAI_CHAT_URL, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(
                f"Error calling OpenAI chat completion  API: {response.status_code}, {response.json()}"
            )
            raise GenericException(
                f"Error calling OpenAI chat completion API: {response.status_code}, {response.json()}",
                status_code=response.status_code,
            )

        logger.info("Received response from llm")
        chat_completion = json.loads(response.json()["result"])["content"]
        total_token_completion = int(response.json()["totalTokens"])

        return total_token_completion, chat_completion
    except Exception as e:
        logger.error("Got the Exception", str(e))
        # raising backoff exception
        raise GenericException(e)


@backoff.on_exception(
    backoff.expo, GenericException, max_tries=20, max_time=60, giveup=is_http_4xx_error
)
async def ias_openai_chat_completion_with_tools(
    engine: str,
    temperature: float,
    max_tokens: int,
    client_id: str = None,
    x_vsl_client_id: str = None,
    r_parent_wf_req_id: str =None,
    bearer_token: str = None,
    messages: List[BaseMessage] = None,
    tools: List[BaseTool] = None,
    tool_choice: str = None,
) -> str:
    """
    Generates a chat completion response for OpenAI model
    :param token: auth token
    :param user_message: user's prompt
    :param engine: model capable for chat completion i.e. gpt*
    :param temperature: value 0-1 that tells model to be more precise or generative
    :param max_tokens: max tokens the prompt & response should be. It depends on the model's capacity
    :return: response from OpenAI model
    """
    try:
        payload = {
            "engine": engine,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
        }

        token = await aget_auth_token(bearer_token)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }
        if x_vsl_client_id is not None:
            headers["x-vsl-client_id"] = x_vsl_client_id
        elif client_id is not None:
            headers["x-vsl-client_id"] = client_id

        if r_parent_wf_req_id is not None:
            headers["x-vsl-parent-req-id"] = r_parent_wf_req_id

        logger.info("Calling chat completion endpoint with tools")
        logger.info(payload)

        response = await httpx_client.request(
            "post", IAS_OPENAI_CHAT_URL, json=payload, headers=headers
        )

        logger.info("Received response from llm")
        logger.info(response.json())

        if response.status_code != 200:
            logger.error(
                f"Error calling OpenAI chat completion API: {response.status_code}, {response.json()}"
            )
            raise GenericException(
                f"Error calling OpenAI chat completion API: {response.status_code}, {response.json()}",
                status_code=response.status_code,
            )
        chat_completion = json.loads(response.json()["result"])

        total_token_completion = int(response.json()["totalTokens"])
        return chat_completion, total_token_completion
    except Exception as e:
        logger.error("Got the Exception", str(e))
        # raising backoff exception
        raise GenericException(e)


@backoff.on_exception(
    backoff.expo, GenericException, max_tries=20, max_time=60, giveup=is_http_4xx_error
)
def ias_openai_embeddings(
    raw_text,
    engine: str,
    client_id: str = None,
    x_vsl_client_id: str = None,
    r_parent_wf_req_id: str = None,
    bearer_token: str = None,
):
    try:
        url = IAS_EMBEDDINGS_URL
        payload = {"input": raw_text, "engine": engine}
        token = get_auth_token(bearer_token)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        if x_vsl_client_id is not None:
            headers["x-vsl-client_id"] = x_vsl_client_id
        elif client_id is not None:
            headers["x-vsl-client_id"] = client_id

        if r_parent_wf_req_id is not None:
            headers["x-vsl-parent-req-id"] = r_parent_wf_req_id
        
        logger.info("Triggering embedding endpoint")
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(
                f"Error calling OpenAI embedding API: {response.status_code}, {response.json()}"
            )
            raise GenericException(
                f"Error calling OpenAI embedding API: {response.status_code}, {response.json()}",
                status_code=response.status_code,
            )

        embeddings = json.loads(response.json()["result"])
        temp = response.json()
        total_token = temp["totalTokens"]
        logger.info("Recevied response from embedding endpoint")

        return embeddings, total_token
    except Exception as e:
        logger.error("Got the Exception", str(e))
        # raising backoff exception
        raise GenericException(e)


class IASOpenaiConversationalLLM(LLM, BaseModel):
    """Wrapper for IAS secured OpenAI chat API"""

    engine: str
    temperature: float
    max_tokens: int
    total_consumed_token: List[int] = Field(default_factory=list)
    system_message: str = None
    client_id: str = None
    x_vsl_client_id: str = None
    r_parent_wf_req_id: str = None
    bearer_auth: str = None

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "IAS_OpenAI"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        prompt_message = prompt

        if self.system_message:
            prompt_message = prompt_message + self.system_message

        token_consumed = self.get_num_tokens(prompt_message)

        total_token_completion, response = ias_openai_chat_completion(
            prompt,
            self.engine,
            self.temperature,
            calculate_max_tokens(self.max_tokens, str(self.engine), token_consumed),
            self.system_message,
            self.client_id,
            self.x_vsl_client_id,
            self.r_parent_wf_req_id
        )

        logger.debug(f"Total tokens consumed: {total_token_completion}")

        self.total_consumed_token.append(total_token_completion)

        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        params = {
            "engine": self.engine,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "total_consumed_token": self.total_consumed_token,
        }
        return params


class IASOpenaiEmbeddings(Embeddings):
    """Wrapper for IAS secured OpenAI embedding API"""

    engine = str
    request = fastapi_request
    client_id: str = None
    x_vsl_client_id: str = None
    bearer_auth: str = None
    r_parent_wf_req_id: str = None
    total_token_embedding: list

    def __init__(
        self,
        engine,
        client_id,
        total_token_embedding,
        request,
        x_vsl_client_id=None,
        r_parent_wf_req_id=None,
        bearer_auth=None,
    ):
        self.engine = engine
        self.total_token_embedding = total_token_embedding
        self.client_id = client_id
        self.x_vsl_client_id = x_vsl_client_id
        self.bearer_auth = bearer_auth
        self.r_parent_wf_req_id = r_parent_wf_req_id
        self.request = request

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeddings search docs."""
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.

        try:
            response, total_token_embedding_temp = ias_openai_embeddings(
                texts,
                self.engine,
                self.client_id,
                self.x_vsl_client_id,
                self.r_parent_wf_req_id,
                self.bearer_auth,
            )
            self.total_token_embedding.append(total_token_embedding_temp)

            # Extract the embeddings
            embeddings: list[list[float]] = [data["embedding"] for data in response]
            return embeddings
        except Exception as e:
            self.handle_error(e)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embeds query text."""
        try:
            response, total_token_embedding_temp = ias_openai_embeddings(
                text,
                self.engine,
                self.client_id,
                self.x_vsl_client_id,
                self.r_parent_wf_req_id,
                self.bearer_auth,
            )
            self.total_token_embedding.append(total_token_embedding_temp)
            # Extract the embeddings
            embeddings: list[list[float]] = [data["embedding"] for data in response]
            return embeddings[0]
        except Exception as e:
            self.handle_error(e)
            raise

    async def handle_error(self, error: Exception):
        """Handles errors by calling push_metric with the sum of totalToken."""
        total_tokens_sum = sum(self.total_token_embedding)
        if total_tokens_sum > 0:
            request = {
                        "headers": self.request.headers
                    }
            
            await request_response_metering(
                request,
                r_parent_wf_req_id=self.r_parent_wf_req_id,
                r_vendor_id=vendors.AZURE
                if self.engine.startswith(("gpt", "text"))
                else vendors.AWS,
                r_param_1=self.engine,
                r_uom_id=uom.TOTAL_TOKENS,
                r_uom_val=total_tokens_sum,
            )


def ias_bedrock_embeddings(
    text,
    model_id: str,
    client_id: str = None,
    x_vsl_client_id: str = None,
    bearer_token: str = None,
    r_parent_wf_req_id: str = None,
):
    try:
        url = IAS_BEDROCK_EMBEDDING_URL
        payload = {"prompt": text, "model": model_id}
        token = get_auth_token(bearer_token)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        }

        if x_vsl_client_id is not None:
            headers["x-vsl-client_id"] = x_vsl_client_id
        elif client_id is not None:
            headers["x-vsl-client_id"] = client_id
        
        if r_parent_wf_req_id is not None:
            headers["x-vsl-parent-req-id"] = r_parent_wf_req_id

        logger.info("Triggering embedding endpoint")
        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(
                f"Error calling Bedrock embedding API: {response.status_code}, {response.json()}"
            )
            raise GenericException(
                f"Error calling Bedrock embedding API: {response.status_code}, {response.json()}",
                status_code=response.status_code,
            )

        embeddings = response.json()["embedding"]
        total_token = response.json()["total_tokens"]
        logger.info("Recevied response from embedding endpoint")
        return embeddings, total_token

    except Exception as e:
        logger.error("Got the Exception", str(e))
        # raising backoff exception
        raise GenericException(e)


class IASBedrockEmbeddings(Embeddings):
    engine = str
    client_id: str = None
    x_vsl_client_id: str = None
    bearer_auth: str = None
    r_parent_wf_req_id: str = None
    total_token_embedding: list
    region_name: Optional[str] = None
    credentials_profile_name: Optional[str] = None
    model_kwargs: Optional[Dict] = None
    endpoint_url: Optional[str] = None
    normalize: bool = False

    def __init__(
        self,
        engine,
        client_id,
        total_token_embedding,
        x_vsl_client_id=None,
        r_parent_wf_req_id=None,
        bearer_auth=None,
    ):
        self.engine = engine
        self.client_id = client_id
        self.x_vsl_client_id = x_vsl_client_id
        self.r_parent_wf_req_id = r_parent_wf_req_id
        self.bearer_auth = bearer_auth
        self.total_token_embedding = total_token_embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeddings search docs."""
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        try:
            results = []
            for text in texts:
                response, total_token_embedding_temp = ias_bedrock_embeddings(
                    text,
                    self.engine,
                    self.client_id,
                    self.x_vsl_client_id,
                    self.bearer_auth,
                    self.r_parent_wf_req_id,
                )
                self.total_token_embedding.append(total_token_embedding_temp)

                results.append(response)

            return results

        except Exception as e:
            self.handle_error(e)
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embeddings  query text."""
        try:
            embedding, total_token_embedding_temp = ias_bedrock_embeddings(
                text,
                self.engine,
                self.client_id,
                self.x_vsl_client_id,
                self.bearer_auth,
                self.r_parent_wf_req_id,
            )
            self.total_token_embedding.append(total_token_embedding_temp)
            return embedding

        except Exception as e:
            self.handle_error(e)
            raise

    async def handle_error(self, error: Exception):
        """Handles errors by calling push_metric with the sum of totalToken."""
        total_tokens_sum = sum(self.total_token_embedding)
        if total_tokens_sum > 0:
            request = {"headers": self.request.headers}
            await request_response_metering(
                request,
                r_parent_wf_req_id=self.r_parent_wf_req_id,
                r_vendor_id=vendors.AZURE
                if self.engine.startswith(("gpt", "text"))
                else vendors.AWS,
                r_param_1=self.engine,
                r_uom_id=uom.TOTAL_TOKENS,
                r_uom_val=total_tokens_sum,
            )


@backoff.on_exception(
    backoff.expo, GenericException, max_tries=20, max_time=60, giveup=is_http_4xx_error
)
def ias_bedrock_completion(
    bearer_token: str,
    user_message: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_message: str = None,
    client_id: str = None,
    x_vsl_client_id: str = None,
    r_parent_wf_req_id: str = None
    
) -> str:
    """
    Generates a completion response for Bedrock model
    :param token: auth token
    :param user_message: user's prompt
    :param model: model capable for completion
    :param temperature: value 0-1 that tells model to be more precise or generative
    :param max_tokens: max tokens the prompt & response should be. It depends on the model's capacity
    :param system_message: system's prompt for instructions to the LLM
    :return: response from Bedrock model
    """
    try:
        if model.startswith("anthropic.claude-3"):
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_message}],
                    }
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if system_message:
                system_msg = {
                    "role": "system",
                    "content": system_message,
                }
                payload["messages"].insert(0, system_msg)
        else:
            if model.startswith("anthropic"):
                user_message = f"\n\nHuman:{user_message}\n\nAssistant:"
            elif model.startswith("amazon"):
                user_message = f"\n\nUser:{user_message}\n\nBot:"

            payload = {
                "model": model,
                "prompt": user_message,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if system_message:
                payload["prompt"] = " ".join([system_message, user_message])

        token = get_auth_token(bearer_token)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        }

        if x_vsl_client_id is not None:
            headers["x-vsl-client_id"] = x_vsl_client_id
        elif client_id is not None:
            headers["x-vsl-client_id"] = client_id

        if r_parent_wf_req_id is not None:
            headers["x-vsl-parent-req-id"] = r_parent_wf_req_id

        logger.info("Calling IAS Bedrock completion API")
        response = requests.post(IAS_BEDROCK_URL, headers=headers, json=payload)

        if response.status_code != 200:
            logger.error(
                f"Error calling IAS Bedrock completion API: {response.status_code}, {response.json()}"
            )
            raise GenericException(
                f"Error calling IAS Bedrock completion API: {response.status_code}, {response.json()}",
                status_code=response.status_code,
            )

        logger.info("Completion created from IAS Bedrock completion API")
        completion_resp = response.json()
        completion = completion_resp["result"]
        total_token_completion = int(completion_resp["total_tokens"])

        return total_token_completion, completion

    except Exception as e:
        logger.error(f"Exception calling IAS Bedrock Completion API: {str(e)}")
        raise GenericException(e)


class IASBedrockLLM(LLM):
    """Wrapper for IAS secured Bedrock completion API"""

    model: str
    temperature: float
    max_tokens: int
    total_consumed_token: List[int] = Field(default_factory=list)
    system_message: str = None
    client_id: str = None
    x_vsl_client_id: str = None
    bearer_auth: str = None
    r_parent_wf_req_id: str = None

    @property
    def _llm_type(self) -> str:
        return "IAS_Bedrock"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given prompt and input."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
    ) -> str:
        # token = federate_auth()
        prompt_message = prompt

        if self.system_message:
            prompt_message = prompt_message + self.system_message

        token_consumed = self.get_num_tokens(prompt_message)
        total_token_completion, response = ias_bedrock_completion(
            self.bearer_auth,
            prompt,
            self.model,
            self.temperature,
            calculate_max_tokens(self.max_tokens, str(self.model), token_consumed),
            self.system_message,
            self.client_id,
            self.x_vsl_client_id,
            self.r_parent_wf_req_id
        )
        logger.debug(f"Total tokens consumed: {total_token_completion}")

        self.total_consumed_token.append(total_token_completion)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "total_consumed_token": self.total_consumed_token,
        }
        return params


class IAS_OpenSearchVectorSearch(OpenSearchVectorSearch):
    def _select_relevance_score_fn(self):
        return lambda score: score

    # Overwriting the default implementation by adding new vector_field_3073 property to support multiple embedding models
    def _default_text_mapping(
        self,
        dim: int,
        engine: str = "nmslib",
        space_type: str = "l2",
        ef_search: int = 512,
        ef_construction: int = 512,
        m: int = 16,
        vector_field: str = "vector_field",
        vector_field_3072: str = "vector_field_3072",
    ) -> Dict:
        """For Approximate k-NN Search, this is the default mapping to create index."""
        return {
            "settings": {"index": {"knn": True, "knn.algo_param.ef_search": ef_search}},
            "mappings": {
                "properties": {
                    vector_field: {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": {"ef_construction": ef_construction, "m": m},
                        },
                    },
                    vector_field_3072: {
                        "type": "knn_vector",
                        "dimension": 3072,
                        "method": {
                            "name": "hnsw",
                            "space_type": space_type,
                            "engine": engine,
                            "parameters": {"ef_construction": ef_construction, "m": m},
                        },
                    },
                }
            },
        }

    # Overwriting the default implementation to support multiple embedding models
    def _default_scripting_text_mapping(
        self,
        dim: int,
        vector_field: str = "vector_field",
    ) -> Dict:
        """For Painless Scripting or Script Scoring,the default mapping to create index."""
        return {
            "mappings": {
                "properties": {
                    vector_field: {"type": "knn_vector", "dimension": dim},
                }
            }
        }

    # Overwriting the default implementation to support multiple embedding models
    def _import_bulk(self) -> Any:
        """Import bulk if available, otherwise raise error."""
        try:
            from opensearchpy.helpers import bulk
        except ImportError:
            raise ImportError(
                "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
            )
        return bulk

    # Overwriting the default implementation to support multiple embedding models
    def _import_not_found_error(self) -> Any:
        """Import not found error if available, otherwise raise error."""
        try:
            from opensearchpy.exceptions import NotFoundError
        except ImportError:
            raise ImportError(
                "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
            )
        return NotFoundError

    # Overwriting the default implementation to support multiple embedding models
    def _is_aoss_enabled(http_auth: Any) -> bool:
        """Check if the service is http_auth is set as `aoss`."""
        if (
            http_auth is not None
            and hasattr(http_auth, "service")
            and http_auth.service == "aoss"
        ):
            return True
        return False

    # Overwriting the default implementation to support multiple embedding models
    def _import_opensearch() -> Any:
        """Import OpenSearch if available, otherwise raise error."""
        try:
            from opensearchpy import OpenSearch
        except ImportError:
            raise ImportError(
                "Could not import OpenSearch. Please install it with `pip install opensearch-py`."
            )
        return OpenSearch

    # Overwriting the default implementation to support multiple embedding models
    def _get_opensearch_client(self, opensearch_url: str, **kwargs: Any) -> Any:
        """Get OpenSearch client from the opensearch_url, otherwise raise error."""
        try:
            opensearch = self._import_opensearch()
            client = opensearch(opensearch_url, **kwargs)
        except ValueError as e:
            raise ImportError(
                f"OpenSearch client string provided is not in proper format. "
                f"Got error: {e} "
            )
        return client

    # Overwriting the default implementation to support multiple embedding models
    def _validate_embeddings_and_bulk_size(
        self, embeddings_length: int, bulk_size: int
    ) -> None:
        """Validate Embeddings Length and Bulk Size."""
        if embeddings_length == 0:
            raise RuntimeError("Embeddings size is zero")
        if bulk_size < embeddings_length:
            raise RuntimeError(
                f"The embeddings count, {embeddings_length} is more than the "
                f"[bulk_size], {bulk_size}. Increase the value of [bulk_size]."
            )

    # Overwriting the default implementation to support multiple embedding models
    def _validate_aoss_with_engines(self, is_aoss: bool, engine: str) -> None:
        """Validate AOSS with the engine."""
        if is_aoss and engine != "nmslib" and engine != "faiss":
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `nmslib` or `faiss` engines"
            )

    # Overwriting the default implementation to support multiple embedding models
    def _bulk_ingest_embeddings(
        self,
        client: Any,
        index_name: str,
        embeddings: List[List[float]],
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        vector_field: str = "vector_field",
        text_field: str = "text",
        mapping: Optional[Dict] = None,
        max_chunk_bytes: Optional[int] = 1 * 1024 * 1024,
        is_aoss: bool = False,
    ) -> List[str]:
        """Bulk Ingest Embeddings into given index."""
        if not mapping:
            mapping = dict()

        bulk = self._import_bulk()
        not_found_error = self._import_not_found_error()
        requests = []
        return_ids = []
        mapping = mapping

        try:
            client.indices.get(index=index_name)
        except not_found_error:
            client.indices.create(index=index_name, body=mapping)

        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            _id = ids[i] if ids else str(uuid.uuid4())
            request = {
                "_op_type": "index",
                "_index": index_name,
                vector_field: embeddings[i],
                text_field: text,
                "metadata": metadata,
            }
            if is_aoss:
                request["id"] = _id
            else:
                request["_id"] = _id
            requests.append(request)
            return_ids.append(_id)
        bulk(client, requests, max_chunk_bytes=max_chunk_bytes)
        if not is_aoss:
            client.indices.refresh(index=index_name)
        return return_ids

    # Overwriting the default implementation to support multiple embedding models
    def __add(
        self,
        texts: Iterable[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        self._validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        dim = len(embeddings[0])
        engine = kwargs.get("engine", "nmslib")
        space_type = kwargs.get("space_type", "l2")
        ef_search = kwargs.get("ef_search", 512)
        ef_construction = kwargs.get("ef_construction", 512)
        m = kwargs.get("m", 16)
        vector_field = kwargs.get("vector_field", "vector_field")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)

        self._validate_aoss_with_engines(self.is_aoss, engine)

        mapping = self._default_text_mapping(
            dim, engine, space_type, ef_search, ef_construction, m, vector_field
        )

        return self._bulk_ingest_embeddings(
            self.client,
            index_name,
            embeddings,
            texts,
            metadatas=metadatas,
            ids=ids,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=self.is_aoss,
        )

    @classmethod
    # Overwriting the default implementation to support multiple embedding models
    def from_embeddings(
        cls,
        embeddings: List[List[float]],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        bulk_size: int = 500,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> OpenSearchVectorSearch:
        """Construct OpenSearchVectorSearch wrapper from pre-vectorized embeddings.

        Example:
            .. code-block:: python

                from langchain_community.vectorstores import OpenSearchVectorSearch
                from langchain_community.embeddings import OpenAIEmbeddings
                embedder = OpenAIEmbeddings()
                embeddings = embedder.embed_documents(["foo", "bar"])
                opensearch_vector_search = OpenSearchVectorSearch.from_embeddings(
                    embeddings,
                    texts,
                    embedder,
                    opensearch_url="http://localhost:9200"
                )

        OpenSearch by default supports Approximate Search powered by nmslib, faiss
        and lucene engines recommended for large datasets. Also supports brute force
        search through Script Scoring and Painless Scripting.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".

        Optional Keyword Args for Approximate Search:
            engine: "nmslib", "faiss", "lucene"; default: "nmslib"

            space_type: "l2", "l1", "cosinesimil", "linf", "innerproduct"; default: "l2"

            ef_search: Size of the dynamic list used during k-NN searches. Higher values
            lead to more accurate but slower searches; default: 512

            ef_construction: Size of the dynamic list used during k-NN graph creation.
            Higher values lead to more accurate graph but slower indexing speed;
            default: 512

            m: Number of bidirectional links created for each new element. Large impact
            on memory consumption. Between 2 and 100; default: 16

        Keyword Args for Script Scoring or Painless Scripting:
            is_appx_search: False

        """
        opensearch_url = get_from_dict_or_env(
            kwargs, "opensearch_url", "OPENSEARCH_URL"
        )
        # List of arguments that needs to be removed from kwargs
        # before passing kwargs to get opensearch client
        keys_list = [
            "opensearch_url",
            "index_name",
            "is_appx_search",
            "vector_field",
            "text_field",
            "engine",
            "space_type",
            "ef_search",
            "ef_construction",
            "m",
            "max_chunk_bytes",
            "is_aoss",
        ]
        cls._validate_embeddings_and_bulk_size(len(embeddings), bulk_size)
        dim = len(embeddings[0])
        # Get the index name from either from kwargs or ENV Variable
        # before falling back to random generation
        index_name = get_from_dict_or_env(
            kwargs, "index_name", "OPENSEARCH_INDEX_NAME", default=uuid.uuid4().hex
        )
        is_appx_search = kwargs.get("is_appx_search", True)
        vector_field = kwargs.get("vector_field", "vector_field")
        text_field = kwargs.get("text_field", "text")
        max_chunk_bytes = kwargs.get("max_chunk_bytes", 1 * 1024 * 1024)
        http_auth = kwargs.get("http_auth")
        is_aoss = cls._is_aoss_enabled(http_auth=http_auth)
        engine = None

        if is_aoss and not is_appx_search:
            raise ValueError(
                "Amazon OpenSearch Service Serverless only "
                "supports `approximate_search`"
            )

        if is_appx_search:
            engine = kwargs.get("engine", "nmslib")
            space_type = kwargs.get("space_type", "l2")
            ef_search = kwargs.get("ef_search", 512)
            ef_construction = kwargs.get("ef_construction", 512)
            m = kwargs.get("m", 16)

            cls._validate_aoss_with_engines(is_aoss, engine)

            mapping = cls._default_text_mapping(
                dim, engine, space_type, ef_search, ef_construction, m, vector_field
            )
        else:
            mapping = cls._default_scripting_text_mapping(dim)

        [kwargs.pop(key, None) for key in keys_list]
        client = cls._get_opensearch_client(opensearch_url, **kwargs)
        cls._bulk_ingest_embeddings(
            client,
            index_name,
            embeddings,
            texts,
            ids=ids,
            metadatas=metadatas,
            vector_field=vector_field,
            text_field=text_field,
            mapping=mapping,
            max_chunk_bytes=max_chunk_bytes,
            is_aoss=is_aoss,
        )
        kwargs["engine"] = engine
        return cls(opensearch_url, index_name, embedding, **kwargs)

    # Overwriting the default implementation to support multiple embedding models
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        bulk_size: int = 500,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids: Optional list of ids to associate with the texts.
            bulk_size: Bulk API request count; Default: 500

        Returns:
            List of ids from adding the texts into the vectorstore.

        Optional Args:
            vector_field: Document field embeddings are stored in. Defaults to
            "vector_field".

            text_field: Document field the text of the document is stored in. Defaults
            to "text".
        """
        embeddings = self.embedding_function.embed_documents(list(texts))
        return self.__add(
            texts,
            embeddings,
            metadatas=metadatas,
            ids=ids,
            bulk_size=bulk_size,
            **kwargs,
        )


class LLMNotCalledException(Exception):
    """Custom exception to indicate that LLM is not called."""

    pass


class IAS_ConversationalRetrievalChain(BaseConversationalRetrievalChain):
    """Chain for having a conversation based on retrieved documents.

    This chain takes in chat history (a list of messages) and new questions,
    and then returns an answer to that question.
    The algorithm for this chain consists of three parts:

    1. Use the chat history and the new question to create a "standalone question".
    This is done so that this question can be passed into the retrieval step to fetch
    relevant documents. If only the new question was passed in, then relevant context
    may be lacking. If the whole conversation was passed into retrieval, there may
    be unnecessary information there that would distract from retrieval.

    2. This new question is passed to the retriever and relevant documents are
    returned.

    3. The retrieved documents are passed to an LLM along with either the new question
    (default behavior) or the original question and chat history to generate a final
    response.

    Example:
        .. code-block:: python

            from langchain.chains import (
                StuffDocumentsChain, LLMChain, ConversationalRetrievalChain
            )
            from langchain_core.prompts import PromptTemplate
            from langchain_community.llms import OpenAI

            combine_docs_chain = StuffDocumentsChain(...)
            vectorstore = ...
            retriever = vectorstore.as_retriever()

            # This controls how the standalone question is generated.
            # Should take `chat_history` and `question` as input variables.
            template = (
                "Combine the chat history and follow up question into "
                "a standalone question. Chat History: {chat_history}"
                "Follow up question: {question}"
            )
            prompt = PromptTemplate.from_template(template)
            llm = OpenAI()
            question_generator_chain = LLMChain(llm=llm, prompt=prompt)
            chain = ConversationalRetrievalChain(
                combine_docs_chain=combine_docs_chain,
                retriever=retriever,
                question_generator=question_generator_chain,
            )
    """

    retriever: BaseRetriever
    """Retriever to use to fetch documents."""
    max_tokens_limit: Optional[int] = None
    """If set, enforces that the documents returned are less than this limit.
    This is only enforced if `combine_docs_chain` is of type StuffDocumentsChain."""
    llm_response_flag: Optional[bool] = True
    """Specify whether to call LLM or not."""
    answer_from_llm_if_no_docs_found: Optional[bool] = False
    """Answer from LLM knowledge if no matching docs found."""
    max_input_tokens_limit: Optional[int] = None

    def _reduce_tokens_below_limit(self, docs: List[Document]) -> List[Document]:
        num_docs = len(docs)
        # For output-token-limited models(ex:gpt4o), we have different limit for input tokens
        if not self.max_input_tokens_limit:
            self.max_input_tokens_limit = self.max_tokens_limit

        if self.max_input_tokens_limit and isinstance(
            self.combine_docs_chain, StuffDocumentsChain
        ):
            tokens = [
                self.combine_docs_chain.llm_chain._get_num_tokens(doc.page_content)
                for doc in docs
            ]
            token_count = sum(tokens[:num_docs])
            while token_count > self.max_input_tokens_limit:
                num_docs -= 1
                token_count -= tokens[num_docs]

        return docs[:num_docs]

    def _get_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: CallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = self.retriever.get_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        if self.llm_response_flag is True and len(docs) > 0:
            return self._reduce_tokens_below_limit(docs)
        elif self.answer_from_llm_if_no_docs_found:
            return self._reduce_tokens_below_limit(docs)
        else:
            raise LLMNotCalledException()

    async def _aget_docs(
        self,
        question: str,
        inputs: Dict[str, Any],
        *,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> List[Document]:
        """Get docs."""
        docs = await self.retriever.aget_relevant_documents(
            question, callbacks=run_manager.get_child()
        )

        if self.llm_response_flag is True and len(docs) > 0:
            return self._reduce_tokens_below_limit(docs)
        elif self.answer_from_llm_if_no_docs_found:
            return self._reduce_tokens_below_limit(docs)
        else:
            raise LLMNotCalledException()

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> BaseConversationalRetrievalChain:
        """Convenience method to load chain from LLM and retriever.

        This provides some logic to create the `question_generator` chain
        as well as the combine_docs_chain.

        Args:
            llm: The default language model to use at every part of this chain
                (eg in both the question generation and the answering)
            retriever: The retriever to use to fetch relevant documents from.
            condense_question_prompt: The prompt to use to condense the chat history
                and new question into a standalone question.
            chain_type: The chain type to use to create the combine_docs_chain, will
                be sent to `load_qa_chain`.
            verbose: Verbosity flag for logging to stdout.
            condense_question_llm: The language model to use for condensing the chat
                history and new question into a standalone question. If none is
                provided, will default to `llm`.
            combine_docs_chain_kwargs: Parameters to pass as kwargs to `load_qa_chain`
                when constructing the combine_docs_chain.
            callbacks: Callbacks to pass to all subchains.
            **kwargs: Additional parameters to pass when initializing
                ConversationalRetrievalChain
        """

        prompt_template = """Use the following pieces of context to answer the question at the end. {answer_with_llm_knowledge}

        {context}

        Question: {question}
        Helpful Answer:"""

        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question", "answer_with_llm_knowledge"],
        )

        combine_docs_chain_kwargs = combine_docs_chain_kwargs or {"prompt": QA_PROMPT}
        doc_chain = load_qa_chain(
            llm,
            chain_type=chain_type,
            verbose=verbose,
            callbacks=callbacks,
            **combine_docs_chain_kwargs,
        )

        _llm = condense_question_llm or llm
        condense_question_chain = LLMChain(
            llm=_llm,
            prompt=condense_question_prompt,
            verbose=verbose,
            callbacks=callbacks,
        )
        return cls(
            retriever=retriever,
            combine_docs_chain=doc_chain,
            question_generator=condense_question_chain,
            callbacks=callbacks,
            **kwargs,
        )


class IAS_ChatModel(BaseChatModel, BaseModel):
    engine: str
    temperature: float
    max_tokens: int
    user_query: str
    total_consumed_token: List[int] = Field(default_factory=list)
    min_response_token: int
    system_message: Optional[str] = (None,)
    client_id: str = (None,)
    x_vsl_client_id: str = None
    bearer_token: str = None
    r_parent_wf_req_id: str =None
    context: list = None

    class Config:
        arbitrary_types_allowed = True

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        messages_dict = [convert_message_to_dict(s) for s in messages]

        # Create chat history array.
        if self.context:
            chat_history = create_chat_history_array(self.context)
            messages_dict[1:1] = chat_history

        if not is_output_token_limited_model(str(self.engine)):
            # Reduce max token based on token consumed
            all_msgs = ""
            for message in messages_dict:
                all_msgs += str(message["content"])

            token_consumed = (
                self.get_num_tokens(
                    (
                        "" + all_msgs + self.user_query + json.dumps(kwargs["tools"])
                        if kwargs["tools"]
                        else ""
                    )
                )
                + self.min_response_token
            )
            if self.max_tokens - token_consumed <= 0:
                token_consumed = 0

            logger.info(
                f"total token by system_message, user_query, kwargs[tools] is - {token_consumed}"
            )
        else:
            token_consumed = 0

        response, total_token_completion = await ias_openai_chat_completion_with_tools(
            self.engine,
            self.temperature,
            self.max_tokens - token_consumed,
            self.client_id,
            self.x_vsl_client_id,
            self.r_parent_wf_req_id,
            self.bearer_token,
            messages_dict,
            kwargs["tools"],
            "auto",
        )
        logger.debug(f"Total tokens consumed: {total_token_completion}")

        self.total_consumed_token.append(total_token_completion)

        return self._create_chat_result(response)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        pass

    def _create_chat_result(
        self, response: Union[dict, openai.BaseModel, str]
    ) -> ChatResult:
        generations = []

        gen = ChatGeneration(
            message=convert_dict_to_message(response),
            generation_info=dict(finish_reason="stop"),
        )
        generations.append(gen)
        llm_output = {
            "token_usage": 0,
            "model_name": self.engine,
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "IAS_OpenAI"


def create_chat_history_array(context: list) -> list[dict]:
    chat_context = copy.deepcopy(context)
    chat_context.reverse()
    chat_history = []

    for i in range(len(context)):
        if i % 2 == 0:
            chat_history.append({"role": "user", "content": chat_context[i]})
        else:
            chat_history.append({"role": "assistant", "content": chat_context[i]})

    return chat_history


def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_)
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        if tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = tool_calls
        return AIMessage(content=content, additional_kwargs=additional_kwargs, id=id_)
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=_dict.get("name"), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=_dict.get("tool_call_id"),
            additional_kwargs=additional_kwargs,
            id=id_,
        )
    else:
        return ChatMessage(content=_dict.get("content", ""), role=role, id=id_)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == None:
                message_dict["content"] = ""
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == None:
                message_dict["content"] = ""
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
