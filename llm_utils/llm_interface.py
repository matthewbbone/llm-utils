from dotenv import load_dotenv
import os
import time
import numpy as np
import concurrent.futures
from tqdm import tqdm
import ast
from abc import ABC, abstractmethod

class LLMInterface(ABC):

    def __init__(self) -> None:
        load_dotenv()
        self.data_path = "data/processed/"
        self.cache_path = "pipeline/utils/cache/"

    @abstractmethod
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        pass

    def _get_default_batch_size(self, num_texts):
        return 100

    def _concurrent_embedding_call(
        self, 
        ids,
        texts, 
        embedding_model, 
        size, 
        db=None,
        batch_size=None,
        verbose=True
    ):
        if batch_size is None:
            batch_size = self._get_default_batch_size(len(texts))
        
        n_batches = len(texts) // batch_size + 1 if len(texts) > batch_size else 1
        text_batches = np.array_split(texts, n_batches)
        text_batches = [batch.tolist() for batch in text_batches]
        id_batches = np.array_split(ids, n_batches)
        id_batches = [batch.tolist() for batch in id_batches]
        
        output_texts = []
        output_embeddings = []
        
        for i, batch in tqdm(
            enumerate(text_batches), 
            desc="Embedding Batches", 
            total=len(text_batches),
            disable=not verbose
        ):
            
            embeddings = []
            n_minibatches = min(os.cpu_count(), len(batch))
            if n_minibatches < 1: n_minibatches = 1
            
            minibatches = np.array_split(np.array(batch), n_minibatches)
            minibatches = [minibatch.tolist() for minibatch in minibatches]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                
                results = list(executor.map(lambda minibatch: self._embedding_call( 
                    minibatch, 
                    embedding_model, 
                    size
                ), minibatches))
                
                for result in results:
                    if result:
                        embeddings += result
            
            if not db is None:
                db.add(
                    documents = text_batches[i],
                    ids = id_batches[i],
                    embeddings = embeddings
                )
            else:
                output_texts += text_batches[i]
                output_embeddings += embeddings
                
        if db is None:
            return output_texts, output_embeddings
        else:
            return [], []

    @abstractmethod
    def _chat_completion_call(
        self,
        model,
        message,
        web_search,
        max_tokens,
        max_tool_calls,
        tool_choice,
        system_message,
        response_format,
        verbosity,
        reasoning
    ):
        pass

    def _chat_completion(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        web_search=False,
        max_tool_calls=3,
        reasoning=None,
        verbosity=None,
        tool_choice=None,
        max_tokens=10_000,
        n_workers=1,
        verbose=True
    ):
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:

            results = list(tqdm(executor.map(lambda i: self._chat_completion_call(
                model,
                user_messages[i],
                web_search,
                max_tokens,
                max_tool_calls,
                tool_choice,
                system_message,
                response_formats[i],
                verbosity,
                reasoning
            ), np.arange(len(user_messages))), total=len(user_messages), desc="Chat Completion", disable=not verbose))

        return dict(zip(ids, results))

    def ask(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        web_search=False,
        max_tool_calls=3,
        tool_choice=None,
        reasoning=None,
        verbosity="medium",
        max_tokens=None,
        n_workers=1
    ):
    
        return self._chat_completion(
            system_message,
            ids,
            user_messages,
            response_formats,
            model,
            web_search=web_search,
            max_tool_calls=max_tool_calls,
            tool_choice=tool_choice,
            reasoning=reasoning,
            verbosity=verbosity,
            max_tokens=max_tokens,
            n_workers=n_workers
        )