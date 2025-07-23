from dotenv import load_dotenv
import os
import openai
import time
from datetime import datetime as dt
import numpy as np
import concurrent.futures
from tqdm import tqdm
import ast
import json
import pandas as pd

class OpenAIInterface():

    def __init__(self) -> None:
        
        load_dotenv()
        
        self.client = openai.Client(api_key=os.getenv("OPENAI_KEY"))
        self.data_path = "data/processed/"
        self.cache_path = "pipeline/utils/cache/"
        
        
    def _embedding_call(self, texts, embedding_model, size, max_retries=5):
        
        attempt = 0
        delay = 1
        while attempt < max_retries:
            
            try:
                response = self.client.embeddings.create(
                    input=texts, 
                    model=embedding_model,
                    dimensions=size
                )
                embeddings = [e.embedding for e in response.data]  
                return embeddings
            except Exception as e:
                attempt += 1
                time.sleep(delay)
                delay *= 2  # Exponential backoff
           
        print(f"Problematic text: {texts}")
        return None 
    
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
            # openai limits to 2048 per request
            batch_size = min(2048 * os.cpu_count(), len(texts))
            # chroma limits to 5461 uploads
            batch_size = min(5461, batch_size)
        
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
            minibatches = np.array_split(np.array(batch), n_minibatches)
            minibatches = [minibatch.tolist() for minibatch in minibatches]
            with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                
                results = list(executor.map(lambda minibatch: self._embedding_call( 
                    minibatch, 
                    embedding_model, 
                    size
                ), minibatches))
                
                for result in results:
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
    
    def _create_embedding_jsonl_job(
        self,
        ids, 
        texts, 
        embedding_model, 
        size, 
        batch_id, 
        file_name
    ):
        
        job_path = file_name + f"-embed-job-{batch_id}.jsonl"
        
        with open(job_path, "w") as f:
            
            for i, text in enumerate(texts):
                entry = {
                    "custom_id": ids[i],
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": {
                        "input": text,
                        "model": embedding_model,
                        "encoding_format": "float",
                        "dimensions": size
                    }
                }
                f.write(json.dumps(entry) + "\n")
        
        return job_path
    
    def _generator(self):
        while True:
            yield  
            
    def _create_batches(self, arr, batch_size):
        
        n_batches = len(arr) // batch_size + 1 if len(arr) > batch_size else 1
        batches = np.array_split(arr, n_batches)
        batches = [batch.tolist() for batch in batches]
        
        return batches
    
    def _wait_for_batch_job(self, batch_job_metas, verbose=True):
        
        # wait for batch jobs to finish
        status_result = []
        output_files = []
        for _ in tqdm(self._generator(), desc="Waiting for Batch Jobs", disable=not verbose):
            
            status_result = []
            output_files = []
            batch_job_statuses = []
            for batch_meta in batch_job_metas:
                meta = self.client.batches.retrieve(batch_meta.id)
                status = meta.status
                output_file_id = meta.output_file_id
                ready = status in ["failed", "completed", "expired", "cancelled"]
                status_result.append(status)
                output_files.append(output_file_id)
                batch_job_statuses.append(ready)
                
            if all(batch_job_statuses):
                break
        
            time.sleep(10)
            
        return status_result, output_files
    
    def _get_batch_results(self, output_files, status_result, file_name):
        
        # get batch job results
        result_paths = []
        for i, output_files in enumerate(output_files):
            if status_result[i] == "completed":
                result = self.client.files.content(output_files)
                result_path = f"{file_name}-result-{i}.jsonl"
                with open(result_path, "w") as f:
                    f.write(result.text)
                result_paths.append(result_path)
                
        return result_paths
                
    def _batch_embedding_call(
        self, 
        ids,
        texts, 
        embedding_model, 
        size, 
        db,
        name,
        verbose=True
    ):
        
        # map from id to text
        id_to_text = dict(zip(ids, texts))
        
        file_name = f"{self.cache_path}{name}-{dt.today().strftime("%Y%m%d%H%M%S")}"
        
        batch_size = 5460 # batch api limit is 50,000 but chroma is 5461
        id_batches = self._create_batches(ids, batch_size)
        text_batches = self._create_batches(texts, batch_size)
        
        batch_job_metas = []
        
        # create batch jobs
        for i, batch in tqdm(
            enumerate(text_batches), 
            desc="Embedding Batches", 
            total=len(text_batches),
            disable=not verbose
        ):
        
            job_path = self._create_embedding_jsonl_job(
                id_batches[i], 
                batch, 
                embedding_model, 
                size, 
                i, 
                file_name
            )
            
            input_file = self.client.files.create(
                file=open(job_path, "rb"),
                purpose="batch"
            )
            
            input_id = input_file.id
            batch_meta = self.client.batches.create(
                input_file_id=input_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": f"Batch {i} for {name}",
                }
            )
            
            batch_job_metas.append(batch_meta)
            
        status_result, output_files = self._wait_for_batch_job(
            batch_job_metas, 
            verbose=verbose
        )
        
        result_paths = self._get_batch_results(
            output_files,
            status_result,
            file_name
        )
                
        # parse batch job results
        for i, path in enumerate(result_paths):
            
            embedding_batch = []
            id_batch = []
            text_batch = []
            with open(path, "r") as f:
                for line in f:
                    
                    try:
                        result = json.loads(line)
                        id = result["custom_id"]
                        embedding = result["response"]["body"]["data"][0]["embedding"]
                    except Exception as e:
                        print(f"Error parsing line: {line}")
                        print(f"Error: {e}")
                        continue
                    
                    text = id_to_text[id]
                    embedding_batch.append(embedding)
                    id_batch.append(id)
                    text_batch.append(text)
                    
            db.add(
                documents = text_batch,
                ids = id_batch,
                embeddings = embedding_batch
            )
        
        
    def save_embeddings(
        self, 
        ids,
        texts, 
        size, 
        db,
        name,
        batch=False,
        verbose=True
    ):
        
        if size > 1536:
            embedding_model = "text-embedding-3-large"
        else:
            embedding_model = "text-embedding-3-large"
    
        if not batch:
            output_texts, output_embeddings = self._concurrent_embedding_call(
                ids,
                texts, 
                embedding_model, 
                size,
                db,
                verbose=verbose
            )
            return output_texts, output_embeddings
        else:
            self._batch_embedding_call(
                ids,
                texts,
                embedding_model,
                size,
                db,
                name,
                verbose=verbose
            )
            return [], []
        
    def _create_chat_completion_jsonl_job(
        self, 
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        web_search,
        max_tokens,
        batch_id,
        file_name
    ):
    
        job_path = file_name + f"-chat-job-{batch_id}.jsonl"
        
        with open(job_path, "w") as f:
            
            for i, text in enumerate(user_messages):
                entry = {
                    "custom_id": str(ids[i]),
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": model,
                        "input": text,
                        "instructions": system_message,
                        "max_completion_tokens": max_tokens,
                        "text": {"format": response_formats[i]}
                    }
                }
                f.write(json.dumps(entry) + "\n")
                
        return job_path
        
    def _batch_chat_completion_call(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        name,
        web_search=False,
        max_tokens=1000,
        verbose=True,
        from_cache=None
    ):
        
        file_name = f"{self.cache_path}{dt.today().strftime('%Y%m%d%H%M%S')}"
        
        batch_size = 5460 # batch api limit is 50,000 but chroma is 5461
        id_batches = self._create_batches(ids, batch_size)
        message_batches = self._create_batches(user_messages, batch_size)
        response_batches = self._create_batches(response_formats, batch_size)
        
        batch_job_metas = []
        
        if from_cache is None:
        
            # create batch jobs
            for i, batch in tqdm(
                enumerate(message_batches),
                desc="Chat Completion Batches",
                total=len(message_batches),
                disable=not verbose
            ):
                
                job_path = self._create_chat_completion_jsonl_job(
                    system_message,
                    id_batches[i],
                    batch,
                    response_batches[i],
                    model,
                    web_search,
                    max_tokens,
                    i,
                    file_name
                )
                
                input_file = self.client.files.create(
                    file=open(job_path, "rb"),
                    purpose="batch"
                )
                
                input_id = input_file.id
                batch_meta = self.client.batches.create(
                    input_file_id=input_id,
                    endpoint="/v1/responses",
                    completion_window="24h",
                    metadata={
                        "description": f"Batch {i} of for {name}",
                    }
                )
                
                batch_job_metas.append(batch_meta)
                
            status_result, output_files = self._wait_for_batch_job(
                batch_job_metas, 
                verbose=verbose
            )
            
            result_paths = self._get_batch_results(
                output_files,
                status_result,
                file_name
            )

        else:
            
            result_paths = []
            for file in os.listdir(self.cache_path):
                if file.startswith(from_cache) and "result" in file:
                    result_paths.append(os.path.join(self.cache_path, file))
                    
                    
        # parse batch job results
        output_results = []
        output_ids = []
        for i, path in enumerate(result_paths):
        
            with open(path, "r") as f:
                for line in f:
                    
                    response = json.loads(line)
                    id = response["custom_id"]
                    
                    try:
                        message = response["response"]["output"][0]["content"][0]["text"]
                        message = ast.literal_eval(message)
                    except:
                        message = response["error"]
                        print(f"Error parsing line: {line}")
                    
                    output_results.append(message)
                    output_ids.append(id)
                    
        return dict(zip(output_ids, output_results))
                  
    def _chat_completion(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        web_search=False,
        max_tool_calls=3,
        tool_choice=None,
        max_tokens=10_000,
        verbose=True
    ):
        
        if web_search:
            tools = [{ "type": "web_search_preview" }]
        else:
            tools = []
        
        results = []
        for i, message in tqdm(enumerate(user_messages), total=len(user_messages), desc="Chat Completion", disable=not verbose):
            
            try:
                res = self.client.responses.create(
                    model=model,
                    input=message,
                    tools=tools,
                    max_output_tokens=max_tokens,
                    max_tool_calls=max_tool_calls,
                    tool_choice=tool_choice,
                    instructions=system_message,
                    text={"format": response_formats[i]},
                )
                
                results.append(ast.literal_eval(res.output_text))
            except Exception as e:
                results.append({ "error": str(e) })

        return dict(zip(ids, results))
        
            
    def ask_gpt(
        self,
        system_message,
        ids,
        user_messages,
        response_formats,
        model,
        name,
        web_search=False,
        max_tool_calls=3,
        tool_choice=None,
        max_tokens=1000,
        batch=False,
        from_cache=None
    ):
    
        if not batch:
            return self._chat_completion(
                system_message,
                ids,
                user_messages,
                response_formats,
                model,
                web_search=web_search,
                max_tool_calls=max_tool_calls,
                tool_choice=tool_choice,
                max_tokens=max_tokens
            )
        else:
            return self._batch_chat_completion_call(
                system_message,
                ids,
                user_messages,
                response_formats,
                model,
                name,
                web_search=web_search,
                max_tokens=max_tokens,
                from_cache=from_cache
            )
    
    
