from typing import Mapping, List, Tuple, Union

import pandas as pd, json, time 
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document, validate_truncate_document_parameters
from bertopic.vectorizers import ClassTfidfTransformer
from tqdm import tqdm 

class AnthropicRepresentationModel(BaseRepresentation):
    def __init__(self, client=None, model_id="claude-sonnet-4-20250514", prompt=None, system_prompt=None, delay_in_seconds=2, nr_docs=5, diversity=None, generator_kwargs=None, doc_length=300, tokenizer=None, **kwargs):
        if not hasattr(client, "invoke_model"): raise ValueError("Invalid client.")
        self.client=client; self.model_id=model_id
        self.prompt = prompt; self.system_prompt = system_prompt
        self.delay_in_seconds = delay_in_seconds; self.nr_docs = nr_docs; self.diversity = diversity
        self.prompts_ = []; self.generator_kwargs = generator_kwargs or {'max_tokens': 250, 'temperature': 0.6}
        self.generator_kwargs.setdefault("max_tokens", 250)
        
    
    def extract_topics(self, topic_model, documents:pd.DataFrame, c_tf_idf: csr_matrix, topics: Mapping[str, List[Tuple[str, float]]]) -> Mapping[str, List[Tuple[str, float]]]:
        """ Extract topics

        Arguments:
            topic_model: The BERTopic model
            documents: A dataframe of documents with their related topics
            c_tf_idf: The c-TF-IDF matrix
            topics: The candidate topics as calculated with c-TF-IDF

        Returns:
            updated_topics: Updated topic representations
        """
        if not topics: return {}
        try: repr_docs_mappings, _, _, _ = topic_model._extract_representative_docs(c_tf_idf, documents, topics, 500, self.nr_docs, self.diversity)
        except Exception as e: print(e)
        updated_topics = {}
        for topic, docs in tqdm(repr_docs_mappings.items(), desc='Generating Topics', disable=True):
            if topic == -1: continue
            if self.doc_length and self.tokenizer:
                truncated_docs = [truncate_document(topic_model, self.doc_length, self.tokenizer, doc) for doc in docs]
            else:
                truncated_docs = docs 
            
            try: prompt_content = self._create_prompt(truncated_docs, topic, topics); self.prompts_.append(prompt_content)
            except Exception as e: print(e); continue
            
            max_retries=3; retry_count=0; analysis_text=None
            while retry_count < max_retries:
                try: 
                    messages=[{'role': 'user', 'content': prompt_content}]
                    body={'system': self.system_prompt, 'messages': messages, **self.generator_kwargs}
                    body_bytes = json.dumps(body).encode('utf-8-sig')
                    invoke_params = {'modelId': self.model_id, 'body': body_bytes, 'contentType': 'application/json'}
                    response = self.client.invoke_model(**invoke_params)
                    response_body = json.loads(response['body'].read().decode('utf-8-sig'))
                    raw_analysis=""
                    if response_body.get('content') and isinstance(response_body['content'], list) and len(response_body['content'] > 0):
                        raw_analysis = response_body['content'][0].get('text', '')
                    
                    analysis_text = raw_analysis.strip()
                    break 

                except Exception as e: print(e); retry_count += 1; time.sleep(self.delay_in_seconds * (retry_count + 1)); continue
            
            if analysis_text is None: analysis_text = "Error: No analysis gererated"
            score=1.0 if not analysis_text.startswith("Error:") else 0.0
            updated_topics[topic] = [(analysis_text, score)]
        if -1 in topics and -1 not in updated_topics: updated_topics[-1] = [(getattr(topic_model, 'topic_labels_', {}).get(-1, 'Outlier Topic'), 1.0)]    
        return updated_topics

    def _create_prompt(self, docs: List[str], topic: int, topics: Mapping[str, List[Tuple[str, float]]]) -> str:
        if topic not in topics: raise KeyError(f"Topic ID {topic} not found.")
        keywords = []
        if topics[topic]:
            try: keywords = list(zip(*topics[topic]))[0]
            except IndexError: keywords = []
        prompt=self.prompt 
        if "[KEYWORDS]" in prompt: prompt = prompt.replace("[KEYWORDS]", ", ".join(keywords))
        if "[DOCUMENTS]" in prompt: doc_string="\n".join([f"- {' '.join(doc.split())}" for doc in docs]); prompt=prompt.replace("[DOCUMENTS]", doc_string.strip())
        return prompt
