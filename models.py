from typing import Mapping, List, Tuple, Union

import pandas as pd, json, time
from scipy.sparse import csr_matrix

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer 
from bertopic.representation import TextGeneration
from bertopic.representation._base import BaseRepresentation
from bertopic.representation._utils import truncate_document
from tqdm import tqdm 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA 

DEFAULT_PROMPT = """
[INST]
I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
[/INST]
"""


class BasicBERTopicModel:
    def __init__(
        self,
        language: str = "korean",
        top_n_words: int = 10,
        n_gram_range: Tuple[int, int] = (1, 1),
        min_topic_size: int = 10,
        nr_topics: Union[int, str] = None,
        low_memory: bool = False,
        embedding_model=None,
        umap_model=None,
        hdbscan_model=None,
        vectorizer_model: CountVectorizer = None,
        ctfidf_model: TfidfTransformer = None,
        representation_model: BaseRepresentation = None,
        verbose: bool = False,
        embed_model_id=None,
        repr_model_id=None,
        prompt=None,
        **kwargs
    ):
        self.language=language; self.top_n_words=top_n_words; self.n_gram_range=n_gram_range
        self.min_topic_size=min_topic_size; self.nr_topics=nr_topics; self.low_memory=low_memory
        self.embedding_model=embedding_model; self.umap_model=umap_model; self.hdbscan_model=hdbscan_model; self.embed_model_id=embed_model_id; self.repr_model_id=repr_model_id
        self.vectorizer_model=vectorizer_model; self.ctfidf_model=ctfidf_model; self.representation_model=representation_model; self.verbose=verbose; self.prompt=prompt
        
        if self.embedding_model is None: 
            try: self.embedding_model = self._load_embedding_model()
            except Exception as e: print(e)
        if self.prompt is None: prompt=DEFAULT_PROMPT
        if self.umap_model is None: self.umap_model = self._load_umap_model()
        if self.hdbscan_model is None: self.hdbscan_model = self._load_hdbscan_model()
        if self.vectorizer_model is None: self.vectorizer_model = self._load_vectorizer_model()
        if self.ctfidf_model is None: self.ctfidf_model = self._load_ctfidf_model()
        if self.representation_model is None: self.representation_model = self._load_representation_model()
        
        self.topic_model = BERTopic(
            language=self.language, 
            top_n_words=self.top_n_words, 
            n_gram_range=self.n_gram_range,
            nr_topics=self.nr_topics,
            low_memory=self.low_memory,
            embedding_model=self.embedding_model, 
            umap_model=self.umap_model, 
            hdbscan_model=self.hdbscan_model, 
            vectorizer_model=self.vectorizer_model, 
            ctfidf_model=self.ctfidf_model,
            representation_model=self.representation_model, 
            verbose=self.verbose, 
            **kwargs
        )
        
    def access_huggingface(self, token):
        from huggingface_hub import login
        if token is None: ValueError('please input your huggingface api key')
        login(token)
        

    def _load_embedding_model(self, model_id='BAAI/bge-m3'):
        try: from sentence_transformers import SentenceTransformer
        except Exception as e: print(e)
        try: embedding_model = SentenceTransformer(model_id)
        except Exception as e: print(e)
        return embedding_model
    
    def _load_umap_model(self):
        try: 
            from umap import UMAP
            umap_model = UMAP(
                n_neighbors=15, 
                n_components=5, 
                min_dist=0.0, 
                metric='cosine',
                low_memory=self.low_memory
            )
            return umap_model 
        
        except (ImportError, ModuleNotFoundError):
            umap_model = PCA(n_components=5)
            return umap_model
    
    def _load_hdbscan_model(self):
        from sklearn.cluster import HDBSCAN
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        return hdbscan_model
    
    def _load_vectorizer_model(self):
        vectorizer_model = CountVectorizer(ngram_range=self.n_gram_range)
        return vectorizer_model 
    
    def _load_ctfidf_model(self):
        ctfidf_model = ClassTfidfTransformer()
        return ctfidf_model 
    
    def _load_representation_model(self, model_id='google-bert/bert-base-uncased'):
        import transformers
        self.access_huggingface()
        from torch import bfloat16
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,  # 4-bit quantization
            bnb_4bit_quant_type='nf4',  # Normalized float 4
            bnb_4bit_use_double_quant=True,  # Second quantization after the first
            bnb_4bit_compute_dtype=bfloat16  # Computation type
        )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remot_code=True, 
            quantization_config=bnb_config if self.quantization else None, 
            device_map='auto'
        )
        model.eval()
        
        generator = transformers.pipeline(
            model=model, tokenizer=tokenizer,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=500,
            repetition_penalty=1.1
        )
        representation_model = TextGeneration(generator, prompt=DEFAULT_PROMPT)
        return representation_model
        
    def set_internal_parameters(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


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
