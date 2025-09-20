import os
import json
import threading
import subprocess
import pandas as pd
from pydantic import PrivateAttr
import tiktoken
import numpy as np

from SyncTokenRateLimiter import (
    SyncTokenRateLimiter,
    get_llm_token_limiter,
    get_embedding_token_limiter,
)

from tqdm import tqdm
from retry import retry
from typing import Dict, Any
from typing import List, Optional
from neo4j import GraphDatabase, Result
from Logger import Logger, monitor_system
from graphdatascience import GraphDataScience
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.vectorstores import Neo4jVector
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.rate_limiters import InMemoryRateLimiter

from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.core.vector_stores.utils import node_to_metadata_dict

from openai import LengthFinishReasonError


class DuplicateEntities(BaseModel):
    entities: List[str] = Field(
        description="Entities that represent the same object or real-world entity and should be merged"
    )


class Disambiguate(BaseModel):
    merge_entities: Optional[List[DuplicateEntities]] = Field(
        description="Lists of entities that represent the same object or real-world entity and should be merged"
    )


class SafeRateLimitedOpenAIEmbedding(OpenAIEmbedding):
    """
    OpenAIEmbedding subclass with synchronous token-limited embedding for queries/documents.
    Compatible with VectorStoreIndex and other LangChain code expecting OpenAIEmbedding.
    """

    _token_limiter: Optional[object] = PrivateAttr(default=None)
    _max_chunk_tokens: int = PrivateAttr(default=4000)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # store model explicitly so we can reference it safely
        self.model_name = kwargs.get("model") or getattr(self, "model", None)

    def set_limits(self, token_limiter, max_chunk_tokens: int = 4000):
        """Assign a SyncTokenRateLimiter and optional max chunk size."""
        self._token_limiter = token_limiter
        self._max_chunk_tokens = int(max_chunk_tokens)

    def _chunk_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        model: str = "text-embedding-ada-002",
    ) -> List[str]:
        """Token-aware splitting of text into chunks."""
        if max_tokens is None:
            max_tokens = self._max_chunk_tokens
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        return chunks or [""]

    def embed_query(self, text: str):
        """Token-limited embedding for a single query."""
        chunks = self._chunk_text(
            text,
            max_tokens=self._max_chunk_tokens,
            model=self.model_name or "text-embedding-ada-002",
        )
        chunk_embeddings = []

        for chunk in chunks:
            if self._token_limiter is not None:
                token_count = self._token_limiter.count_tokens(
                    chunk, model=self.model_name or "text-embedding-ada-002"
                )
                self._token_limiter.wait_for_tokens(token_count)
            # call parent batch method for a single item
            emb = self._get_query_embedding(chunk)
            chunk_embeddings.append(np.array(emb, dtype=float))
        if chunk_embeddings:
            return np.mean(chunk_embeddings, axis=0).tolist()
        return []

    def embed_documents(self, texts: List[str]):
        """Token-limited embedding for multiple documents."""
        all_embeddings = []

        for text in texts:
            chunks = self._chunk_text(
                text,
                max_tokens=self._max_chunk_tokens,
                model=self.model_name or "text-embedding-ada-002",
            )
            chunk_embeddings = []

            for chunk in chunks:
                if self._token_limiter is not None:
                    token_count = self._token_limiter.count_tokens(
                        chunk, model=self.model_name or "text-embedding-ada-002"
                    )
                    self._token_limiter.wait_for_tokens(token_count)
                emb = self._get_text_embedding(chunk)
                chunk_embeddings.append(np.array(emb, dtype=float))
            if chunk_embeddings:
                all_embeddings.append(np.mean(chunk_embeddings, axis=0).tolist())
            else:
                all_embeddings.append([])

        return all_embeddings

    def get_agg_embedding_from_queries(self, queries: List[str]):
        """Aggregate embeddings from a list of queries (required by VectorStoreIndex)."""
        all_embeds = self.embed_documents(queries)
        # filter empty embeddings
        all_embeds = [np.array(e, dtype=float) for e in all_embeds if e]
        if all_embeds:
            return np.mean(all_embeds, axis=0).tolist()
        # fallback to zero vector
        return [0.0] * self.embedding_dimension


class SafeRateLimitedChunkedOpenAIEmbeddings(OpenAIEmbeddings):
    # private attrs (not pydantic-managed fields)
    _token_limiter: Optional[SyncTokenRateLimiter] = PrivateAttr(default=None)
    _max_chunk_tokens: int = PrivateAttr(default=4000)

    def set_limits(
        self, token_limiter: SyncTokenRateLimiter, max_chunk_tokens: int = 4000
    ):
        self._token_limiter = token_limiter
        self._max_chunk_tokens = int(max_chunk_tokens)

    def _chunk_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        model: str = "text-embedding-ada-002",
    ) -> List[str]:
        if max_tokens is None:
            max_tokens = self._max_chunk_tokens
        # Defensive: ensure integer
        if isinstance(max_tokens, tuple):
            max_tokens = int(max_tokens[0])
        max_tokens = int(max_tokens)

        if not isinstance(text, str):
            text = "" if text is None else str(text)
        enc = tiktoken.encoding_for_model(model)
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
        # ensure at least one empty chunk for empty text
        return chunks or [""]

    def embed_query(self, text: str):
        # chunk, rate-limit per chunk, call super().embed_documents for each chunk batch
        chunks = self._chunk_text(
            text,
            max_tokens=self._max_chunk_tokens,
            model=self.model or "text-embedding-ada-002",
        )
        chunk_embeddings = []
        for chunk in chunks:
            if self._token_limiter is not None:
                token_count = self._token_limiter.count_tokens(
                    chunk, model=self.model or "text-embedding-ada-002"
                )
                self._token_limiter.wait_for_tokens(token_count)
            # call parent batch method for a single item to avoid recursion
            emb = super().embed_documents([chunk])[0]
            chunk_embeddings.append(np.array(emb, dtype=float))
        if chunk_embeddings:
            return np.mean(chunk_embeddings, axis=0).tolist()
        return []

    def embed_documents(self, texts: List[str]):
        all_embeddings = []
        for text in texts:
            chunks = self._chunk_text(
                text,
                max_tokens=self._max_chunk_tokens,
                model=self.model or "text-embedding-ada-002",
            )
            chunk_embeddings = []
            for chunk in chunks:
                if self._token_limiter is not None:
                    token_count = self._token_limiter.count_tokens(
                        chunk, model=self.model or "text-embedding-ada-002"
                    )
                    self._token_limiter.wait_for_tokens(token_count)
                # IMPORTANT: use super().embed_documents to avoid recursive embed_query -> embed_documents loops
                emb = super().embed_documents([chunk])[0]
                chunk_embeddings.append(np.array(emb, dtype=float))
            if chunk_embeddings:
                all_embeddings.append(np.mean(chunk_embeddings, axis=0).tolist())
            else:
                all_embeddings.append([])
        return all_embeddings


class RateLimitedChatOpenAI(ChatOpenAI):
    def __init__(self, token_limiter: SyncTokenRateLimiter, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._token_limiter = token_limiter

    def _count_tokens(self, prompt: str) -> int:
        # Count tokens using tiktoken or your preferred method
        import tiktoken

        enc = tiktoken.encoding_for_model(self.model_name)
        return len(enc.encode(prompt))

    def _call(self, prompt: str, stop=None):
        tokens_needed = self._count_tokens(prompt) + (self.max_completion_tokens or 0)
        if self._token_limiter is not None:
            self._token_limiter.wait_for_tokens(tokens_needed)
        return super()._call(prompt, stop=stop)


def chunk_text_by_tokens(
    text: str, max_tokens: int = 4000, model: str = "text-embedding-ada-002"
) -> List[str]:
    """
    Split a text string into chunks that are <= max_tokens tokens each.
    Returns a list of text chunks.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks


def createGRAG(rate_limiter):
    with open("NeoGrag/neo4j_creds.json", "r") as f:
        creds = json.load(f)

    GRAPHRAG_API_KEY = creds["GRAPHRAG_API_KEY"]

    os.environ["NEO4J_URI"] = creds["NEO4J_URI"]
    os.environ["NEO4J_USERNAME"] = creds["NEO4J_USERNAME"]
    os.environ["NEO4J_PASSWORD"] = creds["NEO4J_PASSWORD"]

    graph = Neo4jGraph()

    def num_tokens_from_string(string: str, model: str = "gpt-4.1-nano") -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    data = pd.read_csv("NeoGrag/cord_texts.csv")
    data["tokens"] = [
        num_tokens_from_string(f"{row['title']} {row['text']}")
        for i, row in data.iterrows()
    ]
    data.head()

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1, check_every_n_seconds=1, max_bucket_size=1
    )

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4.1-nano",
        api_key=GRAPHRAG_API_KEY,
        rate_limiter=rate_limiter,
        max_completion_tokens=8000,
    )

    llm_transformer = LLMGraphTransformer(
        llm=llm,
        node_properties=["description"],
        relationship_properties=["description"],
    )

    enc = tiktoken.encoding_for_model("gpt-4.1-nano")

    def split_by_tokens(text: str, max_tokens: int = 2000) -> List[str]:
        """Split text into chunks by token count."""
        tokens = enc.encode(text)
        chunks = []
        for i in range(0, len(tokens), max_tokens):
            chunk = enc.decode(tokens[i : i + max_tokens])
            chunks.append(chunk)
        return chunks

    def process_text(title, text: str, max_tokens: int = 2000) -> List[GraphDocument]:
        """use token-aware splitting instead of fixed character size"""
        print(f"processing: {title}")
        chunks = split_by_tokens(text, max_tokens=max_tokens)
        docs = [Document(page_content=chunk) for chunk in chunks]

        graph_docs = []
        for doc in docs:
            num_tokens_from_string(doc.page_content)

        for doc in docs:
            counter = 0
            while counter < 10:
                try:
                    graph_docs.extend(llm_transformer.convert_to_graph_documents([doc]))
                    counter = 10
                except LengthFinishReasonError as e:
                    if counter < 10:
                        counter += 1
                        print(f"retry {counter} of 10")
                    else:
                        return
        return graph_docs

    graph_documents = []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing documents"):
        graph_document = process_text(row["title"], row["text"])
        graph_documents.extend(graph_document)

    graph.add_graph_documents(
        graph_documents, baseEntityLabel=True, include_source=True
    )
    driver = GraphDatabase.driver(
        creds["NEO4J_URI"], auth=(creds["NEO4J_USERNAME"], creds["NEO4J_PASSWORD"])
    )

    # with driver.session() as session:
    #     # Pull all Document nodes
    #     results = session.run(
    #         """
    #         MATCH (d:Document)
    #         RETURN d.id AS id, d.page_content AS page_content, d.source AS source
    #     """
    #     )
    #     documents = []
    #     for record in results:
    #         documents.append(
    #             {
    #                 "id": record["id"],
    #                 "text": record["page_content"] or "",
    #                 "source": record.get("source"),
    #             }
    #         )

    # graph_documents = [
    #     Document(
    #         page_content=d["text"], metadata={"id": d["id"], "source": d.get("source")}
    #     )
    #     for d in documents
    # ]

    with driver.session() as session:
        result = session.run(
            "MATCH (e:__Entity__) RETURN e.id AS id, e.description AS description"
        )
        records = list(result)

    node_ids = [record["id"] for record in records]
    node_descriptions = [record["description"] for record in records]

    documents = [
        {"id": node_id, "description": node_description}
        for node_id, node_description in zip(node_ids, node_descriptions)
    ]
    chunked_documents = []

    for doc in documents:
        chunks = chunk_text_by_tokens(doc["description"], max_tokens=4000)
        for i, chunk in enumerate(chunks):
            chunked_documents.append(
                {"id": f"{doc['id']}_{i}", "description": chunk}  # unique ID per chunk
            )

    emb = SafeRateLimitedChunkedOpenAIEmbeddings(
        api_key=GRAPHRAG_API_KEY,
        model="text-embedding-ada-002",
    )
    emb.set_limits(get_embedding_token_limiter(), max_chunk_tokens=4000)

    vector = Neo4jVector.from_existing_graph(
        emb,
        node_label="__Entity__",
        text_node_properties=["id", "description"],
        embedding_node_property="embedding",
    )

    gds = GraphDataScience(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    if gds.graph.exists("entities").exists:
        gds.graph.drop("entities")

    G, result = gds.graph.project(
        "entities",  #  Graph name
        "__Entity__",  #  Node projection
        "*",  #  Relationship projection
        nodeProperties=["embedding"],  #  Configuration parameters
    )

    similarity_threshold = 0.95

    gds.knn.mutate(
        G,
        nodeProperties=["embedding"],
        mutateRelationshipType="SIMILAR",
        mutateProperty="score",
        similarityCutoff=similarity_threshold,
    )

    gds.wcc.write(G, writeProperty="wcc", relationshipTypes=["SIMILAR"])

    word_edit_distance = 3
    potential_duplicate_candidates = graph.query(
        """MATCH (e:`__Entity__`)
        WHERE size(e.id) > 4 // longer than 4 characters
        WITH e.wcc AS community, collect(e) AS nodes, count(*) AS count
        WHERE count > 1
        UNWIND nodes AS node
        // Add text distance
        WITH distinct
        [n IN nodes WHERE apoc.text.distance(toLower(node.id), toLower(n.id)) < $distance | n.id] AS intermediate_results
        WHERE size(intermediate_results) > 1
        WITH collect(intermediate_results) AS results
        // combine groups together if they share elements
        UNWIND range(0, size(results)-1, 1) as index
        WITH results, index, results[index] as result
        WITH apoc.coll.sort(reduce(acc = result, index2 IN range(0, size(results)-1, 1) |
                CASE WHEN index <> index2 AND
                    size(apoc.coll.intersection(acc, results[index2])) > 0
                    THEN apoc.coll.union(acc, results[index2])
                    ELSE acc
                END
        )) as combinedResult
        WITH distinct(combinedResult) as combinedResult
        // extra filtering
        WITH collect(combinedResult) as allCombinedResults
        UNWIND range(0, size(allCombinedResults)-1, 1) as combinedResultIndex
        WITH allCombinedResults[combinedResultIndex] as combinedResult, combinedResultIndex, allCombinedResults
        WHERE NOT any(x IN range(0,size(allCombinedResults)-1,1)
            WHERE x <> combinedResultIndex
            AND apoc.coll.containsAll(allCombinedResults[x], combinedResult)
        )
        RETURN combinedResult
        """,
        params={"distance": word_edit_distance},
    )
    potential_duplicate_candidates[:5]

    system_prompt = """You are a data processing assistant. Your task is to identify duplicate entities in a list and decide which of them should be merged.
    The entities might be slightly different in format or content, but essentially refer to the same thing. Use your analytical skills to determine duplicates.

    Here are the rules for identifying duplicates:
    1. Entities with minor typographical differences should be considered duplicates.
    2. Entities with different formats but the same content should be considered duplicates.
    3. Entities that refer to the same real-world object or concept, even if described differently, should be considered duplicates.
    4. If it refers to different numbers, dates, or products, do not merge results
    """
    user_template = """
    Here is the list of entities to process:
    {entities}

    Please identify duplicates, merge them, and provide the merged list.
    """

    extraction_llm = ChatOpenAI(
        model_name="gpt-4.1-nano",
        api_key=GRAPHRAG_API_KEY,
        rate_limiter=rate_limiter,
        max_completion_tokens=8000,
    ).with_structured_output(Disambiguate)

    extraction_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            (
                "human",
                user_template,
            ),
        ]
    )

    extraction_chain = extraction_prompt | extraction_llm

    @retry(tries=3, delay=2)
    def entity_resolution(entities: List[str]) -> Optional[List[str]]:
        return [
            el.entities
            for el in extraction_chain.invoke({"entities": entities}).merge_entities
        ]

    merged_entities = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Submitting all tasks and creating a list of future objects
        futures = [
            executor.submit(entity_resolution, el["combinedResult"])
            for el in potential_duplicate_candidates
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing documents"
        ):
            to_merge = future.result()
            if to_merge:
                merged_entities.extend(to_merge)

    graph.query(
        """
    UNWIND $data AS candidates
    CALL {
    WITH candidates
    MATCH (e:__Entity__) WHERE e.id IN candidates
    RETURN collect(e) AS nodes
    }
    CALL apoc.refactor.mergeNodes(nodes, {properties: {
        `.*`: 'discard'
    }})
    YIELD node
    RETURN count(*)
    """,
        params={"data": merged_entities},
    )

    G.drop()

    if gds.graph.exists("communities").exists:
        gds.graph.drop("communities")

    G, result = gds.graph.project(
        "communities",  #  Graph name
        "__Entity__",  #  Node projection
        {
            "_ALL_": {
                "type": "*",
                "orientation": "UNDIRECTED",
                "properties": {"weight": {"property": "*", "aggregation": "COUNT"}},
            }
        },
    )

    gds.leiden.write(
        G,
        writeProperty="communities",
        includeIntermediateCommunities=True,
        relationshipWeightProperty="weight",
    )

    graph.query(
        "CREATE CONSTRAINT IF NOT EXISTS FOR (c:__Community__) REQUIRE c.id IS UNIQUE;"
    )

    graph.query(
        """
    MATCH (e:`__Entity__`)
    UNWIND range(0, size(e.communities) - 1 , 1) AS index
    CALL {
    WITH e, index
    WITH e, index
    WHERE index = 0
    MERGE (c:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
    ON CREATE SET c.level = index
    MERGE (e)-[:IN_COMMUNITY]->(c)
    RETURN count(*) AS count_0
    }
    CALL {
    WITH e, index
    WITH e, index
    WHERE index > 0
    MERGE (current:`__Community__` {id: toString(index) + '-' + toString(e.communities[index])})
    ON CREATE SET current.level = index
    MERGE (previous:`__Community__` {id: toString(index - 1) + '-' + toString(e.communities[index - 1])})
    ON CREATE SET previous.level = index - 1
    MERGE (previous)-[:IN_COMMUNITY]->(current)
    RETURN count(*) AS count_1
    }
    RETURN count(*)
    """
    )

    graph.query(
        """
    MATCH (c:__Community__)<-[:IN_COMMUNITY*]-(:__Entity__)<-[:MENTIONS]-(d:Document)
    WITH c, count(distinct d) AS rank
    SET c.community_rank = rank;
    """
    )

    community_info = graph.query(
        """
    MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
    WHERE c.level IN [0,1,4]
    WITH c, collect(e ) AS nodes
    WHERE size(nodes) > 1
    CALL apoc.path.subgraphAll(nodes[0], {
        whitelistNodes:nodes
    })
    YIELD relationships
    RETURN c.id AS communityId,
        [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
        [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
    """
    )

    community_template = """Based on the provided nodes and relationships that belong to the same graph community,
    generate a natural language summary of the provided information:
    {community_info}

    Summary:"""  # noqa: E501

    community_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given an input triples, generate the information summary. No pre-amble.",
            ),
            ("human", community_template),
        ]
    )

    community_chain = community_prompt | llm | StrOutputParser()

    def prepare_string(data):
        nodes_str = "Nodes are:\n"
        for node in data["nodes"]:
            node_id = node["id"]
            node_type = node["type"]
            if "description" in node and node["description"]:
                node_description = f", description: {node['description']}"
            else:
                node_description = ""
            nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

        rels_str = "Relationships are:\n"
        for rel in data["rels"]:
            start = rel["start"]
            end = rel["end"]
            rel_type = rel["type"]
            if "description" in rel and rel["description"]:
                description = f", description: {rel['description']}"
            else:
                description = ""
            rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

        return nodes_str + "\n" + rels_str

    def process_community(community):
        stringify_info = prepare_string(community)
        summary = community_chain.invoke({"community_info": stringify_info})
        return {"community": community["communityId"], "summary": summary}

    summaries = []
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_community, community): community
            for community in community_info
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing communities"
        ):
            summaries.append(future.result())

    graph.query(
        """
    UNWIND $data AS row
    MERGE (c:__Community__ {id:row.community})
    SET c.summary = row.summary
    """,
        params={"data": summaries},
    )


def db_query(cypher: str, creds, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """Executes a Cypher statement and returns a DataFrame"""

    driver = GraphDatabase.driver(
        creds["NEO4J_URI"], auth=(creds["NEO4J_USERNAME"], creds["NEO4J_PASSWORD"])
    )
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )


def query_grag(type, query):
    with open("NeoGrag/neo4j_creds.json", "r") as f:
        creds = json.load(f)
    GRAPHRAG_API_KEY = creds["GRAPHRAG_API_KEY"]
    os.environ["OPENAI_API_KEY"] = GRAPHRAG_API_KEY

    if type == "local":
        topChunks = 3
        topCommunities = 3
        topOutsideRels = 10
        topInsideRels = 10
        topEntities = 10
        index_name = "entity"
        content = node_to_metadata_dict(
            TextNode(), remove_text=True, flat_metadata=False
        )

        db_query(
            """
        MATCH (e:__Entity__)
        SET e += $content""",
            creds,
            {"content": content},
        )
        embed_dim = 1536

        community_level = 0  # choose the summary level you want

        retrieval_query = f"""
        WITH collect(node) as nodes
        // Entity - Text Unit Mapping
        WITH
        nodes,
        collect {{
            UNWIND nodes as n
            MATCH (n)<-[:HAS_ENTITY]->(c:__Chunk__)
            WITH c, count(distinct n) as freq
            RETURN c.text AS chunkText
            ORDER BY freq DESC
            LIMIT {topChunks}
        }} AS text_mapping,
        // Entity - Report Mapping filtered by community level
        collect {{
            UNWIND nodes as n
            MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
            WHERE c.rank = {community_level}
            WITH c, c.rank as rank, c.weight AS weight
            RETURN c.summary 
            ORDER BY rank, weight DESC
            LIMIT {topCommunities}
        }} AS report_mapping,
        // Outside Relationships 
        collect {{
            UNWIND nodes as n
            MATCH (n)-[r:RELATED]-(m) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.rank, r.weight DESC 
            LIMIT {topOutsideRels}
        }} as outsideRels,
        // Inside Relationships 
        collect {{
            UNWIND nodes as n
            MATCH (n)-[r:RELATED]-(m) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.rank, r.weight DESC 
            LIMIT {topInsideRels}
        }} as insideRels,
        // Entities description
        collect {{
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        }} as entities
        RETURN 
            "Chunks:" + apoc.text.join(text_mapping, '|') + 
            "\nReports: " + apoc.text.join(report_mapping,'|') +  
            "\nRelationships: " + apoc.text.join(outsideRels + insideRels, '|') + 
            "\nEntities: " + apoc.text.join(entities, "|") AS text, 
            1.0 AS score, 
            coalesce(nodes[0].id, randomUUID()) AS id, 
            apoc.convert.toJson({{
                _node_type: coalesce(nodes[0]._node_type, "Entity"), 
                _node_content: coalesce(nodes[0]._node_content, "")
            }}) AS metadata
        """
        neo4j_vector = Neo4jVectorStore(
            creds["NEO4J_USERNAME"],
            creds["NEO4J_PASSWORD"],
            creds["NEO4J_URI"],
            embed_dim,
            index_name=index_name,
            retrieval_query=retrieval_query,
        )

        embeddings = SafeRateLimitedOpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=GRAPHRAG_API_KEY,
            max_retries=60,
        )
        embeddings.set_limits(get_embedding_token_limiter(), max_chunk_tokens=4000)

        loaded_index = VectorStoreIndex.from_vector_store(neo4j_vector).as_query_engine(
            similarity_top_k=topEntities,
            embed_model=embeddings,
            api_key=GRAPHRAG_API_KEY,
        )
        response = loaded_index.query(query)
        return response.response
    elif type == "global":

        llm = RateLimitedChatOpenAI(
            model="gpt-4.1-nano",
            api_key=GRAPHRAG_API_KEY,
            token_limiter=get_llm_token_limiter(),
            max_completion_tokens=8000,
        )
        MAP_SYSTEM_PROMPT = """
        ---Role---

        You are a helpful assistant responding to questions about data in the tables provided.


        ---Goal---

        Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

        You should use the data provided in the data tables below as the primary context for generating the response.
        If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

        Each key point in the response should have the following element:
        - Description: A comprehensive description of the point.
        - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

        The response should be JSON formatted as follows:
        {{
            "points": [
                {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
                {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
            ]
        }}

        The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

        Points supported by data should list the relevant reports as references as follows:
        "This is an example sentence supported by data references [Data: Reports (report ids)]"

        **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

        For example:
        "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

        where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

        Do not include information where the supporting evidence for it is not provided.


        ---Data tables---

        {context_data}

        ---Goal---

        Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input data tables.

        You should use the data provided in the data tables below as the primary context for generating the response.
        If you don't know the answer or if the input data tables do not contain sufficient information to provide an answer, just say so. Do not make anything up.

        Each key point in the response should have the following element:
        - Description: A comprehensive description of the point.
        - Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

        The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

        Points supported by data should list the relevant reports as references as follows:
        "This is an example sentence supported by data references [Data: Reports (report ids)]"

        **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

        For example:
        "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 64, 46, 34, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

        where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data report in the provided tables.

        Do not include information where the supporting evidence for it is not provided.

        The response should be JSON formatted as follows:
        {{
            "points": [
                {{"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value}},
                {{"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}}
            ]
        }}
        """

        map_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    MAP_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )
        map_chain = map_prompt | llm | StrOutputParser()
        REDUCE_SYSTEM_PROMPT = """
        ---Role---

        You are a helpful assistant responding to questions about a dataset by synthesizing perspectives from multiple analysts.


        ---Goal---

        Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

        Note that the analysts' reports provided below are ranked in the **descending order of importance**.

        If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

        The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

        Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.

        The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

        The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

        **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

        For example:

        "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

        where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

        Do not include information where the supporting evidence for it is not provided.


        ---Target response length and format---

        {response_type}


        ---Analyst Reports---

        {report_data}


        ---Goal---

        Generate a response of the target length and format that responds to the user's question, summarize all the reports from multiple analysts who focused on different parts of the dataset.

        Note that the analysts' reports provided below are ranked in the **descending order of importance**.

        If you don't know the answer or if the provided reports do not contain sufficient information to provide an answer, just say so. Do not make anything up.

        The final response should remove all irrelevant information from the analysts' reports and merge the cleaned information into a comprehensive answer that provides explanations of all the key points and implications appropriate for the response length and format.

        The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

        The response should also preserve all the data references previously included in the analysts' reports, but do not mention the roles of multiple analysts in the analysis process.

        **Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

        For example:

        "Person X is the owner of Company Y and subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also CEO of company X [Data: Reports (1, 3)]"

        where 1, 2, 3, 7, 34, 46, and 64 represent the id (not the index) of the relevant data record.

        Do not include information where the supporting evidence for it is not provided.


        ---Target response length and format---

        {response_type}

        Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
        """

        reduce_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    REDUCE_SYSTEM_PROMPT,
                ),
                (
                    "human",
                    "{question}",
                ),
            ]
        )
        reduce_chain = reduce_prompt | llm | StrOutputParser()
        graph = Neo4jGraph(
            url=creds["NEO4J_URI"],
            username=creds["NEO4J_USERNAME"],
            password=creds["NEO4J_PASSWORD"],
            refresh_schema=False,
        )
        response_type: str = "multiple paragraphs"
        return global_retriever(
            query=query,
            level=2,
            graph=graph,
            map_chain=map_chain,
            reduce_chain=reduce_chain,
            response_type=response_type,
        )


def global_retriever(
    query: str,
    level: int,
    graph,
    map_chain,
    reduce_chain,
    response_type: str = "multiple paragraphs",
) -> str:
    community_data = graph.query(
        """
    MATCH (c:__Community__)
    WHERE c.level = $level
    RETURN c.full_content AS output
    """,
        params={"level": level},
    )
    intermediate_results = []
    for community in tqdm(community_data, desc="Processing communities"):
        intermediate_response = map_chain.invoke(
            {"question": query, "context_data": community["output"]}
        )
        intermediate_results.append(intermediate_response)
    final_response = reduce_chain.invoke(
        {
            "report_data": intermediate_results,
            "question": query,
            "response_type": response_type,
        }
    )
    return final_response


def main():
    # start monitor
    logger = Logger("metrics")
    running_flag = {"running": True}
    monitor_thread = threading.Thread(
        target=monitor_system,
        args=(logger, 1, running_flag),
    )
    monitor_thread.start()
    if False:
        try:
            createGRAG()
        except subprocess.CalledProcessError as e:
            print(f"failed: {e}")
        finally:
            running_flag["running"] = False
            monitor_thread.join()
            print(f"System metrics logged to {logger.get_root_file_path()}")
    else:
        try:
            query_grag("local", "Was hat der Zauberlehrling in diesem Text beschworen?")
        except subprocess.CalledProcessError as e:
            print(f"failed: {e}")
        finally:
            running_flag["running"] = False
            monitor_thread.join()
            print(f"System metrics logged to {logger.get_root_file_path()}")


if __name__ == "__main__":
    main()
