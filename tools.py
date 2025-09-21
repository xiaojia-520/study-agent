from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import config
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)


def creat_collection(collection_name):
    client.recreate_collection(
        collection_name="test_collection",
        vectors_config=VectorParams(size=4, distance=Distance.COSINE)  # 向量维度=4
    )
    return ()


def search_collection(collection_name, q):
    embedding_model = HuggingFaceEmbeddings(
        model_name=r"C:\Users\xiaojia\Desktop\study-agent-master\data\models\embedding\bge-small-zh-v1.5",  # 直接使用字符串
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    flt = Filter(
        must=[FieldCondition(key="session_id", match=MatchValue(value="高等数学"))]
    )

    q = embedding_model.embed_query(q)
    search_result = client.query_points(
        collection_name=collection_name,
        query=q,  # 向量本体
        using="text",  # 指定命名向量的名字
        limit=5,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )

    for p in search_result.points:
        print(p.payload.get("text"))


def delete_collection(collection_name):
    return client.delete_collection(collection_name)

search_collection("asr","a")


