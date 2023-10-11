import sys; sys.path.append('C:/CReSE/CReSE')
from model import CLIPModel, EcEncoder, ProjectionHead
from training import set_CReSE_and_tokenizer

from bertopic import BERTopic
from get_bertopic_from_CReSE import CAS, CustomEmbedder



def load_bertopic_with_embeddings(cluster_n, 
                                  bertopic_seed, 
                                  withoutCREEP=False,
                                  add_emb_model=True):   
    dir_bertopic_CREEP = \
        "G:/내 드라이브/[1] CCADD N CBDL/[1] Personal Research/2022_MSR_drug_repositioning/[2] Code/EC_title_recommendation/0629/bertopic_from_CREEP"
    #set dir_bertopic
    if withoutCREEP:
        dir_bertopic = \
            f"bertopic_save/bertopic_model_cluster{cluster_n}_withoutCREEP" 
    else:
        dir_bertopic = \
            f"bertopic_save/bertopic_model_cluster{cluster_n}_seed{bertopic_seed}" 
            
    #load topic_model
    if withoutCREEP:
        topic_model = BERTopic.load(f"{dir_bertopic_CREEP}/{dir_bertopic}")
    else:
        if add_emb_model:
            model, _ = set_CReSE_and_tokenizer(CAS.embedding_model_name)
            topic_model = BERTopic.load(f"{dir_bertopic_CREEP}/{dir_bertopic}",
                                        embedding_model=model)
        else:
            topic_model = BERTopic.load(f"{dir_bertopic_CREEP}/{dir_bertopic}")
            
    return topic_model