import argparse
import os
import re
import uuid
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sklearn.mixture import GaussianMixture
from umap import UMAP
from sklearn.decomposition import TruncatedSVD
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except ImportError:
    print("⚠️ Librairie Transformers non installée")
    pipeline = None
    AutoTokenizer = None
    AutoModelForSeq2SeqLM = None
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    print("⚠️ Librairie Google Generative AI non installée")
    genai = None
    ChatGoogleGenerativeAI = None

load_dotenv()

# Constants
DOCUMENT_PATHS = [ "D:\\RAPTOR\\PDF_Markdown\\To_share\\txt.txt"]
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "chunks"
RANDOM_SEED = 224



class DocumentProcessor:
    def __init__(self, google_api_key: Optional[str] = None):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.qdrant_client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, timeout=60.0
        )
        # Initialisation du modèle pour le summarization
        self.bart_tokenizer = None
        self.bart_model = None
   

        try:
            from transformers import BartTokenizer, BartForConditionalGeneration
            print("✅ Chargement du tokenizer BART...")
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
            print("✅ Chargement du modèle BART...")
            self.bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
            print("✅ Modèle BART-large-cnn chargé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur de chargement BART: {e}")
            print("Le summarization avec BART sera désactivé")

        
        # Initialiser Gemini pour la génération (question-réponse)
        self.generation_model = None
        if google_api_key and ChatGoogleGenerativeAI:
            try:
                genai.configure(api_key=google_api_key)
                self.generation_model = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0,
                    google_api_key=google_api_key,
                )
                print("✅ Modèle Gemini 2.0 Flash initialisé pour la génération")
            except Exception as e:
                print(f"⚠️ Erreur d'initialisation Gemini: {e}")
                print("La génération avec Gemini sera désactivée")
        else:
            print("⚠️ Clé Google non fournie - génération Gemini désactivée")

    def read_document(self, file_path: str) -> str:
        """Read the content of the text document"""
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()

    def split_into_pages(self, document_content: str) -> List[str]:
        """Split document into pages using the {number}---... separator"""
        pages = []
        current_page = []

        lines = document_content.split("\n")

        for line in lines:
            # Check if line matches the page separator pattern {i}----...
            if re.match(r"^\{\d+\}[-]+", line.strip()):
                if current_page:  # If we have content in current page, save it
                    pages.append("\n".join(current_page))
                    current_page = []
            else:
                if line.strip():  # Only add non-empty lines
                    current_page.append(line)

        # Add the last page if it has content
        if current_page:
            pages.append("\n".join(current_page))

        return pages

  

    def initialize_qdrant_collection(self):
        """Initialize or reset the Qdrant collection"""
        if self.qdrant_client.collection_exists(COLLECTION_NAME):
            self.qdrant_client.delete_collection(COLLECTION_NAME)

        self.qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        text_embeddings = self.embedding_model.embed_documents(texts)
        return np.array(text_embeddings)

    def embed_cluster_texts(self, texts: List[str]) -> pd.DataFrame:
        """Embed and cluster texts"""
        print(f"Embedding {len(texts)} texts...")
        embeddings = self.embed_texts(texts)
        print("Embedding completed, starting clustering...")
    
        cluster_labels = self.perform_clustering(embeddings, dim=10, threshold=0.1)
    
        # DEBUG: Vérifier les labels
        print(f"Cluster labels types: {type(cluster_labels)}")
        if cluster_labels:
            print(f"First few cluster labels: {cluster_labels[:5]}")
    
        df = pd.DataFrame({
            "text": texts,
            "embd": list(embeddings),
            "cluster": cluster_labels,
        })
    
        print(f"DataFrame created with {len(df)} rows")
        return df

    def global_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform global dimensionality reduction on the embeddings using UMAP.
        """
        n_samples = len(embeddings)
    
        # Pour les très petits datasets
        if n_samples <= 5:
            if n_samples <= dim:
                return embeddings[:, :dim]
            else:
                return TruncatedSVD(n_components=dim).fit_transform(embeddings)
    
        if n_neighbors is None:
            n_neighbors = int((n_samples - 1) ** 0.5)
    
        if n_samples <= n_neighbors:
            n_neighbors = max(2, n_samples - 1)
        return UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def local_cluster_embeddings(
        self,
        embeddings: np.ndarray,
        dim: int,
        num_neighbors: int = 10,
        metric: str = "cosine",
    ) -> np.ndarray:
        """
        Perform local dimensionality reduction on the embeddings using UMAP.
        """
        n_samples = len(embeddings)
        # Pour les très petits datasets, retourner les embeddings originaux ou une version réduite simple
        if n_samples <= 5:
            if n_samples <= dim:
                return embeddings[:, :dim]
            else:
                return TruncatedSVD(n_components=dim).fit_transform(embeddings)
    
        # Ajuster le nombre de voisins si nécessaire
        if n_samples <= num_neighbors:
            num_neighbors = max(2, n_samples - 1)
        return UMAP(
            n_neighbors=num_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def get_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_clusters: int = 50,
        random_state: int = RANDOM_SEED,
     ) -> int:
        """Determine optimal number of clusters using BIC"""
        if len(embeddings) <= 1:
            return 1
        max_clusters = min(max_clusters, len(embeddings))
        n_clusters = np.arange(1, max_clusters + 1)

        if len(n_clusters) == 0:
            return 1
    
        bics = []
        for n in n_clusters:
            try:
                gm = GaussianMixture(n_components=n, random_state=random_state)
                gm.fit(embeddings)
                bics.append(gm.bic(embeddings))
            except:
                bics.append(float('inf'))
    
        if not bics or all(b == float('inf') for b in bics):
            return 1
        
        return n_clusters[np.argmin(bics)]

    def GMM_cluster(
        self, embeddings: np.ndarray, threshold: float, random_state: int = 0
    ) -> Tuple[List[int], int]:
        """Cluster embeddings using Gaussian Mixture Model - RETOURNE DES INT"""
        if len(embeddings) == 0:
            return [], 0
        
        max_clusters = min(10, len(embeddings) - 1)
        if max_clusters < 1:
            return [], 0
    
        n_clusters = self.get_optimal_clusters(embeddings, max_clusters=max_clusters)
        if n_clusters < 1:
            return [], 0
        
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
        gm.fit(embeddings)
        probs = gm.predict_proba(embeddings)
    
        # Assigner au cluster avec la plus haute probabilité
        labels = []
        for prob in probs:
            max_prob_idx = np.argmax(prob)
            if prob[max_prob_idx] > threshold:
                labels.append(max_prob_idx)  # ← RETOURNE INT
            else:
                labels.append(-1)  # ← RETOURNE INT
    
        return labels, n_clusters

    def perform_clustering(
        self, embeddings: np.ndarray, dim: int, threshold: float
    ) -> List[str]:
        """
        Perform clustering on the embeddings.
        """
        if len(embeddings) <= dim + 1:
            return ["0:0"] * len(embeddings)

        # Global dimensionality reduction
        reduced_embeddings_global = self.global_cluster_embeddings(embeddings, dim)
        global_labels, n_global_clusters = self.GMM_cluster(reduced_embeddings_global, threshold)
        print(f"Clusters globaux trouvés: {n_global_clusters}")
    
        all_local_clusters = ["unassigned"] * len(embeddings)

        # Iterate through each global cluster
        for global_id in range(n_global_clusters):
            # Extract embeddings belonging to the current global cluster
            mask = np.array([gl == global_id for gl in global_labels])
            global_cluster_embeddings_ = embeddings[mask]

            if len(global_cluster_embeddings_) == 0:
                continue

            if len(global_cluster_embeddings_) <= dim + 1:
                # Pour les petits clusters, assigner séquentiellement
                for i, idx in enumerate(np.where(mask)[0]):
                    all_local_clusters[idx] = f"{global_id}:{i}"
                continue

            # Local dimensionality reduction and clustering
            reduced_embeddings_local = self.local_cluster_embeddings(global_cluster_embeddings_, dim)
            local_labels, n_local_clusters = self.GMM_cluster(reduced_embeddings_local, threshold)

            # Assign local cluster IDs
            for local_id in range(n_local_clusters):
                local_mask = np.array([ll == local_id for ll in local_labels])
                global_indices = np.where(mask)[0][local_mask]

                for idx in global_indices:
                    all_local_clusters[idx] = f"{global_id}:{local_id}"

            # Gérer les points non assignés localement
            unassigned_mask = np.array([ll == -1 for ll in local_labels])
            unassigned_indices = np.where(mask)[0][unassigned_mask]
            for idx in unassigned_indices:
                all_local_clusters[idx] = f"{global_id}:unassigned"

        # Fallback: si tout est unassigned, utiliser un clustering simple
        if all(cluster == "unassigned" for cluster in all_local_clusters):
            print("⚠️ All points unassigned, using fallback clustering")
            from sklearn.cluster import KMeans
            n_clusters = min(5, len(embeddings))
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED)
            simple_labels = kmeans.fit_predict(embeddings)
            return [f"0:{label}" for label in simple_labels]
    
        return all_local_clusters

    def fmt_txt(self, df_cluster: pd.DataFrame) -> str:
        """
        Format the text from a cluster for summarization
        """
        return "\n\n".join(df_cluster["text"].tolist())

    def summarize_with_bart(self, text: str, max_length: int = 150, min_length: int = 30) -> str:
     """
     Fonction de summarization utilisant BART-large-cnn avec tokenization correcte
     """
     if not self.bart_tokenizer or not self.bart_model:
        return "Summarization désactivé - modèle BART non disponible"
    
     try:
        # Tokenization correcte avec le tokenizer BART
        inputs = self.bart_tokenizer(
            text, 
            max_length=1024,  # Limite de BART
            truncation=True, 
            return_tensors="pt"
        )
        
        # Génération du résumé avec paramètres optimisés
        summary_ids = self.bart_model.generate(
            inputs.input_ids,
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        
        # Détokenization
        summary = self.bart_tokenizer.decode(
            summary_ids[0], 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        
        return summary
        
     except Exception as e:
        print(f"❌ Erreur lors de la summarization BART: {e}")
        return f"Erreur de summarization: {str(e)}"


    def embed_cluster_summarize_texts(
        self, texts: List[str], level: int
     ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Embeds, clusters, and summarizes a list of texts.
        """
        # Embed and cluster the texts
        df_clusters = self.embed_cluster_texts(texts)

        # Préparer les clusters uniques pour le summarization
        all_clusters = df_clusters["cluster"].unique()
        print(f"--Generated {len(all_clusters)} clusters--")
        #print(f"--Limiting to {max_summaries} summaries maximum--")

        # Summarization avec limite et pauses
        summaries = []
        clusters_processed = 0
        processed_cluster_ids = []  # Pour garder une trace des clusters traités
        
        for cluster_id in all_clusters:
      
                
            if cluster_id == "unassigned":
                continue
            
            df_cluster = df_clusters[df_clusters["cluster"] == cluster_id]
            formatted_txt = self.fmt_txt(df_cluster)
            
            text_length = len(formatted_txt.split())
            if self.bart_tokenizer:
             tokens = self.bart_tokenizer(formatted_txt, return_tensors="pt", truncation=False)
             text_length = tokens.input_ids.shape[1]
            else:
             # Fallback simple si BART n'est pas disponible
             text_length = len(formatted_txt.split())

            # Ajustement dynamique des paramètres de summarization
            if text_length < 100:
             max_len = 50
             min_len = 20
            elif text_length < 300:
             max_len = 100
             min_len = 40
            elif text_length < 600:
             max_len = 150
             min_len = 60
            else:
             max_len = 200
             min_len = 80

            print(f"📝 Processing cluster {cluster_id} ({clusters_processed + 1}) - {text_length} tokens")
            summary = self.summarize_with_bart(formatted_txt, max_length=max_len, min_length=min_len)
            summaries.append(summary)
            processed_cluster_ids.append(cluster_id)
            clusters_processed += 1

           # CORRECTION: S'assurer que tous les tableaux ont la même longueur
           # Utiliser uniquement les clusters qui ont été effectivement traités
        df_summary = pd.DataFrame({
              "summaries": summaries,
              "level": [level] * len(summaries),
              "cluster": processed_cluster_ids,  # Utiliser les IDs des clusters traités
          })

        return df_clusters, df_summary


    def recursive_embed_cluster_summarize(
        self, 
        texts: List[str], 
        level: int = 1, 
        n_levels: int = 3,
        #max_summaries_per_level: int = 5
        ) -> Dict[int, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Implémentation récursive pour construire la pyramide de résumés
        """
        results = {}
    
        # Traitement du niveau actuel
        df_clusters, df_summary = self.embed_cluster_summarize_texts(
        texts, level
        )
        results[level] = (df_clusters, df_summary)
    
        # Condition d'arrêt récursive
        unique_clusters = df_summary['cluster'].nunique() if not df_summary.empty else 0
    
        if level < n_levels and unique_clusters > 1:
         # Récursion sur les résumés
            next_level_texts = df_summary['summaries'].tolist()
            next_results = self.recursive_embed_cluster_summarize(
            next_level_texts, level + 1, n_levels
            )
            results.update(next_results)
    
        return results
    



    def create_chunks_metadata(self, pages: List[str], source_doc: str) -> List[Dict]:
        """
        Préparer les métadonnées des chunks avec information sur le document source
        """
        return [
            {
                "text": page,
                "page_num": i + 1,
                "source_document": source_doc,  # Nouveau: document source
                "document_name": os.path.basename(source_doc),  # Nom du fichier
            }
            for i, page in enumerate(pages)
        ]

    def upload_chunks_to_qdrant(
        self, df_clusters: pd.DataFrame, chunks_metadata: List[Dict]
    ) -> np.ndarray:
        """Upload chunks to Qdrant with clustering information"""
        texts = [chunk["text"] for chunk in chunks_metadata]
        embeddings = self.embed_texts(texts)

        # Perform clustering
        cluster_labels = df_clusters["cluster"].tolist()

        # Ajout de l'affichage des clusters ici
        print("Cluster labels (structure réelle) :")
        for idx, label in enumerate(cluster_labels):
            print(f"Chunk {idx}: {label}")

        points = []
        for idx, chunk in enumerate(chunks_metadata):
            payload = {
                "text": chunk["text"],
                "page_number": chunk["page_num"],
                "cluster": cluster_labels[idx],
                "source_document": chunk["source_document"],  
                "document_name": chunk["document_name"], 
            }

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embeddings[idx].tolist(),
                    payload=payload,
                )
            )

        # Insert into Qdrant
        self.qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ {len(points)} chunks inserted into Qdrant")

        return embeddings

    def build_summary_tree(self, texts: List[str], n_levels: int = 3) -> Dict[str, Any]:
        """
        Construit l'arbre hiérarchique de résumés
        """
        print("🌳 Building summary tree...")
        
        # Exécuter le processus récursif de clustering et summarization
        results = self.recursive_embed_cluster_summarize(
            texts, level=1, n_levels=n_levels
        )
        
        # Construire l'arbre avec tous les textes
        all_texts = texts.copy()  # Textes originaux (feuilles)
        
        # Ajouter les résumés de chaque niveau
        for level in sorted(results.keys()):
            summaries = results[level][1]['summaries'].tolist()
            all_texts.extend(summaries)
        
        print(f"✅ Summary tree built with {len(all_texts)} nodes")
        return {
            "results": results,
            "all_texts": all_texts,
            "leaf_texts": texts
        }

    '''def create_rag_chain(self, source_filter: Optional[str] = None):
        """
        Crée la chaîne RAG pour la question-réponse
        """
        if not self.model:
            print("⚠️ Modèle non disponible - RAG chain désactivée")
            return None
            
        # Définir le prompt RAG
        rag_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
        Question: {question} 
        Context: {context} 
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(rag_template)
        
        # Définir le retriever
        def retriever(question: str, top_k: int = 5):
            # Embed the question
            question_embedding = self.embedding_model.embed_query(question)
            query_filter = None
            if source_filter:
                    query_filter = {
                        "must": [{
                            "key": "source_document",
                            "match": {"value": source_filter}
                        }]
                    }
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=question_embedding,
                query_filter=query_filter,
                limit=top_k
            )
            
            # Extract the text from results
            return [hit.payload["text"] for hit in search_result]
        
    
        # Formatter les documents
        def format_docs(docs):
            return "\n\n".join(docs)
        
        # Créer la chaîne RAG
        rag_chain = (
            {"context": lambda x: format_docs(retriever(x["question"])), "question": lambda x: x["question"]}
            | prompt
            | self.model
            | StrOutputParser()
        )
        
        return rag_chain'''
    #############
    def hierarchical_retriever(self, question: str, summary_tree: Dict[str, Any], 
                              top_k_per_level: List[int] = [1, 3, 5]) -> List[str]:
         """
         NOUVEAU: Retriever hiérarchique qui navigue à travers les niveaux de l'arbre.
        
         Fonctionnement:
         1. Embed la question pour avoir un vecteur de recherche
         2. Commence au niveau 3 (le plus abstrait): trouve les clusters de résumés les plus pertinents
         3. Descend au niveau 2: trouve les sous-clusters pertinents dans les clusters sélectionnés du niveau 3
         4. Arrive au niveau 1: sélectionne les chunks les plus pertinents dans les clusters finaux
         5. Retourne les chunks les plus pertinents pour la réponse du niv 
        
         Args:
            question: La question de l'utilisateur
            summary_tree: L'arbre de résumés construit avec build_summary_tree()
            top_k_per_level: Nombre de clusters à sélectionner à chaque niveau [niveau3, niveau2, niveau1]
        
         Returns:
            Liste des chunks textuels les plus pertinents pour répondre à la question
         """
         question_embedding = self.embedding_model.embed_query(question)
         selected_cluster_paths = set()
        
         print(f"🔍 Début de la recherche hiérarchique pour: '{question}'")
        
         # 1. RECHERCHE AU NIVEAU 3 (le plus abstrait)
         if 3 in summary_tree["results"]:
            df_summary_l3 = summary_tree["results"][3][1]
            l3_summaries = df_summary_l3["summaries"].tolist()
            
            if l3_summaries:
                l3_embeddings = self.embed_texts(l3_summaries)
                
                # Trouver les clusters les plus pertinents au niveau 3 par similarité cosinus
                similarities = np.dot(l3_embeddings, question_embedding)
                top_indices = np.argsort(similarities)[-top_k_per_level[0]:]
                
                print(f"📊 Niveau 3: {len(l3_summaries)} clusters, sélection de {len(top_indices)}")
                
                for idx in top_indices:
                    cluster_id = df_summary_l3.iloc[idx]["cluster"]
                    selected_cluster_paths.add(f"{cluster_id}")  # Format: "2" (niveau 3)
                    print(f"   → Cluster sélectionné: {cluster_id}")
        
        # 2. PROPAGATION AU NIVEAU 2 (sous-clusters)
         if 2 in summary_tree["results"] and selected_cluster_paths:
            df_summary_l2 = summary_tree["results"][2][1]
            l2_summaries = df_summary_l2["summaries"].tolist()
            l2_embeddings = self.embed_texts(l2_summaries)
            
            new_selected_paths = set()
            
            for parent_path in selected_cluster_paths:
                parent_cluster = parent_path  # "2" (niveau 3)
                
                # Trouver les clusters de niveau 2 qui appartiennent à ce parent
                # Format: "2:1", "2:2", etc.
                mask = df_summary_l2["cluster"] == parent_cluster
                relevant_l2_clusters = df_summary_l2[mask]
                
                if len(relevant_l2_clusters) > 0:
                    l2_subset_embeddings = l2_embeddings[mask]
                    similarities = np.dot(l2_subset_embeddings, question_embedding)
                    top_indices = np.argsort(similarities)[-top_k_per_level[1]:]
                    
                    for idx in top_indices:
                        cluster_id = relevant_l2_clusters.iloc[idx]["cluster"]
                        new_selected_paths.add(cluster_id)
                        print(f"   → Niveau 2 - Cluster sélectionné: {cluster_id}")
            
            selected_cluster_paths = new_selected_paths
        
        # 3. PROPAGATION AU NIVEAU 1 (chunks originaux)
         final_chunks = []
         if 1 in summary_tree["results"] and selected_cluster_paths:
            df_clusters_l1 = summary_tree["results"][1][0]
            
            print(f"📊 Niveau 1: Recherche dans {len(selected_cluster_paths)} chemins de cluster")
            
            for cluster_path in selected_cluster_paths:
                # Trouver tous les chunks qui appartiennent à ce chemin de cluster final
                mask = df_clusters_l1["cluster"] == cluster_path
                chunks_in_cluster = df_clusters_l1[mask]["text"].tolist()
                
                if chunks_in_cluster:
                    # Embedder et classer les chunks par pertinence
                    chunk_embeddings = self.embed_texts(chunks_in_cluster)
                    similarities = np.dot(chunk_embeddings, question_embedding)
                    top_indices = np.argsort(similarities)[-top_k_per_level[2]:]
                    
                    for idx in top_indices:
                        final_chunks.append(chunks_in_cluster[idx])
                    print(f"   → Cluster {cluster_path}: {len(top_indices)} chunks sélectionnés")
        
         print(f"✅ Recherche hiérarchique terminée: {len(final_chunks)} chunks trouvés")
         return final_chunks
    
    def format_hierarchical_docs(self, docs: List[str]) -> str:
        """
        NOUVEAU: Formate les documents pour le contexte RAG.
        Remplace l'ancien format_docs() avec une mise en forme améliorée.
        
        Args:
            docs: Liste des chunks textuels à formater
            
        Returns:
            String formaté avec les documents contextuels
        """
        if not docs:
            return "Aucun contexte trouvé pour cette question."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            # Limite la longueur et ajoute un séparateur clair
            truncated_doc = doc[:1000] + "..." if len(doc) > 1000 else doc
            formatted_docs.append(f"[Document {i}]:\n{truncated_doc}\n")
        
        return "\n" + "="*50 + "\n".join(formatted_docs) + "="*50
    
    def create_hierarchical_rag_chain(self, summary_tree: Dict[str, Any]) -> Optional[Callable]:
        """
        NOUVEAU: Crée une chaîne RAG complète utilisant la navigation hiérarchique.
        Remplace complètement l'ancien create_rag_chain().
        
        Cette fonction:
        1. Combine le hierarchical_retriever pour trouver le contexte
        2. Utilise format_hierarchical_docs pour formater le contexte
        3. Crée un prompt optimisé pour les réponses hiérarchiques
        4. Construit la chaîne LangChain complète
        
        Args:
            summary_tree: L'arbre de résumés construit précédemment
            
        Returns:
            Une fonction LangChain ready-to-use ou None si le modèle n'est pas disponible
        """
        if not self.generation_model:
            print("⚠️ Modèle non disponible - RAG chain désactivée")
            return None
            
        # Prompt optimisé pour la structure hiérarchique
        hierarchical_template = """Vous êtes un assistant expert. Utilisez EXCLUSIVEMENT les informations contextuelles suivantes pour répondre à la question.

CONTEXTE:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Répondez de manière précise et concise
2. Basez-vous uniquement sur le contexte fourni
3. Structurez votre réponse si plusieurs aspects sont couverts
4. Si le contexte ne contient pas la réponse, dites-le clairement

RÉPONSE:"""
        
        prompt = ChatPromptTemplate.from_template(hierarchical_template)
        
        def hierarchical_retriever_wrapper(input_dict: Dict[str, str]) -> Dict[str, str]:
            """
            Wrapper qui connecte la question au hierarchical_retriever
            """
            question = input_dict["question"]
            relevant_chunks = self.hierarchical_retriever(
                question, 
                summary_tree,
                top_k_per_level=[1, 2, 3]  # Paramètres ajustables
            )
            
            formatted_context = self.format_hierarchical_docs(relevant_chunks)
            return {"context": formatted_context, "question": question}
        
        # Construction de la chaîne LangChain complète
        hierarchical_rag_chain = (
            hierarchical_retriever_wrapper
            | prompt
            | self.generation_model
            | StrOutputParser()
        )
        
        return hierarchical_rag_chain
    ########
    
    def process_documents(self, file_paths: List[str], n_levels: int = 3):
        """
        Traite plusieurs documents et les combine dans la base vectorielle
        
        Args:
            file_paths (List[str]): Liste des chemins vers les documents
            n_levels (int): Nombre de niveaux hiérarchiques
            
        """
        all_chunks_metadata = []
        all_texts = []
        
        # Traiter chaque document
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️ Fichier non trouvé: {file_path}")
                continue
                
            print(f"📖 Traitement du document: {os.path.basename(file_path)}")
            
            # Lire et diviser le document
            content = self.read_document(file_path)
            pages = self.split_into_pages(content)
            
            # Préparer les métadonnées avec information sur le document source
            chunks_metadata = self.create_chunks_metadata(pages, source_doc=file_path)
            texts = [chunk["text"] for chunk in chunks_metadata]
            
            all_chunks_metadata.extend(chunks_metadata)
            all_texts.extend(texts)
            
            print(f"✅ {len(pages)} pages extraites de {os.path.basename(file_path)}")
        
        if not all_texts:
            raise ValueError("Aucun document valide à traiter")
        
        print(f"📊 Total: {len(all_texts)} chunks across {len(file_paths)} documents")
        
        # Construire l'arbre de résumés sur tous les documents combinés
        summary_tree = self.build_summary_tree(
            all_texts, n_levels=n_levels
        )

        # Initialiser Qdrant
        self.initialize_qdrant_collection()

        # Uploader tous les chunks vers Qdrant
        if 1 in summary_tree["results"]:
            df_clusters_level1 = summary_tree["results"][1][0]
            self.upload_chunks_to_qdrant(df_clusters_level1, all_chunks_metadata)
        else:
            print("⚠️ Aucun résultat trouvé pour le niveau 1")
            df_clusters_fallback = self.embed_cluster_texts(all_texts)
            self.upload_chunks_to_qdrant(df_clusters_fallback, all_chunks_metadata)

        # Créer la chaîne RAG
        rag_chain = self.create_hierarchical_rag_chain(summary_tree)

        return {
            "summary_tree": summary_tree,
            "chunks_metadata": all_chunks_metadata,
            "rag_chain": rag_chain
        }

    def query_document(self, question: str, rag_chain: Any = None) -> str:
        """
        Pose une question en utilisant la chaîne RAG hiérarchique
        """
        if rag_chain is None:
            return "RAG chain non disponible"
        
        try:
            response = rag_chain.invoke({"question": question})
            return response
        except Exception as e:
            return f"Erreur lors de la requête: {str(e)}"

    def interactive_query_mode(self, rag_chain: Any):
      """Mode interactif pour poser des questions"""
      print("\n🔍 Mode interactif activé. Tapez 'quit' pour quitter.")
      while True:
        question = input("\n🤖 Posez votre question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if rag_chain:
            answer = self.query_document(question, rag_chain)
            print(f"\n📝 Réponse:\n{answer}")
        else:
            print("❌ Chaîne RAG non disponible")
if __name__ == "__main__":
    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(description="Système RAG hiérarchique")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Mode interactif"
    )
    parser.add_argument(
        "--levels",
        "-l",
        type=int,
        default=3,
        help="Nombre de niveaux hiérarchiques (1-3)",
    )
    parser.add_argument(
        "--question", "-q", type=str, help="Question spécifique à poser"
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Charger des fichiers existants (utilise les chemins définis dans le code)",
    )

    args = parser.parse_args()

    google_api_key = os.environ.get("GOOGLE_API_KEY")
    processor = DocumentProcessor(google_api_key=google_api_key)

    # Options pour charger des fichiers existants
    LOAD_EXISTING = False  # ← Conservé comme False par défaut
    TREE_FILE_PATH = (
        "summary_tree_20241215_143022.pkl"  # Chemin vers votre fichier d'arbre
    )
    METADATA_FILE_PATH = "chunks_metadata_20241215_143022.json"  # Chemin vers votre fichier de métadonnées

    try:
        if args.load_existing and TREE_FILE_PATH and METADATA_FILE_PATH:
            # Charger les fichiers existants
            print("🔄 Chargement des fichiers existants...")
            summary_tree = processor.load_summary_tree(TREE_FILE_PATH)
            chunks_metadata = processor.load_chunks_metadata(METADATA_FILE_PATH)

            if summary_tree and chunks_metadata:
                print("✅ Fichiers chargés avec succès")
                rag_chain = processor.create_hierarchical_rag_chain(summary_tree)
            else:
                print("❌ Échec du chargement, traitement des documents...")
                results = processor.process_documents(
                    DOCUMENT_PATHS, n_levels=args.levels
                )
                summary_tree = results["summary_tree"]
                chunks_metadata = results["chunks_metadata"]
                rag_chain = results["rag_chain"]
        else:
            # Traitement normal avec le nombre de niveaux spécifié
            results = processor.process_documents(DOCUMENT_PATHS, n_levels=args.levels)
            summary_tree = results["summary_tree"]
            chunks_metadata = results["chunks_metadata"]
            rag_chain = results["rag_chain"]
            tree_file = results.get("tree_file")
            metadata_file = results.get("metadata_file")

        # Afficher les résultats
        print(
            f"\n✅ Processing complete with {len(chunks_metadata)} chunks from {len(DOCUMENT_PATHS)} documents"
        )
        print(
            f"🌳 Summary tree has {len(summary_tree['all_nodes'])} nodes with relations"
        )
        print(f"📊 Niveaux hiérarchiques: {args.levels}")

        # Gestion des différents modes d'interaction
        if args.question and rag_chain:
            # Mode question unique
            print(f"\n🤖 Question: {args.question}")
            answer = processor.query_document(args.question, rag_chain=rag_chain)
            print(f"📝 Réponse:\n{answer}")

        elif args.interactive and rag_chain:
            # Mode interactif
            processor.interactive_query_mode(rag_chain)

        else:
            # Mode par défaut - seulement afficher les infos, pas de question test
            print(
                f"\n🤖 Système RAG prêt. Utilisez --question ou --interactive pour poser des questions."
            )

        # Option: Sauvegarde manuelle supplémentaire
        SAVE_MANUALLY = False
        if SAVE_MANUALLY:
            manual_tree_file = processor.save_summary_tree(
                summary_tree, "manual_save_tree.pkl"
            )
            manual_metadata_file = processor.save_chunks_metadata(
                chunks_metadata, "manual_save_metadata.json"
            )
            print(f"💾 Sauvegarde manuelle: {manual_tree_file}, {manual_metadata_file}")

    except Exception as e:
        print(f"❌ Erreur critique: {e}")
        import traceback

        traceback.print_exc()
