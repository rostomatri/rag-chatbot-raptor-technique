<img width="338" height="261" alt="Capture d'écran 2025-09-21 131509" src="https://github.com/user-attachments/assets/d9783e88-d76d-481d-8223-beff68d89960" />
<img width="712" height="232" alt="Capture d'écran 2025-09-21 131447" src="https://github.com/user-attachments/assets/0882d362-2db5-4e89-ae7f-480a37baec38" />
<img width="787" height="446" alt="Capture d'écran 2025-09-21 131432" src="https://github.com/user-attachments/assets/722ff000-93cb-4096-a253-ff3f4cbb38e9" />
<img width="427" height="332" alt="Capture d'écran 2025-09-21 131413" src="https://github.com/user-attachments/assets/b53d1c7f-3f5c-4f13-b8dc-5100cd28f7bd" />
# rag-chatbot-raptor-technique
Ce projet implémente un **chatbot intelligent** capable de traiter **texte et tableaux** grâce à une combinaison d'approches avancées :  

- **Chunking sémantique** (texte + tableau).  
- **RAG (Retrieval-Augmented Generation)** avec **RAPTOR** (*Recursive Abstractive Processing for Tree-Organized Retrieval*) pour la récupération d'informations pertinentes.  
- **LangChain** pour l'orchestration complète du pipeline.  
- **LLM Summarization** pour réorganiser, résumer et synthétiser les réponses.  
- **Qdrant** comme base de données vectorielle pour un stockage performant des embeddings.  

---

## 🚀 Fonctionnalités Clés

✅ **Chunking sémantique avancé** : segmentation hiérarchique des documents (texte & tableaux).  
✅ **Pipeline RAG complet** : retrieval + génération de réponse via LLMs.  
✅ **Technique Raptor** : regroupement récursif et pondération des chunks les plus pertinents.  
✅ **Réorganisation & Résumé** : génération de réponses synthétiques et cohérentes.  
✅ **Support tabulaire** : extraction d’insights et description automatique des tables.  
✅ **Qdrant Vector Store** : indexation et recherche rapide des embeddings vectoriels.  

---

## 🖼️ Architecture



1. **Prétraitement & Chunking** – Segmentation sémantique avec RAPTOR.  
2. **Indexation** – Stockage des embeddings dans Qdrant.  
3. **Recherche** – Sélection des chunks les plus pertinents via similarité cosinus.  
4. **Génération** – Utilisation d’un LLM via LangChain.  
5. **Synthèse** – Résumé et réorganisation pour une réponse finale claire.  

---

## 📜 Technologies Utilisées

| Bibliothèque / Version           | Composants / Modèles                          | Utilité                                                | Choix |
|--------------------------------|-----------------------------------------------|------------------------------------------------------|------|
| **LangChain**                  | PromptTemplate, RetrievalQA, Chains           | Orchestration du pipeline, construction des prompts  | Standard RAG |
| **Qdrant**                     | Collections, Vectors                          | Stockage et recherche vectorielle haute performance | Base vectorielle |
| **sentence-transformers (v2.2)** | `thenlper/gte-small`                          | Génération d’embeddings textuels compacts (384-dim) | Rapide et léger |
| **transformers (v4.41)**       | AutoTokenizer, AutoModelForSeq2SeqLM, pipeline | Génération de texte, résumé, traduction (BART/T5)   | SOTA en NLP |
| **scikit-learn (v1.3)**        | cluster, TruncatedSVD, GaussianMixture        | Clustering, réduction de dimension                  | Analyse et structuration |
| **umap-learn (v0.5)**          | UMAP                                          | Réduction dimensionnelle pour visualiser les embeddings | Visualisation intuitive |
| **google-generativeai (v0.3)** + **langchain_google_genai** | ChatGoogleGenerativeAI | Intégration avec Gemini (PaLM/ChatGPT-like)         | Exploite LLMs Google |
| **Docker**                     | Qdrant container                              | Déploiement local ou cloud                           | Portable et simple |

---

## ⚙️ Installation & Exécution

### 1️⃣ Cloner le dépôt

git clone https://github.com/rostomatri/chatbot-rag-raptor-langchain.git
cd chatbot-rag-raptor-langchain
### 2️⃣ Installer les dépendances

pip install -r requirements.txt
### 3️⃣ Lancer Qdrant avec Docker

docker run -p 6333:6333 qdrant/qdrant
### 4️⃣ Démarrer le chatbot

python src/raptor.py
