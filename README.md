# 🤖 Agent Orchestration — LLM Agents, Multi-Agent Systems & RAG

> Un projet d'apprentissage et de démonstration construit avec **CrewAI** et **Mistral AI**, couvrant le cycle complet du développement d'agents IA : du *prompt engineering* à l'orchestration multi-agents, jusqu'à un pipeline **RAG** (Retrieval-Augmented Generation) complet sur documents PDF.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF5A5F)
![Mistral AI](https://img.shields.io/badge/Mistral%20AI-LLM-FF7000)
![RAG](https://img.shields.io/badge/RAG-Vector%20Search-4B8BBE)

---

## 🎯 Objectif

Ce dépôt documente ma montée en compétences sur la construction de **systèmes d'agents IA**, organisée en **3 phases progressives**. Chaque phase est autonome et illustre un concept clé du développement d'applications basées sur les LLM.

L'objectif : passer d'un simple appel de modèle à un **système multi-agents autonome** capable de raisonner à partir de documents réels.

---

## 🧱 Architecture du projet

```
Agent-Orchestration/
├── 1.Prompts/            # Phase 1 — Fondamentaux des prompts & tokens
├── 2.Messages/           # Phase 2 — Conversations & collaboration multi-agents
├── 3.VectorDatabases/    # Phase 3 — Embeddings, recherche vectorielle & RAG
├── requirements.txt
└── README.md
```

---

## 📚 Les 3 phases

### Phase 1 — `1.Prompts/` · Fondamentaux du Prompt Engineering
Maîtrise des briques de base d'un agent LLM.

| Fichier | Concept démontré |
|---|---|
| `static_prompt.py` | Prompt fixe, appel single-shot |
| `dynamic_prompt.py` | Prompt paramétré par variables |
| `template.py` | Template interactif réutilisable |
| `tokens.py` | Estimation des tokens et du coût |
| `context_window.py` | Limites de contexte comparées entre modèles |
| `chat_prompt.py` | Agent avec rôle & *backstory* |
| `project.py` | Mini-projet de synthèse |

### Phase 2 — `2.Messages/` · Conversations & Collaboration
Gestion du dialogue, des rôles système et de la **collaboration entre agents**.

| Fichier | Concept démontré |
|---|---|
| `messages_basic.py` | Échange single-turn |
| `messages_chat.py` | Rôles explicites (System / User) |
| `conversation.py` | Conversation multi-tours entre 2 agents (Curieux ↔ Expert) |
| `chat_agent.py` | **Chat interactif** en direct avec un agent (*Orion, le Mentor IA*) |
| `project.py` | **Pipeline d'interview** : 3 agents (Journaliste → Architecte → Rédacteur) orchestrés séquentiellement |

### Phase 3 — `3.VectorDatabases/` · RAG & Recherche Sémantique
Construction d'un pipeline **RAG** complet : des embeddings jusqu'à la génération augmentée.

| Fichier | Concept démontré |
|---|---|
| `embeddings.py` | Génération de vecteurs (HuggingFace / `all-MiniLM-L6-v2`) |
| `vector_store.py` | Création d'une base vectorielle à partir d'un PDF |
| `retrieval.py` | Recherche sémantique (*retrieval*) dans la base |
| `rag_pipeline.py` | Pipeline RAG complet : *Retrieval + Generation* via un agent |
| `final_mini_project.py` | 🎓 **AI Study Partner** — 3 agents (Archiviste → Pédagogue → Quiz Master) génèrent un guide d'étude personnalisé à partir d'un PDF |

---

## 🛠️ Stack technique

- **Orchestration d'agents** : [CrewAI](https://www.crewai.com/) (`Agent`, `Task`, `Crew`, `Process`)
- **LLM** : Mistral AI (`mistral-small-latest`)
- **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace)
- **RAG / Vector Search** : `PDFSearchTool` (CrewAI Tools)
- **Langage** : Python 3.12
- **Config** : `python-dotenv`

---

## 🚀 Installation & utilisation

### 1. Cloner le dépôt
```bash
git clone git@github.com:ahmed2025-Web/Agent-Orchestration.git
cd Agent-Orchestration
```

### 2. Créer l'environnement virtuel
```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

### 3. Configurer la clé API
Créer un fichier `.env` à la racine :
```env
MISTRAL_API_KEY=votre_clé_api_mistral
```

### 4. Lancer un exemple
```bash
# Phase 1 — un prompt simple
python 1.Prompts/static_prompt.py

# Phase 2 — chat interactif avec un agent
python 2.Messages/chat_agent.py

# Phase 3 — le projet final RAG multi-agents
python 3.VectorDatabases/final_mini_project.py
```

---

## 💡 Compétences mises en avant

- ✅ Conception et **orchestration de systèmes multi-agents** (séquentiel, passage de contexte entre tâches)
- ✅ **Prompt engineering** : prompts statiques, dynamiques, rôles système, gestion des tokens & du contexte
- ✅ Implémentation d'un **pipeline RAG** de bout en bout (chunking → embeddings → retrieval → génération)
- ✅ Intégration d'**APIs LLM** et de modèles d'embeddings open-source
- ✅ Code structuré, progressif et documenté

---

## 👤 Auteur

**Ahmed El Amri**
📧 elamriahmed2005@gmail.com
🔗 [github.com/ahmed2025-Web](https://github.com/ahmed2025-Web)
