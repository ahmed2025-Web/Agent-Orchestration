"""
Phase 3: Embeddings (Génération de vecteurs)

"""

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    
    print("=" * 60)
    print("🧠 GÉNÉRATION D'EMBEDDINGS (VECTEURS)")
    print("=" * 60)

    # 1. Chargement du modèle d'embedding gratuit
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2. Le texte que l'on veut transformer
    texte_exemple = "Un cogniton est une notion abstraite."
    print(f"Texte à encoder : '{texte_exemple}'")

    # 3. Génération du vecteur
    vecteur = embeddings_model.embed_query(texte_exemple)

    # 4. Résultat
    print(f"Dimension du vecteur généré : {len(vecteur)} nombres")
    print(f"Aperçu des 5 premières valeurs du vecteur : {vecteur[:5]}")
    print("=" * 60)

except ImportError:
    print("Error")
