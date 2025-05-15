import os
import json
import logging
from typing import Tuple
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define French intent categories
intent_categories_fr = {
    "match_schedule": "L'utilisateur veut connaître la date d'un match à venir (par exemple, 'Quand est le prochain match du WAC ?').",
    "match_score": "L'utilisateur cherche les résultats ou le score d'un match déjà joué, potentiellement en mentionnant une année (par exemple, 'Résultat match Wydad hier', 'match Raja 2017').",
    "latest_news": "L'utilisateur cherche les actualités récentes ou les dernières informations générales (par exemple, 'Quoi de neuf ?', 'Dernières actualités Arryadia').",
    "program_information": "L'utilisateur demande des informations sur des programmes TV, des émissions, ou du contenu vidéo spécifique (par exemple, 'Montre-moi les documentaires', 'Quand est le journal télévisé ?').",
    "generic_search": "L'intention de l'utilisateur n'est pas claire, est trop générale, ou ne correspond à aucune autre catégorie spécifique."
}

def classify_intent(query: str) -> Tuple[str, float, str]:
    """
    Classifie la requête de l'utilisateur en utilisant le LLM de Groq pour déterminer l'intention de recherche.
    
    Args:
        query (str): La requête de l'utilisateur
        
    Returns:
        Tuple[str, float, str]: Un tuple contenant:
            - intent (str): L'intention classifiée
            - confidence (float): Score de confiance
            - description (str): Description de l'intention
    """
    try:
        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        
        # Initialize Groq client with default base URL
        client = Groq(api_key=api_key)
        
        # Construct system prompt in French
        system_prompt = f"""Tu es un classificateur d'intention expert pour SNRT (Société Nationale de Radiodiffusion et de Télévision du Maroc). 
Ta tâche est d'analyser la requête de l'utilisateur et de la classer dans l'une des catégories prédéfinies.

Tu dois renvoyer ta réponse sous forme d'objet JSON avec trois clés :
- "intent" : l'identifiant de la catégorie (doit être l'une des clés suivantes: {', '.join(intent_categories_fr.keys())})
- "confidence" : un nombre décimal entre 0.0 et 1.0
- "reasoning" : une brève explication en français pour ta classification

Voici les intentions disponibles :
"""
        for key, desc in intent_categories_fr.items():
            system_prompt += f"- {key}: {desc}\n"
        
        system_prompt += """
Instructions spécifiques pour la classification :

1. Détection temporelle et sportive :
   - Si la requête contient une année (ex: 2017, 2022) ou une référence temporelle passée (hier, la semaine dernière) ET des termes liés au sport (match, score, Wydad, Raja, football), l'intention est très probablement "match_score". Attribue une confiance élevée (0.8-1.0) dans ce cas.
   - Si la requête contient une référence temporelle future (demain, la semaine prochaine) ET des termes liés au sport, l'intention est probablement "match_schedule".

2. Termes sportifs courants :
   - Les mots comme "match", "score", "résultat", "Wydad", "Raja", "WAC", "FAR" sont des indicateurs forts d'intentions liées au sport.
   - La présence d'années ou de dates est un indicateur fort pour "match_score".

3. Général :
   - Si la requête est ambiguë ou ne correspond clairement à aucune catégorie, utilise "generic_search" avec une confiance basse (0.3-0.5).
   - La confiance doit refléter la certitude de la classification (0.0-1.0).

Ta réponse doit être uniquement l'objet JSON, sans texte supplémentaire avant ou après.
"""
        
        # Construct user prompt in French
        user_prompt = f'Requête utilisateur : "{query}"\n\nClassifie cette requête.'
        
        # Call Groq API
        chat_completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        response_content = chat_completion.choices[0].message.content
        classification = json.loads(response_content)
        
        # Extract and validate the classification
        intent = classification.get("intent")
        try:
            confidence = float(classification.get("confidence", 0.0))
        except (ValueError, TypeError):
            logger.warning(f"Invalid confidence value in LLM response for query '{query}': {classification.get('confidence')}")
            confidence = 0.3
        
        reasoning = classification.get("reasoning", "L'LLM n'a pas fourni de justification.")
        
        # Validate intent is one of our categories
        if intent not in intent_categories_fr:
            logger.warning(f"Invalid intent category returned by LLM for query '{query}': {intent}")
            raise ValueError(f"Invalid intent category: {intent}")
        
        # Validate confidence is between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return intent, confidence, reasoning
        
    except Exception as e:
        logger.error(f"Error in classify_intent for query '{query}': {e}", exc_info=True)
        # Fallback to generic search on any error
        return (
            "generic_search",
            0.5,
            "Erreur lors de la classification de l'intention. Recherche générique effectuée."
        ) 