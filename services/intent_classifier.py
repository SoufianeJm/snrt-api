import os
import json
import logging
from typing import Tuple
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define French intent categories with rich descriptions
intent_categories_fr = {
    "match_score": """L'utilisateur recherche des informations sur des matchs passés. Cela peut inclure :
- Des résultats de matchs spécifiques (scores, statistiques)
- Des rediffusions ou résumés vidéo de matchs
- Des articles d'analyse post-match
- Des références à des années ou dates passées
- Des noms d'équipes ou de compétitions
Cette intention couvre tous les contenus liés aux matchs déjà joués, qu'ils soient sous forme de vidéos, articles ou données de match.""",

    "match_schedule": """L'utilisateur s'intéresse à la programmation des matchs à venir. Cela inclut :
- Les dates et heures des prochains matchs
- Les chaînes de diffusion
- Les compétitions à venir
- Les équipes qui vont s'affronter
Cette intention concerne la planification et l'information sur les événements sportifs futurs.""",

    "latest_news": """L'utilisateur cherche des informations actuelles et récentes. Cela peut être :
- Des actualités sportives du jour
- Des dernières nouvelles sur les équipes
- Des informations sur les transferts
- Des mises à jour sur les compétitions en cours
Cette intention couvre tout le contenu d'actualité récent, qu'il soit sportif ou général.""",

    "program_information": """L'utilisateur recherche des informations sur la programmation TV. Cela inclut :
- Des émissions de télévision
- Des documentaires
- Des magazines sportifs
- Des émissions spéciales
Cette intention concerne le contenu programmé régulièrement, distinct des matchs en direct ou des actualités.""",

    "generic_search": """L'utilisateur effectue une recherche générale ou ambiguë qui ne correspond pas clairement aux autres catégories. Cela peut être :
- Des requêtes trop vagues
- Des questions générales sur le sport
- Des recherches qui touchent plusieurs catégories
Cette intention sert de fallback pour les requêtes qui ne rentrent pas dans les autres catégories."""
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
        system_prompt = """Tu es un classificateur d'intention expert pour SNRT (Société Nationale de Radiodiffusion et de Télévision du Maroc). 
Ta tâche est d'analyser la requête de l'utilisateur et de déterminer laquelle des descriptions d'intention ci-dessous correspond le mieux à son objectif.

Tu dois renvoyer ta réponse sous forme d'objet JSON avec trois clés :
- "intent" : l'identifiant de la catégorie (doit être l'une des clés suivantes: {', '.join(intent_categories_fr.keys())})
- "confidence" : un nombre décimal entre 0.0 et 1.0
- "reasoning" : une brève explication en français pour ta classification

Voici les intentions disponibles avec leurs descriptions détaillées :
"""
        for key, desc in intent_categories_fr.items():
            system_prompt += f"\n{key}:\n{desc}\n"

        system_prompt += """
Voici quelques exemples pour te guider :

Exemple 1:
Requête utilisateur : "Je veux revoir le match du Wydad contre Raja de l'année dernière"
Réponse JSON attendue :
{
  "intent": "match_score",
  "confidence": 0.9,
  "reasoning": "L'utilisateur souhaite revoir un match passé (Wydad vs Raja, l'année dernière), ce qui correspond à une recherche de résultats ou de rediffusions."
}

Exemple 2:
Requête utilisateur : "C'est quand le prochain match de l'équipe nationale ?"
Réponse JSON attendue :
{
  "intent": "match_schedule",
  "confidence": 0.95,
  "reasoning": "L'utilisateur demande la date d'un futur match de l'équipe nationale, indiquant un intérêt pour la programmation à venir."
}

Exemple 3:
Requête utilisateur : "actualités sportives aujourd'hui"
Réponse JSON attendue :
{
  "intent": "latest_news",
  "confidence": 0.85,
  "reasoning": "L'utilisateur cherche les informations sportives du jour, ce qui correspond à une recherche d'actualités récentes."
}

Exemple 4:
Requête utilisateur : "Quand passe le magazine sportif hebdomadaire ?"
Réponse JSON attendue :
{
  "intent": "program_information",
  "confidence": 0.9,
  "reasoning": "L'utilisateur s'intéresse à la programmation d'une émission régulière, ce qui correspond à une recherche d'information sur la programmation TV."
}

Exemple 5:
Requête utilisateur : "tout sur le football marocain"
Réponse JSON attendue :
{
  "intent": "generic_search",
  "confidence": 0.6,
  "reasoning": "La requête est trop générale et pourrait concerner des actualités, des matchs, ou des programmes. Une recherche générique est plus appropriée."
}

Ta réponse doit être uniquement l'objet JSON, sans texte supplémentaire avant ou après."""
        
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