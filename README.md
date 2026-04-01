# Memento

Memento est une prothèse cognitive vocale destinée aux patients atteints de la maladie d’Alzheimer. L’idée centrale est simple : donner au patient un accès permanent, à portée de voix, à sa propre mémoire personnelle. Un compagnon vocal qui connaît ses proches, ses habitudes, ses souvenirs et ses routines, et peut y répondre à tout moment de manière naturelle et rassurante — sans écran, sans bouton, sans aucune interaction cognitive requise.
La mémoire du patient est alimentée de deux façons : par les aidants et la famille via une application dédiée, et de manière automatique en captant le contexte des interactions quotidiennes — visites, conversations, émotions — avec le consentement de chacun. Ce contexte vivant est stocké dans un graphe de mémoire personnalisé et affectif, bien au-delà de ce que peut offrir n’importe quel assistant généraliste.
L’objectif n’est pas de guérir, mais de réduire les moments de confusion et d’angoisse, de préserver l’autonomie le plus longtemps possible, et de soulager les aidants. Dans un second temps, une couche visuelle — idéalement via des lunettes connectées — permettra à l’assistant de reconnaître les visages en temps réel et d’enrichir encore davantage le contexte de chaque interaction.

# TODO:
Micro → Whisper + VAD
              ↓
        LlamaIndex
              ↓
   ChromaDB ←→ Neo4j
              ↓
    Ministral 3 8B
              ↓
         Voxtral TTS
              ↓
           Haut-parleur