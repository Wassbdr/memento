# Memento Use-Case Evaluation Report

- Total scenarios: 9
- Deterministic pass: 1/9
- Final pass (deterministic + judge): 1/9
- Avg final score: 61.22/100
- Avg factual grounding (judge): 2.56/5
- Avg reassurance tone (judge): 2.44/5

## Per scenario

| Case | Deterministic | Judge | Final | Score |
|---|---:|---:|---:|---:|
| UC-001 | FAIL | FAIL | FAIL | 50.00 |
| UC-002 | FAIL | FAIL | FAIL | 44.00 |
| UC-003 | FAIL | FAIL | FAIL | 56.00 |
| UC-004 | FAIL | FAIL | FAIL | 60.00 |
| UC-005 | FAIL | FAIL | FAIL | 56.00 |
| UC-006 | FAIL | FAIL | FAIL | 65.50 |
| UC-007 | FAIL | FAIL | FAIL | 52.00 |
| UC-008 | FAIL | FAIL | FAIL | 67.50 |
| UC-009 | PASS | PASS | PASS | 100.00 |

## Findings

### UC-001 - FAIL
- Goal: Gerer une confusion matinale avec repere domicile + routine immediate.
- Final score: 50.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_start_with_reassurance -> first sentence should contain reassuring phrasing
- Deterministic fail: response.must_include_temporal_anchor -> answer should include a near-term temporal anchor
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 1.0, "clarity_concision": 3.0, "cognitive_load": 3.0, "clinical_usefulness": 1.0}
- Judge violations: ["Le systeme a recupere des informations pertinentes sur Sophie Benali (aide a domicile, vient le matin en semaine) mais l'assistant ne les utilise pas pour repondre", 'Ne commence pas par une reassurance comme requis par les response_requirements', "N'inclut pas d'ancrage temporel comme requis par les response_requirements", "La formulation 'Je n'ai pas assez d'informations fiables' peut augmenter l'anxiete de la patiente desorientee", "L'offre 'si vous voulez' reporte la reassurance sur la patiente au lieu de la fournir immediatement"]
- Judge improvement: L'assistant disposait de trois souvenirs directement pertinents : Sophie Benali est l'aide a domicile de Rose, elle vient le matin en semaine pour le petit-dejeuner et les medicaments, et un episode recent (il y a 3 jours) mentionnant Sophie accompagnant Rose. La reponse aurait du utiliser ces informations pour fournir immediatement : (1) une reassurance empathique au debut, (2) un ancrage temporel ('Nous sommes lundi matin, il est environ 9h'), (3) la localisation de Sophie ('Sophie Benali, votre aide a domicile, vient vous aider le matin en semaine. Elle devrait arriver bientot pour le petit-dejeuner'). Exemple de reponse adaptee : 'Ne vous inquietez pas, Rose. Vous etes dans votre appartement rue des Lilas a Lyon. Nous sommes lundi matin. Sophie Benali, votre aide a domicile, vient vous aider le matin en semaine. Elle devrait arriver bientot pour le petit-dejeuner et les medicaments.'

### UC-002 - FAIL
- Goal: Identifier correctement un proche mentionne de facon indirecte.
- Final score: 44.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: context.episodes_any -> expected any=['Passage de Lucas apres les cours']; matched=[]; actual=[]
- Deterministic fail: context.places_any -> expected any=['Salon']; matched=[]; actual=['Cuisine', 'Cuisine', 'Cuisine', 'Cuisine', 'Cuisine']
- Deterministic fail: response.must_include_relationship_hint -> answer should include relationship wording
- Deterministic fail: response.must_avoid_generic_answer -> answer should mention concrete memory anchors
- Judge overall: False
- Judge scores: {"factual_grounding": 4.0, "reassurance_tone": 3.0, "clarity_concision": 4.0, "cognitive_load": 4.0, "clinical_usefulness": 2.0}
- Judge improvement: Le modele a echoue a exploiter le contexte disponible. Le check 'context.trusted_people_any' montre que 'Lucas Martin' etait present dans les donnees, mais le modele ne l'a pas utilise. Il aurait pu dire : 'C'etait peut-etre Lucas, votre petit-fils ? Il passe souvent vous voir apres l'ecole.' Cette reponse aurait ete concrete, ancree dans la memoire relationnelle, et aurait repondu au besoin emotionnel de Rose sans risquer d'erreur factuelle puisque Lucas Martin etait bien dans le contexte. L'echec principal est clinique : l'opportunite de fournir un repere identitaire significatif a ete manquée.

### UC-003 - FAIL
- Goal: Rassurer sur une visite familiale proche sans surcharge.
- Final score: 56.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_include_future_near_anchor -> answer should project a near future milestone
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 1.0, "clarity_concision": 2.0, "cognitive_load": 2.0, "clinical_usefulness": 1.0}
- Judge violations: ["L'assistant possede pourtant les informations necessaires (routine Dejeuner du dimanche a 12h30, episode recent positif avec Claire le 29/03) mais refuse de les utiliser pour rassurer", 'La reponse ne projette pas dans le futur proche comme requis (il est 11h50, Claire arrive a 12h30 - ceci est un ancrage futur immediat evident)', "Le ton est passif-defensif : l'assistant demande a la patiente si elle 'veut' un repere rassurant au lieu de le fournir directement", "Non-utilisation des memoires disponibles pour adresser l'anxiete de la patiente concernant l'oubli", "La patiente exprime une peur d'etre oubliee et la reponse ne la reconnait pas explicitement"]
- Judge improvement: L'assistant devrait utiliser les memoires disponibles pour rassurer proactivement Rose. Il est 11h50 un dimanche - Claire vient regulierement le dimanche a 12h30 (routine Dejeuners du dimanche). Un exemple de reponse adaptee : 'Claire n'est pas en retard, elle arrive le dimanche midi comme d'habitude. Il est encore tot, vous avez le temps de preparer la table. Vous vous etes bien amusees ensemble le 29 mars dernier.' Cette reponse utiliserait l'ancrage futur proche (12h30), reconnaitrait l'emotion de la patiente, et s'appuierait sur un souvenir positif recent.

### UC-004 - FAIL
- Goal: Desescalader une angoisse de fin d'apres-midi en institution.
- Final score: 60.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_include_immediate_next_step -> answer should include an immediate next step
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 2.0, "clarity_concision": 3.0, "cognitive_load": 2.0, "clinical_usefulness": 1.0}
- Judge violations: ['Le response.must_include_immediate_next_step a echoue: la reponse ne propose aucun pas immediat concret au patient', "Non-utilisation des informations contextuelles disponibles: les memoires recuperees contiennent des ancres rassurantes pertinentes (residence des Tilleuls, presence de Ines, cafe de l'apres-midi) qui auraient pu etre utilisees"]
- Judge improvement: La reponse devrait utiliser les memoires disponibles pour proposer immediatement un repere rassurant concret. Exemple: 'Vous etes a la residence des Tilleuls, dans votre chambre. C'est l'heure du cafe de l'apres-midi. Ines est la pour vous accompagner.' Cela aurait rempli l'exigence must_include_immediate_next_step et maximise l'utilite clinique pour desescalader l'angoisse.

### UC-005 - FAIL
- Goal: Soutenir l'identite personnelle via souvenir positif specifique.
- Final score: 56.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: context.emotions_any -> expected any=['fierte']; matched=[]; actual=[]
- Deterministic fail: response.must_include_identity_support -> answer should reinforce identity
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 2.0, "clarity_concision": 4.0, "cognitive_load": 4.0, "clinical_usefulness": 1.0}
- Judge violations: ["Le patient demande un soutien d'identité sur son passé de mécanicien. Les souvenirs récupérés contiennent 'Discussion autour des voitures anciennes' et 'Atelier souvenirs mecanique', informations directement pertinentes. L'assistant ignore ces données disponibles et refuse de répondre, manquant l'objectif clinique principal."]
- Judge improvement: L'assistant aurait dû exploiter les souvenirs disponibles. Exemple de réponse appropriée : 'Non, Monsieur Jean, vous étiez bien mécanicien. Vous aimiez en parler, surtout quand on discutait des voitures anciennes.' Cette réponse aurait confirmé l'identité, utilisé les données contextuelles (émotion fierté possible), et répondu au besoin émotionnel du patient.

### UC-006 - FAIL
- Goal: Preparer sereinement le traitement du soir.
- Final score: 65.50/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_include_person_name -> answer should include a known person name
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 3.0, "clarity_concision": 4.0, "cognitive_load": 3.0, "clinical_usefulness": 2.0}
- Judge violations: ["L'assistant disposait d'informations fiables sur Ines Dupont (infirmiere referente qui apporte le traitement du soir a 18h30) mais ne les a pas utilisees", "Le requisitoire 'must_include_person_name' n'est pas respecte (check echoue)", 'La reponse ne repond pas a la question immediate du patient concernant le traitement du soir']
- Judge improvement: L'assistant aurait dû utiliser les informations disponibles dans les memoires : Ines Dupont est l'infirmiere referente de Jean et apporte le traitement du soir a 18h30. Une reponse adaptee aurait ete : 'C'est Ines, votre infirmiere, qui viendra vous apporter vos comprimes dans votre chambre, comme d'habitude, vers 18h30.' Cette reponse aurait integre le nom de personne requis, utilise le contexte factuel fiable, et rassure le patient en precisant le moment et le lieu du soin.

### UC-007 - FAIL
- Goal: Rassurer apres sieste sur la presence du conjoint.
- Final score: 52.00/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_include_spousal_anchor -> answer should anchor around spouse
- Deterministic fail: response.must_include_concrete_sensory_cue -> answer should include concrete sensory cue
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 2.0, "clarity_concision": 3.0, "cognitive_load": 3.0, "clinical_usefulness": 1.0}
- Judge violations: ["L'assistant declare ne pas avoir d'informations fiables alors que le contexte contient des informations certaines sur Samir (mari d'Amina, vit avec elle, importance emotionnelle 0.99). Cette hesitation est une forme de sous-utilisation des donnees disponibles.", "Le ton est deflectif et non rassurant. La reponse 'Je peux vous rappeler un repere rassurant si vous voulez' ne repond pas a l'anxiete immediate de la patiente qui demande si elle est seule.", "Les exigences de reponse ne sont pas satisfaites: absence d'ancrage au conjoint (must_include_spousal_anchor) et absence de signal sensoriel concret (must_include_concrete_sensory_cue)."]
- Judge improvement: L'assistant devrait utiliser les informations disponibles pour rassurer immediatement Amina. Exemple de reponse adaptee: 'Samir est la, il est dans la cuisine en train de preparer le the. Vous etes dans le salon, c'est l'heure du the de fin d'apres-midi. Vous n'etes pas seule.' Cette reponse ancre au conjoint (Samir est la), inclut un signal sensoriel concret (preparation du the, vapeur, odeur), utilise la routine connue (the de fin d'apres-midi), et rassure directement sur la presence. L'assistant ne doit pas hesiter a utiliser des informations fiables pour calmer l'anxiete.

### UC-008 - FAIL
- Goal: Donner un repere clair sur le prochain appel de la fille.
- Final score: 67.50/100
- Answer: Je n'ai pas assez d'informations fiables pour repondre a cela. Je peux vous rappeler un repere rassurant si vous voulez.
- Deterministic fail: response.must_include_time_projection -> answer should project a concrete time reference
- Judge overall: False
- Judge scores: {"factual_grounding": 2.0, "reassurance_tone": 3.0, "clarity_concision": 4.0, "cognitive_load": 4.0, "clinical_usefulness": 2.0}
- Judge violations: ["L'assistant ignore une routine validee et disponible (Appel video avec Leila, chaque mardi a 19h00) qui permettrait de repondre a la question du patient. La memoire indique explicitement l'horaire et la strategie d'accompagnement associee."]
- Judge improvement: L'assistant disposait d'une routine validee indiquant qu'Amina recoit un appel video de Leila chaque mardi a 19h00. Le 7 avril etant un mardi, il etait tout a fait approprie de dire : 'Leila vous appellera ce soir a 19h, comme d'habitude.' Cette reponse aurait ete factuellement fondee, rassurante et directement utile. L'assistant ne doit pas sous-estimer la valeur des routines et episodes recuperes, qui constituent des informations fiables pour ce type de question.

### UC-009 - PASS
- Goal: Recadrer une confusion de planning sans frustrer.
- Final score: 100.00/100
- Answer: Amina, le marche, c'est un tres beau souvenir avec Samir.

Mais aujourd'hui, je ne sais pas quel jour on est. Je ne peux pas vous dire si c'est le moment d'y aller.

Ce que je sais, c'est que le marche, c'est le **samedi matin a 09h30**. Samir vous accompagne. Et le cabas raye est bien pres de la porte.

Quand le samedi viendra, Samir sera la pour vous accompagner. C'est une belle sortie, tous les deux.

Pour l'instant, vous etes bien chez vous, avec Samir. Est-ce que vous aimeriez prendre le the dans le salon, pres du tapis bleu ?
- Deterministic checks: all pass
- Judge overall: True
- Judge scores: {"factual_grounding": 5.0, "reassurance_tone": 5.0, "clarity_concision": 5.0, "cognitive_load": 5.0, "clinical_usefulness": 5.0}
- Judge improvement: La réponse est déjà très bien calibrée. Une seule suggestion mineure : la phrase 'je ne sais pas quel jour on est' pourrait être légèrement reformulée en 'je ne suis pas sûre du jour aujourd'hui' pour éviter toute ambiguïté, bien que la formulation actuelle reste acceptable et non anxiogène.
