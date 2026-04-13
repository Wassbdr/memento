# Scenarios realistes de cas d'utilite Memento

Objectif: verifier que Memento fonctionne bien sur des situations plausibles du quotidien, avec des profils differents et des niveaux d'anxiete variables.

Convention:
- Chaque scenario possede un identifiant unique (UC-xxx).
- Les scenarios reutilisent les snapshots deja presents dans `experimentations/poc_test_cases.json`.
- Le test se fait sur la capacite de reassurance, de reorientation et de rappel autobiographique utile.

## UC-001 - Rose, confusion du matin avant les medicaments

Patient: Rose (domicile, aide a domicile en semaine)

Contexte:
- Lundi matin, 08:55
- Rose est dans la cuisine, semble anxieuse

Phrase patient:
- "Je suis perdue ce matin, Sophie est ou ?"

Utilite Memento attendue:
- Reassurer d'abord
- Rappeler que Rose est chez elle (repere domicile)
- Donner un repere immediat: Sophie + routine medicaments du matin

## UC-002 - Rose, identification d'un proche apres une visite

Patient: Rose

Contexte:
- Jeudi matin, lendemain d'une visite
- Rose se souvient d'une interaction mais pas du nom

Phrase patient:
- "Le jeune homme venu hier pour la tele, c'etait qui ?"

Utilite Memento attendue:
- Recuperer l'episode pertinent (visite de Lucas)
- Nommer Lucas et son lien familial
- Eviter une reponse floue ou generique

## UC-003 - Rose, crainte d'abandon avant le dejeuner du dimanche

Patient: Rose

Contexte:
- Dimanche 11:50
- Rose doute de la visite de sa fille

Phrase patient:
- "Claire m'a oubliee aujourd'hui ?"

Utilite Memento attendue:
- Rassurer explicitement
- Reancrer dans la routine "Dejeuner du dimanche"
- Donner un repere temporel proche et concret

## UC-004 - Jean, angoisse de fin d'apres-midi en residence

Patient: Jean (unite protegee)

Contexte:
- Mardi 16:25
- Moment sensible de fin de journee

Phrase patient:
- "Je dois rentrer chez moi tout de suite."

Utilite Memento attendue:
- Rappeler le lieu actuel sans confrontation brutale
- Mentionner le prochain rituel apaisant (cafe)
- Introduire un proche de confiance si pertinent

## UC-005 - Jean, besoin de valorisation identitaire

Patient: Jean

Contexte:
- Matinee calme
- Jean cherche une validation de son identite passee

Phrase patient:
- "J'etais bon mecanicien ou je me trompe ?"

Utilite Memento attendue:
- Recuperer un souvenir identitaire positif (atelier/voitures)
- Soutenir l'estime de soi
- Reponse concise et concrete

## UC-006 - Jean, anticipation du traitement du soir

Patient: Jean

Contexte:
- 18:20, avant le traitement

Phrase patient:
- "Qui vient ce soir pour mes comprimes ?"

Utilite Memento attendue:
- Nommer Ines clairement
- Rappeler l'ordre simple des etapes (apres le cafe)
- Maintenir un ton rassurant

## UC-007 - Amina, peur d'etre seule apres la sieste

Patient: Amina (vit avec son mari)

Contexte:
- 16:55, juste avant le the

Phrase patient:
- "Samir est sorti ? Je suis seule ?"

Utilite Memento attendue:
- Reassurer sur la presence du conjoint
- Rappeler la routine de fin d'apres-midi
- Donner un repere sensoriel concret (plateau de the, salon)

## UC-008 - Amina, recherche du prochain appel de sa fille

Patient: Amina

Contexte:
- Mardi 18:50
- Amina cherche Leila

Phrase patient:
- "Je peux parler a Leila ce soir ?"

Utilite Memento attendue:
- Rappeler la routine d'appel video du mardi
- Donner un horizon temporel proche
- Eviter d'ajouter des details inventes

## UC-009 - Amina, confusion de planning sur la sortie au marche

Patient: Amina

Contexte:
- Mercredi 10:00
- Amina pense que la sortie est immediate

Phrase patient:
- "On va au marche maintenant ?"

Utilite Memento attendue:
- Repondre sans brusquerie
- Recaler vers le bon repere (samedi matin)
- Rappeler que Samir accompagne la sortie
